"""
Sentence-Level HMAC-Net Training Script

使用隱含 Aspect 發現模組訓練句子級別情感分析模型

特點:
- 支援任何資料集（通過 dataset_config.json）
- 保留 PMAC 和 IARM 創新
- 自動發現隱含的語義面向
- 可解釋性分析

使用範例:
    python experiments/train_sentence_level.py --dataset imdb --use_pmac --use_iarm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from transformers import AutoTokenizer
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset_manager import DatasetManager
from data.sentence_level_dataset import create_sentence_level_dataloaders
from models.hmacnet_sentence_level import HMACNetSentenceLevel


def train_sentence_level_model(args):
    """訓練句子級別模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 創建時間戳資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent

    # 實驗名稱
    exp_name = f"{args.dataset}_{'pmac' if args.use_pmac else 'nopmac'}_{'iarm' if args.use_iarm else 'noiarm'}"
    exp_name += f"_asp{args.num_implicit_aspects}_drop{args.dropout}"

    # 時間戳實驗資料夾
    exp_dir = project_root / 'results' / 'sentence_level' / f"{timestamp}_{exp_name}"
    checkpoints_dir = exp_dir / 'checkpoints'
    visualizations_dir = exp_dir / 'visualizations'
    reports_dir = exp_dir / 'reports'

    # 創建所有資料夾
    for dir_path in [checkpoints_dir, visualizations_dir, reports_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n實驗資料夾: {exp_dir}")

    # 載入資料集
    print("\n" + "="*80)
    print("載入資料集")
    print("="*80)

    dataset_manager = DatasetManager()

    try:
        train_samples, val_samples, test_samples = dataset_manager.load_dataset(
            args.dataset,
            val_split_ratio=args.val_split_ratio,
            limit=args.limit
        )
    except Exception as e:
        print(f"\n錯誤: {e}")
        print("\n可用的資料集:")
        for code in dataset_manager.list_datasets():
            print(f"  - {code}")
        return

    # 獲取資料集資訊
    dataset_info = dataset_manager.get_dataset_info(args.dataset)
    num_classes = dataset_info['num_classes']
    domain = dataset_info['domain']

    print(f"\n資料集資訊:")
    print(f"  名稱: {dataset_info['name']}")
    print(f"  領域: {domain}")
    print(f"  類別數: {num_classes}")
    print(f"  標籤: {dataset_info['label_names']}")

    # 載入 tokenizer
    print(f"\n載入 {args.bert_model} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # 創建 DataLoaders
    print("\n創建 DataLoaders...")
    train_loader, val_loader, test_loader = create_sentence_level_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_text_len,
        num_implicit_aspects=args.num_implicit_aspects,
        domain=domain
    )

    # 創建模型
    print("\n" + "="*80)
    print("創建句子級別 HMAC-Net")
    print("="*80)

    model = HMACNetSentenceLevel(
        bert_model_name=args.bert_model,
        freeze_bert=args.freeze_bert,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        dropout=args.dropout,
        num_implicit_aspects=args.num_implicit_aspects,
        domain=domain,
        use_pmac=args.use_pmac,
        pmac_composition_mode=args.pmac_mode,
        use_iarm=args.use_iarm,
        iarm_relation_mode=args.iarm_mode,
        iarm_num_heads=args.iarm_heads,
        iarm_num_layers=args.iarm_layers,
        fusion_strategy=args.fusion_strategy
    ).to(device)

    print(f"\n模型配置:")
    print(f"  BERT: {args.bert_model}")
    print(f"  隱含 Aspects 數量: {args.num_implicit_aspects}")
    print(f"  領域: {domain}")
    print(f"  PMAC: {'Enabled' if args.use_pmac else 'Disabled'} ({args.pmac_mode if args.use_pmac else 'N/A'})")
    print(f"  IARM: {'Enabled' if args.use_iarm else 'Disabled'} ({args.iarm_mode if args.use_iarm else 'N/A'})")
    print(f"  融合策略: {args.fusion_strategy}")

    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 學習率調度器
    scheduler = None
    if args.use_scheduler:
        total_steps = len(train_loader) * args.epochs // args.accumulation_steps
        warmup_steps = int(args.warmup_ratio * total_steps)

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

    # 損失函數
    criterion = nn.CrossEntropyLoss()

    # 訓練
    print("\n" + "="*80)
    print("開始訓練")
    print("="*80)

    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0

    # 記錄訓練歷史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': [],
        'epochs': []
    }

    for epoch in range(args.epochs):
        # 訓練階段
        model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, batch in enumerate(progress_bar):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits, _ = model(text_ids, text_mask)

            # Loss
            loss = criterion(logits, labels)
            loss = loss / args.accumulation_steps

            # Backward
            loss.backward()

            # 梯度累積
            if (batch_idx + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                if scheduler is not None:
                    scheduler.step()

            train_loss += loss.item() * args.accumulation_steps
            train_steps += 1

            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item() * args.accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_train_loss = train_loss / train_steps

        # 驗證階段
        val_metrics = evaluate_model(model, val_loader, device, criterion)

        # 記錄歷史
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(val_metrics['loss']))
        history['val_accuracy'].append(float(val_metrics['accuracy']))
        history['val_f1'].append(float(val_metrics['f1']))

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1: {val_metrics['f1']:.4f}")

        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存最佳模型
            save_path = checkpoints_dir / f'best_model_epoch{epoch+1}_f1_{best_val_f1:.4f}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"  [SAVED] Best model: {save_path.name}")
        else:
            patience_counter += 1
            print(f"  Patience: {patience_counter}/{args.patience}")

            if patience_counter >= args.patience:
                print("\nEarly stopping!")
                break

    # 測試階段
    print("\n" + "="*80)
    print("測試集評估")
    print("="*80)

    # 載入最佳模型
    best_model_path = checkpoints_dir / f'best_model_epoch{best_epoch}_f1_{best_val_f1:.4f}.pt'
    model.load_state_dict(torch.load(best_model_path))

    test_metrics = evaluate_model(model, test_loader, device, criterion, detailed=True)

    print(f"\n測試集結果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")

    if 'classification_report' in test_metrics:
        print(f"\n分類報告:")
        print(test_metrics['classification_report'])

    # 保存結果
    results = {
        'args': vars(args),
        'dataset_info': dataset_info,
        'best_val_f1': float(best_val_f1),
        'best_epoch': best_epoch,
        'history': history,
        'test_metrics': {k: v for k, v in test_metrics.items() if k != 'classification_report'}
    }

    results_path = reports_dir / 'experiment_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n結果已保存: {results_path}")

    # 保存摘要
    summary_path = reports_dir / 'experiment_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"句子級別 HMAC-Net 實驗摘要 - {timestamp}\n")
        f.write("="*80 + "\n\n")

        f.write(f"資料集: {dataset_info['name']}\n")
        f.write(f"領域: {domain}\n")
        f.write(f"類別數: {num_classes}\n\n")

        f.write("模型配置:\n")
        f.write(f"  隱含 Aspects: {args.num_implicit_aspects}\n")
        f.write(f"  PMAC: {'Enabled' if args.use_pmac else 'Disabled'} ({args.pmac_mode})\n")
        f.write(f"  IARM: {'Enabled' if args.use_iarm else 'Disabled'} ({args.iarm_mode})\n")
        f.write(f"  融合策略: {args.fusion_strategy}\n\n")

        f.write("訓練結果:\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Best Val F1: {best_val_f1:.4f}\n")
        f.write(f"  Test Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Test F1: {test_metrics['f1']:.4f}\n")

    print(f"實驗摘要已保存: {summary_path}")
    print(f"\n[COMPLETE] 所有結果已保存至: {exp_dir}")


def evaluate_model(model, dataloader, device, criterion, detailed=False):
    """評估模型"""
    model.eval()

    all_preds = []
    all_labels = []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward
            logits, _ = model(text_ids, text_mask)

            # Loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1

            # 預測
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 計算指標
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / num_batches

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    if detailed:
        report = classification_report(all_labels, all_preds)
        metrics['classification_report'] = report

    return metrics


def main():
    parser = argparse.ArgumentParser(description='訓練句子級別 HMAC-Net')

    # 資料集參數
    parser.add_argument('--dataset', type=str, required=True,
                       help='資料集代號（見 dataset_config.json）')
    parser.add_argument('--val_split_ratio', type=float, default=0.1,
                       help='驗證集分割比例')
    parser.add_argument('--limit', type=int, default=None,
                       help='限制載入的樣本數（用於快速測試）')
    parser.add_argument('--max_text_len', type=int, default=256,
                       help='最大文本長度')

    # 模型參數
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                       help='BERT 模型名稱')
    parser.add_argument('--freeze_bert', action='store_true',
                       help='是否凍結 BERT')
    parser.add_argument('--hidden_dim', type=int, default=768,
                       help='隱藏層維度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout 比率')
    parser.add_argument('--num_implicit_aspects', type=int, default=5,
                       help='隱含 aspects 數量')

    # PMAC 參數
    parser.add_argument('--use_pmac', action='store_true', default=False,
                       help='是否使用 PMAC')
    parser.add_argument('--pmac_mode', type=str, default='selective',
                       choices=['sequential', 'pairwise', 'attention', 'selective'],
                       help='PMAC 組合模式')

    # IARM 參數
    parser.add_argument('--use_iarm', action='store_true', default=False,
                       help='是否使用 IARM')
    parser.add_argument('--iarm_mode', type=str, default='transformer',
                       choices=['transformer', 'gat', 'bilinear'],
                       help='IARM 關係模式')
    parser.add_argument('--iarm_heads', type=int, default=4,
                       help='IARM 注意力頭數')
    parser.add_argument('--iarm_layers', type=int, default=2,
                       help='IARM 層數')

    # 融合參數
    parser.add_argument('--fusion_strategy', type=str, default='weighted_pooling',
                       choices=['mean', 'max', 'weighted_pooling', 'attention'],
                       help='Aspect 融合策略')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                       help='訓練輪數')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                       help='梯度累積步數')
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                       help='是否使用學習率調度器')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                       help='Warmup 步數比例')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("句子級別 HMAC-Net 訓練")
    print("="*80)
    print("\n配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    train_sentence_level_model(args)


if __name__ == '__main__':
    main()
