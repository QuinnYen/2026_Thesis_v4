"""
Multi-Aspect HMAC-Net Training Script

訓練句子級別的多面向情感分析模型
真正發揮 PMAC 和 IARM 創新模組的作用
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.semeval_multiaspect import load_multiaspect_data
from data.multiaspect_dataset import create_multiaspect_dataloaders
from models.bert_embedding import BERTForABSA
from models.aaha_enhanced import AAHAEnhanced
from models.pmac_enhanced import PMACMultiAspect
from models.iarm_enhanced import IARMMultiAspect
from models.base_model import BaseModel, MLP
from utils.focal_loss import get_loss_function


class HMACNetMultiAspect(BaseModel):
    """
    Multi-Aspect HMAC-Net

    支持句子級別的多面向建模
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,  # 保持 BERT 維度，不降維
        num_classes: int = 3,
        dropout: float = 0.1,
        # PMAC 參數
        use_pmac: bool = True,
        pmac_composition_mode: str = 'sequential',  # 'sequential', 'pairwise', 'attention'
        # IARM 參數
        use_iarm: bool = True,
        iarm_relation_mode: str = 'transformer',  # 'transformer', 'gat', 'bilinear'
        iarm_num_heads: int = 4,
        iarm_num_layers: int = 2
    ):
        super(HMACNetMultiAspect, self).__init__()

        self.use_pmac = use_pmac
        self.use_iarm = use_iarm
        self.hidden_dim = hidden_dim

        # BERT 嵌入
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert_absa.hidden_size

        # 投影層（如果需要）
        if bert_hidden_size != hidden_dim:
            self.text_projection = nn.Linear(bert_hidden_size, hidden_dim)
            self.aspect_projection = nn.Linear(bert_hidden_size, hidden_dim)
        else:
            self.text_projection = nn.Identity()
            self.aspect_projection = nn.Identity()

        # AAHA 模組（為每個 aspect 提取上下文）
        self.aaha = AAHAEnhanced(
            hidden_dim=hidden_dim,
            aspect_dim=hidden_dim,
            word_attention_dims=[hidden_dim // 2],
            phrase_attention_dims=[hidden_dim // 2],
            sentence_attention_dims=[hidden_dim],
            attention_dropout=0.0,
            output_dropout=dropout
        )

        # PMAC Multi-Aspect（真正的多面向組合）
        if use_pmac:
            self.pmac = PMACMultiAspect(
                input_dim=hidden_dim,
                fusion_dim=hidden_dim,
                num_composition_layers=2,
                hidden_dim=128,
                dropout=dropout,
                composition_mode=pmac_composition_mode
            )

        # IARM Multi-Aspect（真正的面向間關係建模）
        if use_iarm:
            self.iarm = IARMMultiAspect(
                input_dim=hidden_dim,
                relation_dim=hidden_dim,
                num_heads=iarm_num_heads,
                num_layers=iarm_num_layers,
                dropout=dropout,
                relation_mode=iarm_relation_mode
            )

        # 分類器（為每個 aspect 獨立分類）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        參數:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            aspect_input_ids: [batch, max_aspects, aspect_len]
            aspect_attention_mask: [batch, max_aspects, aspect_len]
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
        """
        batch_size, max_aspects, aspect_len = aspect_input_ids.shape

        # 1. BERT 編碼文本（一次）
        text_emb = self.bert_absa.bert_embedding(
            text_input_ids,
            attention_mask=text_attention_mask
        )  # [batch, seq_len, bert_dim]

        text_hidden = self.text_projection(text_emb)  # [batch, seq_len, hidden_dim]

        # 2. BERT 編碼每個 aspect
        aspect_hidden_list = []
        for i in range(max_aspects):
            # 只處理有效的 aspects
            if aspect_mask[:, i].any():
                asp_emb = self.bert_absa.bert_embedding(
                    aspect_input_ids[:, i, :],
                    attention_mask=aspect_attention_mask[:, i, :]
                )  # [batch, aspect_len, bert_dim]

                # 使用 [CLS] token
                asp_repr = asp_emb[:, 0, :]  # [batch, bert_dim]
                asp_hidden = self.aspect_projection(asp_repr)  # [batch, hidden_dim]
            else:
                asp_hidden = torch.zeros(batch_size, self.hidden_dim, device=text_hidden.device)

            aspect_hidden_list.append(asp_hidden)

        aspect_hiddens = torch.stack(aspect_hidden_list, dim=1)  # [batch, max_aspects, hidden_dim]

        # 3. AAHA - 為每個 aspect 提取上下文
        context_vectors = []
        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                ctx, _ = self.aaha(
                    text_hidden,
                    aspect_hiddens[:, i, :],
                    text_attention_mask.float()
                )
                context_vectors.append(ctx)
            else:
                context_vectors.append(torch.zeros(batch_size, self.hidden_dim, device=text_hidden.device))

        context_vectors = torch.stack(context_vectors, dim=1)  # [batch, max_aspects, hidden_dim]

        # 4. PMAC - 多面向組合（創新核心！）
        if self.use_pmac:
            composed_features = self.pmac(context_vectors, aspect_mask)
        else:
            composed_features = context_vectors

        # 5. IARM - 面向間關係建模（創新核心！）
        if self.use_iarm:
            enhanced_features, _ = self.iarm(composed_features, aspect_mask)
        else:
            enhanced_features = composed_features

        # 6. 分類
        logits = self.classifier(enhanced_features)  # [batch, max_aspects, 3]

        return logits


def generate_training_visualizations(results, save_dir):
    """
    生成所有訓練曲線圖表
    """
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")

    history = results['history']
    epochs = history['epochs']

    print("\n生成圖表...")

    # 1. 綜合指標圖（2x2）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Metrics Overview', fontsize=18, fontweight='bold')

    # 訓練損失
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 驗證準確率
    val_acc = history['val_accuracy']
    axes[0, 1].plot(epochs, val_acc, 'g-o', linewidth=2, markersize=5)
    best_epoch = np.argmax(val_acc) + 1
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Validation Accuracy (Best: Epoch {best_epoch})', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 驗證F1
    val_f1 = history['val_f1_macro']
    axes[1, 0].plot(epochs, val_f1, 'm-o', linewidth=2, markersize=5)
    best_epoch_f1 = np.argmax(val_f1) + 1
    axes[1, 0].axvline(x=best_epoch_f1, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Validation F1 Macro (Best: Epoch {best_epoch_f1})', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 每類F1
    f1_per_class = np.array(history['val_f1_per_class'])
    class_names = ['Negative', 'Neutral', 'Positive']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        axes[1, 1].plot(epochs, f1_per_class[:, i], marker='o', label=class_name,
                       color=color, linewidth=2, markersize=5)
    axes[1, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=11, loc='best')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'comprehensive_training_metrics.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {save_path.name}")
    plt.close()

    # 2. 每類別F1詳細曲線
    plt.figure(figsize=(12, 7))
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        f1_scores = f1_per_class[:, i]
        plt.plot(epochs, f1_scores, marker='o', label=f'{class_name} F1',
                color=color, linewidth=2, markersize=6)
        final_f1 = f1_scores[-1]
        plt.text(epochs[-1], final_f1, f' {final_f1:.3f}', fontsize=10,
                color=color, verticalalignment='center', fontweight='bold')

    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score', fontsize=14, fontweight='bold')
    plt.title('Per-Class F1 Score Curves', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_dir / 'per_class_f1_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  [SAVED] {save_path.name}")
    plt.close()

    # 3. 生成訓練報告
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Multi-Aspect HMAC-Net Training Report")
    report_lines.append("="*80)
    report_lines.append("")

    args = results['args']
    report_lines.append("Configuration:")
    report_lines.append(f"  Model: {args['bert_model']}")
    report_lines.append(f"  PMAC: {'Enabled' if args['use_pmac'] else 'Disabled'} ({args['pmac_mode']})")
    report_lines.append(f"  IARM: {'Enabled' if args['use_iarm'] else 'Disabled'} ({args['iarm_mode']})")
    report_lines.append(f"  Loss Type: {args.get('loss_type', 'ce')}")
    if args.get('loss_type') == 'focal':
        report_lines.append(f"  Focal Gamma: {args.get('focal_gamma', 2.0)}")
        report_lines.append(f"  Class Weights: {args.get('class_weights', [1.0, 1.0, 1.0])}")
    report_lines.append("")

    report_lines.append("Epoch-by-Epoch Metrics:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Epoch':<8} {'Train Loss':<12} {'Val Acc':<10} {'Val F1':<10} {'Neg F1':<10} {'Neu F1':<10} {'Pos F1':<10}")
    report_lines.append("-" * 80)

    for i, epoch in enumerate(epochs):
        train_loss = history['train_loss'][i]
        val_acc = history['val_accuracy'][i]
        val_f1 = history['val_f1_macro'][i]
        f1_class = history['val_f1_per_class'][i]
        report_lines.append(
            f"{epoch:<8} {train_loss:<12.4f} {val_acc:<10.4f} {val_f1:<10.4f} "
            f"{f1_class[0]:<10.4f} {f1_class[1]:<10.4f} {f1_class[2]:<10.4f}"
        )

    report_lines.append("-" * 80)
    report_lines.append("")

    # 測試結果
    test_metrics = results['test_metrics']
    report_lines.append("Test Results:")
    report_lines.append(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    report_lines.append(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    test_f1_per_class = test_metrics['f1_per_class']
    report_lines.append(f"  F1 per class:")
    report_lines.append(f"    Negative: {test_f1_per_class[0]:.4f}")
    report_lines.append(f"    Neutral:  {test_f1_per_class[1]:.4f}")
    report_lines.append(f"    Positive: {test_f1_per_class[2]:.4f}")
    report_lines.append("")
    report_lines.append("="*80)

    report_path = save_dir / 'training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"  [SAVED] {report_path.name}")


def compute_multi_aspect_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    aspect_mask: torch.Tensor,
    is_virtual: torch.Tensor = None,
    virtual_weight: float = 0.5
):
    """
    計算 multi-aspect 損失

    參數:
        logits: [batch, max_aspects, num_classes]
        labels: [batch, max_aspects] (-100 表示 ignore)
        aspect_mask: [batch, max_aspects] - bool
        is_virtual: [batch, max_aspects] - bool，標記虛擬 aspects
        virtual_weight: 虛擬 aspect 的損失權重

    返回:
        loss: scalar
    """
    batch_size, max_aspects, num_classes = logits.shape

    # Flatten
    logits_flat = logits.view(-1, num_classes)  # [batch * max_aspects, 3]
    labels_flat = labels.view(-1)  # [batch * max_aspects]

    # 計算損失（ignore_index=-100 會自動跳過）
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction='none')

    # 重塑回原形狀
    loss = loss.view(batch_size, max_aspects)

    # 如果有虛擬 aspect，降低其權重
    if is_virtual is not None:
        weights = torch.where(is_virtual, virtual_weight, 1.0)
        loss = loss * weights

    # 只在有效 aspects 上計算平均損失
    loss = (loss * aspect_mask.float()).sum() / aspect_mask.float().sum().clamp(min=1)

    return loss


def evaluate_multi_aspect(model, dataloader, device):
    """
    評估 multi-aspect 模型

    返回 aspect-level 指標（展開為獨立的 aspect-sentiment pairs）
    """
    model.eval()

    valid_preds = []
    valid_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            aspect_ids = batch['aspect_input_ids'].to(device)
            aspect_mask_input = batch['aspect_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)

            # Forward
            logits = model(text_ids, text_mask, aspect_ids, aspect_mask_input, aspect_mask)

            # 預測
            preds = torch.argmax(logits, dim=-1)  # [batch, max_aspects]

            # 直接展開為 aspect-level（過濾無效和虛擬 aspects）
            preds_cpu = preds.cpu()
            labels_cpu = labels.cpu()
            mask_cpu = aspect_mask.cpu()

            for i in range(preds_cpu.size(0)):
                for j in range(preds_cpu.size(1)):
                    if mask_cpu[i, j] and labels_cpu[i, j] != -100:
                        valid_preds.append(preds_cpu[i, j].item())
                        valid_labels.append(labels_cpu[i, j].item())

    # 計算指標
    accuracy = accuracy_score(valid_labels, valid_preds)
    f1_macro = f1_score(valid_labels, valid_preds, average='macro')
    precision = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
    recall = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)

    # 每類 F1
    f1_per_class = f1_score(valid_labels, valid_preds, average=None, zero_division=0)

    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class.tolist()
    }


def train_multiaspect_model(args):
    """訓練 Multi-Aspect 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 數據路徑
    project_root = Path(__file__).parent.parent
    train_path = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Train_v2.xml'
    test_path = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Test_Gold.xml'

    # 加載數據
    print("\n" + "="*80)
    print("加載 Multi-Aspect 數據")
    print("="*80)
    train_samples, test_samples = load_multiaspect_data(
        train_path=str(train_path),
        test_path=str(test_path),
        min_aspects=args.min_aspects,
        max_aspects=args.max_aspects,
        include_single_aspect=args.include_single_aspect,
        virtual_aspect_mode=args.virtual_aspect_mode
    )

    # 分割驗證集
    val_size = int(0.1 * len(train_samples))
    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]

    print(f"\n數據分割:")
    print(f"  訓練: {len(train_samples)}")
    print(f"  驗證: {len(val_samples)}")
    print(f"  測試: {len(test_samples)}")

    # 加載 tokenizer
    print(f"\n加載 {args.bert_model} tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # 創建 DataLoaders
    print("\n創建 DataLoaders...")
    train_loader, val_loader, test_loader = create_multiaspect_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        max_aspect_len=args.max_aspect_len,
        max_num_aspects=args.max_aspects
    )

    # 創建模型
    print("\n" + "="*80)
    print("創建 Multi-Aspect HMAC-Net")
    print("="*80)
    model = HMACNetMultiAspect(
        bert_model_name=args.bert_model,
        freeze_bert=args.freeze_bert,
        hidden_dim=args.hidden_dim,
        num_classes=3,
        dropout=args.dropout,
        use_pmac=args.use_pmac,
        pmac_composition_mode=args.pmac_mode,
        use_iarm=args.use_iarm,
        iarm_relation_mode=args.iarm_mode,
        iarm_num_heads=args.iarm_heads,
        iarm_num_layers=args.iarm_layers
    ).to(device)

    print(f"\n模型配置:")
    print(f"  BERT: {args.bert_model}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  PMAC: {'Enabled' if args.use_pmac else 'Disabled'} ({args.pmac_mode if args.use_pmac else 'N/A'})")
    print(f"  IARM: {'Enabled' if args.use_iarm else 'Disabled'} ({args.iarm_mode if args.use_iarm else 'N/A'})")

    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 訓練
    print("\n" + "="*80)
    print("開始訓練")
    print("="*80)

    best_val_f1 = 0
    patience_counter = 0

    # 記錄訓練歷史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1_macro': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1_per_class': [],
        'epochs': []
    }

    for epoch in range(args.epochs):
        # 訓練階段
        model.train()
        train_loss = 0
        train_steps = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            aspect_ids = batch['aspect_input_ids'].to(device)
            aspect_mask_input = batch['aspect_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)
            is_virtual = batch['is_virtual'].to(device)

            # Forward
            logits = model(text_ids, text_mask, aspect_ids, aspect_mask_input, aspect_mask)

            # Loss
            if args.loss_type != 'ce':
                # 使用 Focal Loss 或 Adaptive Loss
                if 'loss_fn' not in locals():
                    loss_fn = get_loss_function(
                        loss_type=args.loss_type,
                        alpha=args.class_weights,
                        gamma=args.focal_gamma,
                        virtual_weight=args.virtual_weight
                    ).to(device)
                loss = loss_fn(logits, labels, aspect_mask, is_virtual)
            else:
                # 使用標準 CE Loss
                loss = compute_multi_aspect_loss(
                    logits, labels, aspect_mask, is_virtual,
                    virtual_weight=args.virtual_weight
                )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = train_loss / train_steps

        # 驗證階段
        val_metrics = evaluate_multi_aspect(model, val_loader, device)

        # 記錄歷史數據
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(val_metrics.get('loss', 0)))
        history['val_accuracy'].append(float(val_metrics['accuracy']))
        history['val_f1_macro'].append(float(val_metrics['f1_macro']))
        history['val_precision'].append(float(val_metrics['precision']))
        history['val_recall'].append(float(val_metrics['recall']))
        history['val_f1_per_class'].append([float(f1) for f1 in val_metrics['f1_per_class']])

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  Val F1 (Macro): {val_metrics['f1_macro']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}")
        print(f"  Val Recall: {val_metrics['recall']:.4f}")
        print(f"  Val F1 per class: {val_metrics['f1_per_class']}")

        # Early stopping
        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            patience_counter = 0

            # 保存最佳模型
            save_path = project_root / 'results' / 'checkpoints' / f'hmac_multiaspect_best_f1_{best_val_f1:.4f}.pt'
            save_path.parent.mkdir(parents=True, exist_ok=True)
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

    # 加載最佳模型
    best_model_path = project_root / 'results' / 'checkpoints' / f'hmac_multiaspect_best_f1_{best_val_f1:.4f}.pt'
    model.load_state_dict(torch.load(best_model_path))

    test_metrics = evaluate_multi_aspect(model, test_loader, device)

    print(f"\n測試集結果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 per class (neg/neu/pos): {test_metrics['f1_per_class']}")

    # 保存結果
    results = {
        'args': vars(args),
        'best_val_f1': float(best_val_f1),
        'history': history,
        'test_metrics': {k: float(v) if isinstance(v, (float, np.float32, np.float64)) else v
                        for k, v in test_metrics.items()}
    }

    results_path = project_root / 'results' / 'reports' / 'multiaspect_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n結果已保存: {results_path}")

    # 自動生成訓練曲線圖表
    print("\n" + "="*80)
    print("生成訓練曲線圖表")
    print("="*80)
    visualizations_dir = project_root / 'results' / 'visualizations'
    visualizations_dir.mkdir(parents=True, exist_ok=True)

    try:
        generate_training_visualizations(results, visualizations_dir)
        print(f"\n[COMPLETE] All visualizations saved to: {visualizations_dir}")
    except Exception as e:
        print(f"\n[WARNING] Failed to generate visualizations: {e}")

    print("\n[COMPLETE] Training finished!")


def main():
    parser = argparse.ArgumentParser(description='訓練 Multi-Aspect HMAC-Net')

    # 數據參數
    parser.add_argument('--min_aspects', type=int, default=2,
                        help='最小 aspect 數量（用於過濾）')
    parser.add_argument('--max_aspects', type=int, default=8,
                        help='最大 aspect 數量（超過則截斷）')
    parser.add_argument('--include_single_aspect', action='store_true', default=True,
                        help='是否包含單 aspect（帶虛擬 aspect）')
    parser.add_argument('--virtual_aspect_mode', type=str, default='overall',
                        choices=['overall', 'context', 'none'],
                        help='虛擬 aspect 模式')
    parser.add_argument('--max_text_len', type=int, default=128,
                        help='最大文本長度')
    parser.add_argument('--max_aspect_len', type=int, default=10,
                        help='最大 aspect 長度')

    # 模型參數
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                        help='BERT 模型名稱')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='是否凍結 BERT')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='隱藏層維度（建議保持 768）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 比率')

    # PMAC 參數
    parser.add_argument('--use_pmac', action='store_true', default=True,
                        help='是否使用 PMAC')
    parser.add_argument('--pmac_mode', type=str, default='sequential',
                        choices=['sequential', 'pairwise', 'attention'],
                        help='PMAC 組合模式')

    # IARM 參數
    parser.add_argument('--use_iarm', action='store_true', default=True,
                        help='是否使用 IARM')
    parser.add_argument('--iarm_mode', type=str, default='transformer',
                        choices=['transformer', 'gat', 'bilinear'],
                        help='IARM 關係模式')
    parser.add_argument('--iarm_heads', type=int, default=4,
                        help='IARM 注意力頭數')
    parser.add_argument('--iarm_layers', type=int, default=2,
                        help='IARM 層數')

    # 訓練參數
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='訓練輪數')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='梯度裁剪')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--virtual_weight', type=float, default=0.5,
                        help='虛擬 aspect 損失權重')
    parser.add_argument('--loss_type', type=str, default='ce',
                        choices=['ce', 'focal', 'adaptive'],
                        help='Loss function type: ce (CrossEntropy), focal (FocalLoss), adaptive (AdaptiveWeighted)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (default: 2.0, 0=CE)')
    parser.add_argument('--class_weights', type=float, nargs=3, default=None,
                        help='Class weights [neg, neu, pos], e.g., --class_weights 1.0 2.0 1.0')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("Multi-Aspect HMAC-Net 訓練")
    print("="*80)
    print("\n配置:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    train_multiaspect_model(args)


if __name__ == '__main__':
    main()
