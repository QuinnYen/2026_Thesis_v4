"""
Multi-Aspect HMAC-Net Training Script

訓練句子級別的多面向情感分析模型
真正發揮 PMAC 和 IARM 創新模組的作用
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import json
from datetime import datetime
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
from models.pmac_selective import SelectivePMACMultiAspect
from models.iarm_enhanced import IARMMultiAspect
from models.base_model import BaseModel, MLP
from utils.focal_loss import get_loss_function

# Import baseline models
from baselines import create_baseline


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
        gate_bias_init: float = -3.0,  # Selective PMAC gate 初始化偏置
        gate_weight_gain: float = 0.1,  # Selective PMAC gate 權重增益
        # IARM 參數 (Transformer-based)
        use_iarm: bool = True,
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

        # PMAC Multi-Aspect（Selective PMAC - 本論文核心創新）
        if use_pmac:
            self.pmac = SelectivePMACMultiAspect(
                input_dim=hidden_dim,
                fusion_dim=hidden_dim,
                num_composition_layers=2,
                hidden_dim=256,
                dropout=dropout,
                use_layer_norm=True,
                gate_bias_init=gate_bias_init,
                gate_weight_gain=gate_weight_gain
            )

        # IARM Multi-Aspect（Transformer-based 關係建模）
        if use_iarm:
            self.iarm = IARMMultiAspect(
                input_dim=hidden_dim,
                relation_dim=hidden_dim,
                num_heads=iarm_num_heads,
                num_layers=iarm_num_layers,
                dropout=dropout
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
        gate_stats = None
        if self.use_pmac:
            composed_features, pmac_outputs = self.pmac(context_vectors, aspect_mask)
            # 如果是 Selective PMAC，pmac_outputs 就是 gate 值張量
            if isinstance(self.pmac, SelectivePMACMultiAspect) and pmac_outputs is not None:
                gate_stats = pmac_outputs
        else:
            composed_features = context_vectors

        # 5. IARM - 面向間關係建模（創新核心！）
        if self.use_iarm:
            enhanced_features, _ = self.iarm(composed_features, aspect_mask)
        else:
            enhanced_features = composed_features

        # 6. 分類
        logits = self.classifier(enhanced_features)  # [batch, max_aspects, 3]

        return logits, gate_stats


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
    report_lines.append(f"  PMAC: {'Enabled (Selective)' if args['use_pmac'] else 'Disabled'}")
    report_lines.append(f"  IARM: {'Enabled (Transformer)' if args['use_iarm'] else 'Disabled'}")
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

    # Gate統計（如果有）
    if 'gate_stats' in test_metrics and test_metrics['gate_stats'] is not None:
        gate_stats = test_metrics['gate_stats']
        report_lines.append("Selective PMAC Gate Statistics:")
        report_lines.append(f"  Gate Mean: {gate_stats['mean']:.4f} ± {gate_stats['std']:.4f}")
        report_lines.append(f"  Gate Range: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]")
        report_lines.append(f"  Gate Median: {gate_stats['median']:.4f}")
        report_lines.append(f"  Gate Q25-Q75: [{gate_stats['q25']:.4f}, {gate_stats['q75']:.4f}]")
        report_lines.append(f"  Sparsity (gate < 0.1): {gate_stats['sparsity']*100:.1f}%")
        report_lines.append(f"  Activation Rate (gate > 0.5): {gate_stats['activation_rate']*100:.1f}%")
        report_lines.append(f"  Total Gates: {gate_stats['total_gates']}")
        report_lines.append("")
        report_lines.append("Interpretation:")
        if gate_stats['sparsity'] > 0.7:
            report_lines.append("  ✓ High sparsity: Most aspects remain independent")
        elif gate_stats['sparsity'] > 0.3:
            report_lines.append("  ~ Moderate sparsity: Balanced composition")
        else:
            report_lines.append("  ✗ Low sparsity: May be over-composing aspects")
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


def analyze_gate_statistics(all_gates):
    """分析收集的gate統計數據"""
    if not all_gates:
        return None

    gates_array = np.array(all_gates)

    stats = {
        'mean': float(gates_array.mean()),
        'std': float(gates_array.std()),
        'min': float(gates_array.min()),
        'max': float(gates_array.max()),
        'median': float(np.median(gates_array)),
        'q25': float(np.percentile(gates_array, 25)),
        'q75': float(np.percentile(gates_array, 75)),
        'sparsity': float((gates_array < 0.1).mean()),  # gate < 0.1 比例
        'activation_rate': float((gates_array > 0.5).mean()),  # gate > 0.5 比例
        'total_gates': len(gates_array)
    }

    return stats


def evaluate_multi_aspect(model, dataloader, device, loss_fn=None, args=None, collect_gates=False):
    """
    評估 multi-aspect 模型

    返回 aspect-level 指標（展開為獨立的 aspect-sentiment pairs）
    同時計算 validation loss
    """
    model.eval()

    valid_preds = []
    valid_labels = []
    total_loss = 0.0
    num_batches = 0
    all_gates = []  # 收集所有gate值

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ascii=True):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            aspect_ids = batch['aspect_input_ids'].to(device)
            aspect_mask_input = batch['aspect_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)
            is_virtual = batch['is_virtual'].to(device)

            # Forward
            logits, gate_stats = model(text_ids, text_mask, aspect_ids, aspect_mask_input, aspect_mask)

            # 收集gate統計
            if collect_gates and gate_stats is not None:
                all_gates.extend(gate_stats.cpu().numpy().flatten().tolist())

            # 計算 loss（與訓練時使用相同的 loss 函數）
            if loss_fn is not None and args is not None:
                if args.loss_type != 'ce':
                    # 使用 Focal Loss 或其他損失
                    batch_loss = loss_fn(logits, labels, aspect_mask, is_virtual)
                else:
                    # 使用標準 CE Loss
                    batch_loss = compute_multi_aspect_loss(
                        logits, labels, aspect_mask, is_virtual,
                        virtual_weight=args.virtual_weight
                    )
                total_loss += batch_loss.item()
                num_batches += 1

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

    # 計算平均 loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    # 分析gate統計
    gate_analysis = None
    if collect_gates and all_gates:
        gate_analysis = analyze_gate_statistics(all_gates)

    result = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'f1_per_class': f1_per_class.tolist()
    }

    if gate_analysis is not None:
        result['gate_stats'] = gate_analysis

    return result


def train_multiaspect_model(args):
    """訓練 Multi-Aspect 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    # 創建時間戳資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent

    # 實驗名稱（基於配置）
    if args.baseline:
        # Baseline 實驗
        exp_name = f"baseline_{args.baseline}"
        exp_name += f"_drop{args.dropout}_bs{args.batch_size}x{args.accumulation_steps}"
        exp_name += f"_{args.loss_type}"
    else:
        # Full Model 實驗
        exp_name = f"{'pmac' if args.use_pmac else 'nopmac'}_{'iarm' if args.use_iarm else 'noiarm'}"
        exp_name += f"_drop{args.dropout}_bs{args.batch_size}x{args.accumulation_steps}"
        exp_name += f"_{args.loss_type}"

    # 時間戳實驗資料夾
    # Baseline 實驗保存在 results/baseline/ 下，其他實驗保存在 results/experiments/ 下
    if args.baseline:
        exp_dir = project_root / 'results' / 'baseline' / f"{timestamp}_{exp_name}"
    else:
        exp_dir = project_root / 'results' / 'experiments' / f"{timestamp}_{exp_name}"

    checkpoints_dir = exp_dir / 'checkpoints'
    visualizations_dir = exp_dir / 'visualizations'
    reports_dir = exp_dir / 'reports'

    # 創建所有資料夾
    for dir_path in [checkpoints_dir, visualizations_dir, reports_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"\n實驗資料夾: {exp_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    print(f"  Visualizations: {visualizations_dir}")
    print(f"  Reports: {reports_dir}")

    # 數據路徑
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

    # 根據 baseline 參數選擇模型
    if args.baseline:
        print(f"創建 Baseline 模型: {args.baseline}")
        print("="*80)
        model = create_baseline(
            baseline_type=args.baseline,
            bert_model_name=args.bert_model,
            freeze_bert=args.freeze_bert,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            dropout=args.dropout
        ).to(device)

        print(f"\n模型配置:")
        print(f"  模型類型: Baseline - {args.baseline}")
        print(f"  BERT: {args.bert_model}")
        print(f"  Dropout: {args.dropout}")

    else:
        print("創建 Multi-Aspect HMAC-Net (Full Model)")
        print("="*80)
        model = HMACNetMultiAspect(
            bert_model_name=args.bert_model,
            freeze_bert=args.freeze_bert,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            dropout=args.dropout,
            use_pmac=args.use_pmac,
            gate_bias_init=args.gate_bias_init,
            gate_weight_gain=args.gate_weight_gain,
            use_iarm=args.use_iarm,
            iarm_num_heads=args.iarm_heads,
            iarm_num_layers=args.iarm_layers
        ).to(device)

        print(f"\n模型配置:")
        print(f"  BERT: {args.bert_model}")
        print(f"  Hidden dim: {args.hidden_dim}")
        print(f"  Dropout: {args.dropout}")
        print(f"  PMAC: {'Enabled (Selective)' if args.use_pmac else 'Disabled'}")
        print(f"  IARM: {'Enabled (Transformer)' if args.use_iarm else 'Disabled'}")

    print(f"\n訓練配置:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Accumulation steps: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Use scheduler: {args.use_scheduler}")

    # 優化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 學習率調度器（Cosine Annealing + Warmup）
    scheduler = None
    if args.use_scheduler:
        total_steps = len(train_loader) * args.epochs // args.accumulation_steps
        warmup_steps = int(args.warmup_ratio * total_steps)

        # Warmup: 從 0.01*lr 線性增加到 lr
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )

        # Cosine Annealing: 從 lr 降到 1e-7
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=1e-7
        )

        # 組合調度器
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )

        print(f"\n學習率調度器:")
        print(f"  Total steps: {total_steps}")
        print(f"  Warmup steps: {warmup_steps} ({args.warmup_ratio*100:.0f}%)")
        print(f"  Cosine annealing: {total_steps - warmup_steps} steps")
        print(f"  LR range: {args.lr*0.01:.2e} -> {args.lr:.2e} -> 1e-7")

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

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ascii=True)
        for batch_idx, batch in enumerate(progress_bar):
            text_ids = batch['text_input_ids'].to(device)
            text_mask = batch['text_attention_mask'].to(device)
            aspect_ids = batch['aspect_input_ids'].to(device)
            aspect_mask_input = batch['aspect_attention_mask'].to(device)
            labels = batch['labels'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)
            is_virtual = batch['is_virtual'].to(device)

            # Forward
            logits, gate_stats = model(text_ids, text_mask, aspect_ids, aspect_mask_input, aspect_mask)

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

            # 梯度累積：將 loss 除以累積步數
            loss = loss / args.accumulation_steps

            # Backward
            loss.backward()

            # 每 accumulation_steps 步更新一次參數
            if (batch_idx + 1) % args.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # 更新學習率
                if scheduler is not None:
                    scheduler.step()

            train_loss += loss.item() * args.accumulation_steps  # 恢復原始 loss 用於顯示
            train_steps += 1

            # 顯示當前學習率
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item() * args.accumulation_steps:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_train_loss = train_loss / train_steps

        # 驗證階段 - 傳入 loss_fn 和 args 以計算 validation loss
        loss_fn_for_eval = loss_fn if args.loss_type != 'ce' else None
        val_metrics = evaluate_multi_aspect(model, val_loader, device, loss_fn_for_eval, args)

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
            best_epoch = epoch + 1
            patience_counter = 0

            # 如果使用Selective PMAC，收集gate統計 (baseline 沒有 gate)
            if args.use_pmac and not args.baseline:
                print("\n  [Gate Analysis] 收集Selective PMAC的gate統計...")
                gate_metrics = evaluate_multi_aspect(model, val_loader, device, loss_fn_for_eval, args, collect_gates=True)
                if 'gate_stats' in gate_metrics:
                    gate_stats = gate_metrics['gate_stats']
                    print(f"  Gate Mean: {gate_stats['mean']:.4f} ± {gate_stats['std']:.4f}")
                    print(f"  Gate Range: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]")
                    print(f"  Gate Median: {gate_stats['median']:.4f}")
                    print(f"  Sparsity (< 0.1): {gate_stats['sparsity']*100:.1f}%")
                    print(f"  Activation Rate (> 0.5): {gate_stats['activation_rate']*100:.1f}%")

            # 保存最佳模型到時間戳資料夾
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

    # 加載最佳模型
    best_model_path = checkpoints_dir / f'best_model_epoch{best_epoch}_f1_{best_val_f1:.4f}.pt'
    model.load_state_dict(torch.load(best_model_path))

    # 測試集評估（也計算 loss）
    # 如果使用Selective PMAC，收集gate統計 (baseline 沒有 gate)
    collect_test_gates = args.use_pmac and not args.baseline  # Selective PMAC 總是收集 gate 統計
    loss_fn_for_eval = loss_fn if args.loss_type != 'ce' else None
    test_metrics = evaluate_multi_aspect(model, test_loader, device, loss_fn_for_eval, args, collect_gates=collect_test_gates)

    print(f"\n測試集結果:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1 per class (neg/neu/pos): {test_metrics['f1_per_class']}")

    # 輸出gate統計
    if 'gate_stats' in test_metrics:
        gate_stats = test_metrics['gate_stats']
        print(f"\nSelective PMAC Gate Statistics:")
        print(f"  Gate Mean: {gate_stats['mean']:.4f} ± {gate_stats['std']:.4f}")
        print(f"  Gate Range: [{gate_stats['min']:.4f}, {gate_stats['max']:.4f}]")
        print(f"  Sparsity (< 0.1): {gate_stats['sparsity']*100:.1f}%")
        print(f"  Activation Rate (> 0.5): {gate_stats['activation_rate']*100:.1f}%")

    # 保存結果
    results = {
        'args': vars(args),
        'best_val_f1': float(best_val_f1),
        'history': history,
        'test_metrics': {k: float(v) if isinstance(v, (float, np.float32, np.float64)) else v
                        for k, v in test_metrics.items()}
    }

    # 保存結果到時間戳資料夾
    results_path = reports_dir / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n結果已保存: {results_path}")

    # 保存實驗配置
    config_path = reports_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"配置已保存: {config_path}")

    # 自動生成訓練曲線圖表
    print("\n" + "="*80)
    print("生成訓練曲線圖表")
    print("="*80)

    try:
        generate_training_visualizations(results, visualizations_dir)
        print(f"\n[COMPLETE] All visualizations saved to: {visualizations_dir}")
    except Exception as e:
        print(f"\n[WARNING] Failed to generate visualizations: {e}")

    # 生成實驗摘要
    summary_path = reports_dir / 'experiment_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"實驗摘要 - {timestamp}\n")
        f.write("="*80 + "\n\n")

        f.write("實驗配置:\n")
        f.write(f"  實驗名稱: {exp_name}\n")
        if args.baseline:
            f.write(f"  模型類型: Baseline - {args.baseline}\n")
        else:
            f.write(f"  模型類型: Full HMAC-Net\n")
            f.write(f"  PMAC: {'Enabled' if args.use_pmac else 'Disabled'}\n")
            f.write(f"  IARM: {'Enabled' if args.use_iarm else 'Disabled'}\n")
        f.write(f"  Dropout: {args.dropout}\n")
        f.write(f"  Batch Size: {args.batch_size} x {args.accumulation_steps} = {args.batch_size * args.accumulation_steps}\n")
        f.write(f"  Loss Type: {args.loss_type}\n")
        f.write(f"  Learning Rate: {args.lr}\n")
        f.write(f"  Epochs: {args.epochs}\n\n")

        f.write("訓練結果:\n")
        f.write(f"  Best Epoch: {best_epoch}\n")
        f.write(f"  Best Val F1: {best_val_f1:.4f}\n")
        f.write(f"  Test Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Test F1 (Macro): {test_metrics['f1_macro']:.4f}\n")
        f.write(f"  Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Test Recall: {test_metrics['recall']:.4f}\n\n")

        f.write("Per-Class Test F1:\n")
        f.write(f"  Negative: {test_metrics['f1_per_class'][0]:.4f}\n")
        f.write(f"  Neutral:  {test_metrics['f1_per_class'][1]:.4f}\n")
        f.write(f"  Positive: {test_metrics['f1_per_class'][2]:.4f}\n\n")

        f.write("文件路徑:\n")
        f.write(f"  實驗資料夾: {exp_dir}\n")
        f.write(f"  最佳模型: {checkpoints_dir / f'best_model_epoch{best_epoch}_f1_{best_val_f1:.4f}.pt'}\n")
        f.write(f"  結果JSON: {results_path}\n")
        f.write(f"  配置JSON: {config_path}\n")
        f.write(f"  可視化: {visualizations_dir}\n")

    print(f"\n實驗摘要已保存: {summary_path}")
    print("\n[COMPLETE] Training finished!")
    print(f"\n所有實驗結果已保存至: {exp_dir}")


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

    # 模型類型選擇
    parser.add_argument('--baseline', type=str, default=None,
                        choices=['bert_only', 'bert_aaha', 'bert_mean'],
                        help='使用 baseline 模型 (bert_only, bert_aaha, bert_mean)。'
                             '如果不指定，則使用完整的 HMAC-Net')

    # 模型參數
    parser.add_argument('--bert_model', type=str, default='distilbert-base-uncased',
                        help='BERT 模型名稱')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='是否凍結 BERT')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='隱藏層維度（建議保持 768）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 比率')

    # PMAC 參數 (Selective PMAC - 本論文核心創新)
    parser.add_argument('--use_pmac', action='store_true', default=False,
                        help='是否使用 Selective PMAC')
    parser.add_argument('--gate_bias_init', type=float, default=-3.0,
                        help='Gate 偏置初始值 (-2.0≈0.12, -3.0≈0.05, -4.0≈0.02)')
    parser.add_argument('--gate_weight_gain', type=float, default=0.1,
                        help='Gate 權重初始化增益')
    parser.add_argument('--gate_sparsity_weight', type=float, default=0.0,
                        help='Gate 稀疏性正則化權重 (0=不使用, 推薦0.001-0.01)')
    parser.add_argument('--gate_sparsity_type', type=str, default='l1',
                        choices=['l1', 'l2', 'hoyer', 'target'],
                        help='Gate 稀疏性正則化類型')

    # IARM 參數 (Transformer-based Inter-Aspect Relation Modeling)
    parser.add_argument('--use_iarm', action='store_true', default=False,
                        help='是否使用 IARM (Transformer)')
    parser.add_argument('--iarm_heads', type=int, default=4,
                        help='IARM Transformer 注意力頭數')
    parser.add_argument('--iarm_layers', type=int, default=2,
                        help='IARM Transformer 層數')

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
    parser.add_argument('--accumulation_steps', type=int, default=2,
                        help='梯度累積步數（有效 batch size = batch_size * accumulation_steps）')
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help='是否使用 Cosine Annealing + Warmup 學習率調度器')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warmup 步數比例（總步數的百分比）')
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
