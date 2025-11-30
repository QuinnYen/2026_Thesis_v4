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
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, DebertaV2Tokenizer
import argparse
from pathlib import Path
import sys
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import json
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式後端
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.semeval_multiaspect import load_multiaspect_data
from data.semeval2016_multiaspect import load_semeval2016_data
from data.mams_multiaspect import load_mams_data
from data.memd_multiaspect import load_memd_data
from data.multiaspect_dataset import create_multiaspect_dataloaders
from models.bert_embedding import BERTForABSA
from models.base_model import BaseModel
from utils.focal_loss import get_loss_function, get_combined_loss_function, MultiAspectContrastiveLoss, get_distillation_loss_function
from utils.dataset_analyzer import analyze_dataset, print_dataset_stats
from utils.model_selector import select_model, get_model_config, print_selection_result

# Import baseline models
from experiments.baselines import create_baseline

# Import improved models
from experiments.improved_models import create_improved_model


def load_soft_labels(soft_labels_file: str, dataset: str) -> dict:
    """
    載入 Claude soft labels

    Args:
        soft_labels_file: soft labels JSON 文件名（相對於 data/soft_labels/）
        dataset: 數據集名稱（用於自動推斷文件名）

    Returns:
        dict: {(text, aspect): soft_label} 的映射
    """
    project_root = Path(__file__).parent.parent

    # 確定文件路徑
    if soft_labels_file:
        soft_labels_path = project_root / 'data' / 'soft_labels' / soft_labels_file
    else:
        # 自動推斷文件名
        soft_labels_path = project_root / 'data' / 'soft_labels' / f'{dataset}_soft_labels.json'

    if not soft_labels_path.exists():
        print(f"  [Warning] Soft labels file not found: {soft_labels_path}")
        print(f"  [Warning] Please run: python data/claude_soft_labels.py --dataset {dataset}")
        return None

    print(f"  [Distillation] Loading soft labels from: {soft_labels_path.name}")

    with open(soft_labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 建立 (text, aspect) -> soft_label 的映射
    soft_labels_map = {}
    for sample in data.get('samples', []):
        key = (sample['text'], sample['aspect'])
        soft_labels_map[key] = sample['soft_label']

    print(f"  [Distillation] Loaded {len(soft_labels_map)} soft labels")

    return soft_labels_map


def generate_training_visualizations(results, save_dir):
    """
    生成所有訓練曲線圖表

    生成的圖表：
    1. loss_and_accuracy.png - 損失函數與準確率曲線圖
    2. learning_curves.png - 學習曲線（F1 Score）
    3. train_val_comparison.png - Train vs Val 對比圖
    4. confusion_matrix.png - 混淆矩陣
    """
    from sklearn.metrics import confusion_matrix

    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    sns.set_style("whitegrid")

    history = results['history']
    epochs = history['epochs']
    test_metrics = results.get('test_metrics', {})

    # 靜默生成圖表

    # ========== 1. 損失函數與準確率曲線圖 (Loss and Accuracy Curves) ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Loss and Accuracy Curves', fontsize=16, fontweight='bold')

    # 左圖：Loss 曲線
    axes[0].plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=5, label='Train Loss')
    if 'val_loss' in history and any(history['val_loss']):
        axes[0].plot(epochs, history['val_loss'], 'r-s', linewidth=2, markersize=5, label='Val Loss')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右圖：Accuracy 曲線
    val_acc = history['val_accuracy']
    axes[1].plot(epochs, val_acc, 'g-o', linewidth=2, markersize=5, label='Val Accuracy')
    best_epoch = np.argmax(val_acc) + 1
    best_acc = max(val_acc)
    axes[1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best: Epoch {best_epoch}')
    axes[1].axhline(y=best_acc, color='r', linestyle=':', alpha=0.3)
    axes[1].scatter([best_epoch], [best_acc], color='r', s=100, zorder=5)
    axes[1].annotate(f'{best_acc:.4f}', (best_epoch, best_acc), textcoords="offset points",
                     xytext=(10, 5), fontsize=10, fontweight='bold', color='r')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'loss_and_accuracy.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 2. 學習曲線 (Learning Curves) - F1 Score ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Learning Curves (F1 Score)', fontsize=16, fontweight='bold')

    # 左圖：Macro F1
    val_f1 = history['val_f1_macro']
    axes[0].plot(epochs, val_f1, 'm-o', linewidth=2, markersize=5, label='Val F1 (Macro)')
    best_epoch_f1 = np.argmax(val_f1) + 1
    best_f1 = max(val_f1)
    axes[0].axvline(x=best_epoch_f1, color='r', linestyle='--', alpha=0.5)
    axes[0].scatter([best_epoch_f1], [best_f1], color='r', s=100, zorder=5)
    axes[0].annotate(f'{best_f1:.4f}', (best_epoch_f1, best_f1), textcoords="offset points",
                     xytext=(10, 5), fontsize=10, fontweight='bold', color='r')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0].set_title(f'Validation F1 Macro (Best: Epoch {best_epoch_f1})', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 右圖：Per-Class F1
    f1_per_class = np.array(history['val_f1_per_class'])
    class_names = ['Negative', 'Neutral', 'Positive']
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        axes[1].plot(epochs, f1_per_class[:, i], marker='o', label=class_name,
                    color=color, linewidth=2, markersize=5)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Per-Class F1 Scores', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'learning_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 3. Train vs Val 對比圖 ==========
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Train vs Validation Comparison', fontsize=16, fontweight='bold')

    # 左圖：Loss 對比
    axes[0].plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=5, label='Train Loss')
    if 'val_loss' in history and any(history['val_loss']):
        axes[0].plot(epochs, history['val_loss'], 'r-s', linewidth=2, markersize=5, label='Val Loss')
    axes[0].fill_between(epochs, history['train_loss'],
                         history['val_loss'] if 'val_loss' in history and any(history['val_loss']) else history['train_loss'],
                         alpha=0.2, color='gray', label='Gap')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Loss: Train vs Validation', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # 判斷過擬合/欠擬合
    if 'val_loss' in history and any(history['val_loss']):
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        gap = final_val_loss - final_train_loss
        if gap > 0.1:
            status = "Overfitting detected"
            status_color = 'red'
        elif gap < -0.05:
            status = "Underfitting detected"
            status_color = 'orange'
        else:
            status = "Good fit"
            status_color = 'green'
        axes[0].text(0.02, 0.98, status, transform=axes[0].transAxes, fontsize=11,
                    fontweight='bold', color=status_color, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # 右圖：Precision, Recall, F1 對比
    axes[1].plot(epochs, history['val_f1_macro'], 'g-o', linewidth=2, markersize=5, label='F1 Macro')
    axes[1].plot(epochs, history['val_precision'], 'b-s', linewidth=2, markersize=5, label='Precision')
    axes[1].plot(epochs, history['val_recall'], 'r-^', linewidth=2, markersize=5, label='Recall')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1].set_title('Validation Metrics Comparison', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'train_val_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # ========== 4. 混淆矩陣 (Confusion Matrix) ==========
    if 'confusion_matrix' in test_metrics:
        cm = np.array(test_metrics['confusion_matrix'])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Confusion Matrix (Test Set)', fontsize=16, fontweight='bold')

        # 左圖：數值混淆矩陣
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Actual', fontsize=12, fontweight='bold')
        axes[0].set_title('Counts', fontsize=14, fontweight='bold')

        # 右圖：正規化混淆矩陣（百分比）
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(cm_normalized, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Actual', fontsize=12, fontweight='bold')
        axes[1].set_title('Percentage (%)', fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = save_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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
    if args.get('baseline'):
        report_lines.append(f"  Type: Baseline - {args['baseline']}")
    elif args.get('improved'):
        report_lines.append(f"  Type: Improved - {args['improved']}")
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

    # 混淆矩陣（文字格式）
    if 'confusion_matrix' in test_metrics:
        cm = test_metrics['confusion_matrix']
        cm_array = np.array(cm)
        report_lines.append("Confusion Matrix:")
        report_lines.append("                  Predicted")
        report_lines.append("                  Neg      Neu      Pos")
        report_lines.append(f"  Actual Neg     {cm_array[0,0]:>5}    {cm_array[0,1]:>5}    {cm_array[0,2]:>5}")
        report_lines.append(f"  Actual Neu     {cm_array[1,0]:>5}    {cm_array[1,1]:>5}    {cm_array[1,2]:>5}")
        report_lines.append(f"  Actual Pos     {cm_array[2,0]:>5}    {cm_array[2,1]:>5}    {cm_array[2,2]:>5}")
        report_lines.append("")
        # 計算每類的 Precision, Recall
        report_lines.append("Per-class Metrics:")
        class_names = ['Negative', 'Neutral', 'Positive']
        for i, name in enumerate(class_names):
            tp = cm_array[i, i]
            fn = cm_array[i, :].sum() - tp  # 實際為 i 但預測錯誤
            fp = cm_array[:, i].sum() - tp  # 預測為 i 但實際不是
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            report_lines.append(f"  {name:10s}: Precision={precision:.4f}, Recall={recall:.4f}")
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


def create_llrd_optimizer(model, base_lr, weight_decay, llrd_decay=0.95):
    """
    創建帶有 Layer-wise Learning Rate Decay (LLRD) 的優化器

    設計理念：
        - 高層 (靠近 Output): 保持 base_lr，快速適應 ABSA 任務
        - 底層 (靠近 Input): 較小 lr，保留 BERT 原本的通用特徵

    參數:
        model: 模型
        base_lr: 基礎學習率 (用於最高層和非 BERT 參數)
        weight_decay: 權重衰減
        llrd_decay: 衰減係數 (default: 0.95，每往下一層 LR 乘上此係數)

    返回:
        AdamW optimizer with layer-wise learning rates
    """
    # 找到 BERT 模型
    bert_model = None
    if hasattr(model, 'bert_absa'):
        bert_model = model.bert_absa.bert_embedding.bert
    elif hasattr(model, 'bert'):
        bert_model = model.bert

    if bert_model is None:
        # 如果找不到 BERT，使用統一學習率
        print("  LLRD: BERT model not found, using uniform lr")
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # 分組參數
    param_groups = []
    no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias', 'layer_norm']

    # BERT 有 12 層 encoder layers (layer 0-11)
    # 我們希望: layer 11 (最高) 用 base_lr, layer 0 (最低) 用 base_lr * decay^11
    num_layers = 12

    # 1. BERT embeddings (最底層，學習率最小)
    embedding_lr = base_lr * (llrd_decay ** (num_layers + 1))
    embedding_params = []
    embedding_params_no_decay = []

    for name, param in bert_model.embeddings.named_parameters():
        if any(nd in name for nd in no_decay):
            embedding_params_no_decay.append(param)
        else:
            embedding_params.append(param)

    if embedding_params:
        param_groups.append({
            'params': embedding_params,
            'lr': embedding_lr,
            'weight_decay': weight_decay
        })
    if embedding_params_no_decay:
        param_groups.append({
            'params': embedding_params_no_decay,
            'lr': embedding_lr,
            'weight_decay': 0.0
        })

    # 2. BERT encoder layers (layer 0-11)
    for layer_idx in range(num_layers):
        layer_lr = base_lr * (llrd_decay ** (num_layers - layer_idx - 1))
        layer_params = []
        layer_params_no_decay = []

        layer = bert_model.encoder.layer[layer_idx]
        for name, param in layer.named_parameters():
            if any(nd in name for nd in no_decay):
                layer_params_no_decay.append(param)
            else:
                layer_params.append(param)

        if layer_params:
            param_groups.append({
                'params': layer_params,
                'lr': layer_lr,
                'weight_decay': weight_decay
            })
        if layer_params_no_decay:
            param_groups.append({
                'params': layer_params_no_decay,
                'lr': layer_lr,
                'weight_decay': 0.0
            })

    # 3. BERT pooler (如果存在)
    if hasattr(bert_model, 'pooler') and bert_model.pooler is not None:
        pooler_params = []
        pooler_params_no_decay = []
        for name, param in bert_model.pooler.named_parameters():
            if any(nd in name for nd in no_decay):
                pooler_params_no_decay.append(param)
            else:
                pooler_params.append(param)

        if pooler_params:
            param_groups.append({
                'params': pooler_params,
                'lr': base_lr,
                'weight_decay': weight_decay
            })
        if pooler_params_no_decay:
            param_groups.append({
                'params': pooler_params_no_decay,
                'lr': base_lr,
                'weight_decay': 0.0
            })

    # 4. 非 BERT 參數 (classifier, attention layers 等) - 使用 base_lr
    bert_param_ids = set()
    for param in bert_model.parameters():
        bert_param_ids.add(id(param))

    other_params = []
    other_params_no_decay = []
    for name, param in model.named_parameters():
        if id(param) not in bert_param_ids:
            if any(nd in name for nd in no_decay):
                other_params_no_decay.append(param)
            else:
                other_params.append(param)

    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
    if other_params_no_decay:
        param_groups.append({
            'params': other_params_no_decay,
            'lr': base_lr,
            'weight_decay': 0.0
        })

    # 打印 LLRD 信息
    print(f"  LLRD enabled: decay={llrd_decay}")
    print(f"    Embeddings lr: {embedding_lr:.2e}")
    print(f"    Layer 0 lr:    {base_lr * (llrd_decay ** 11):.2e}")
    print(f"    Layer 11 lr:   {base_lr:.2e}")
    print(f"    Classifier lr: {base_lr:.2e}")

    return torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)


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


def evaluate_multi_aspect(model, dataloader, device, loss_fn=None, args=None, collect_gates=False, return_confusion_matrix=False):
    """
    評估 multi-aspect 模型

    返回 aspect-level 指標（展開為獨立的 aspect-sentiment pairs）
    同時計算 validation loss

    參數:
        return_confusion_matrix: 是否返回混淆矩陣（用於視覺化）
    """
    model.eval()

    valid_preds = []
    valid_labels = []
    valid_probs = []  # 收集預測概率（用於 AUC）
    total_loss = 0.0
    num_batches = 0
    all_gates = []  # 收集所有gate值
    layer_attention_weights = None  # 收集 HBL 的 layer attention 權重
    all_branch_weights = []  # 收集 AH-IARN 的分支權重
    hsa_level_weights = None  # 收集 HSA 層級權重

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", ascii=True, leave=False):
            # 使用新的 sentence-pair 格式
            pair_ids = batch['pair_input_ids'].to(device)
            pair_mask = batch['pair_attention_mask'].to(device)
            pair_type_ids = batch['pair_token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)
            is_virtual = batch['is_virtual'].to(device)

            # Forward with sentence-pair format
            logits, extras = model(pair_ids, pair_mask, pair_type_ids, aspect_mask)

            # 收集 gate 統計（PMAC 模組）或 layer attention（HBL 模組）或 branch weights（AH-IARN）
            if extras is not None:
                if isinstance(extras, dict):
                    # HBL 模型：收集 layer attention 權重
                    if 'layer_attention' in extras:
                        if layer_attention_weights is None:
                            layer_attention_weights = extras['layer_attention']
                    # AH-IARN 模型：收集分支權重
                    if 'branch_weights' in extras:
                        all_branch_weights.append(extras['branch_weights'])
                    # 收集 HSA 層級權重
                    if 'hsa_level_weights' in extras and hsa_level_weights is None:
                        hsa_level_weights = extras['hsa_level_weights']
                elif collect_gates:
                    # PMAC 模型：收集 gate 值
                    all_gates.extend(extras.cpu().numpy().flatten().tolist())

            # 計算 loss（與訓練時使用相同的 loss 函數）
            if loss_fn is not None and args is not None:
                if args.loss_type == 'distill':
                    # 蒸餾模式驗證時：使用不帶 soft labels 的 focal loss
                    # 因為驗證集沒有 soft labels，且評估時只關心 hard label 的準確度
                    batch_loss = compute_multi_aspect_loss(
                        logits, labels, aspect_mask, is_virtual,
                        virtual_weight=args.virtual_weight
                    )
                elif args.loss_type != 'ce':
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

            # 預測 - 支持非對稱 Logit 調整 (Asymmetric Logit Adjustment)
            #
            # 策略說明：
            #   1. neutral_boost: 提升 Neutral logits，解決 Neu→Pos/Neu→Neg 誤判
            #   2. neg_suppress:  抑制 Negative logits，解決 Neu→Neg 誤判（筆電領域）
            #   3. pos_suppress:  抑制 Positive logits，解決 Neu→Pos 誤判（餐廳領域）
            #
            # 領域特性：
            #   - 筆電領域：主要問題是 Neu→Neg，使用 neg_suppress
            #   - 餐廳領域：Yelp 正向偏誤導致 Neu→Pos，使用 pos_suppress
            #
            # 數學效果：
            #   調整後 logits: [L_Neg - neg_suppress, L_Neu + neutral_boost, L_Pos - pos_suppress]
            #
            neutral_boost = getattr(args, 'neutral_boost', 0.0) if args else 0.0
            neg_suppress = getattr(args, 'neg_suppress', 0.0) if args else 0.0
            pos_suppress = getattr(args, 'pos_suppress', 0.0) if args else 0.0

            if neutral_boost != 0.0 or neg_suppress != 0.0 or pos_suppress != 0.0:
                # 複製 logits 避免修改原始值
                adjusted_logits = logits.clone()

                # 1. 提升 Neutral (通用)
                if neutral_boost > 0:
                    adjusted_logits[:, :, 1] = adjusted_logits[:, :, 1] + neutral_boost

                # 2. 抑制 Negative (解決 Neu→Neg，筆電領域)
                if neg_suppress > 0:
                    adjusted_logits[:, :, 0] = adjusted_logits[:, :, 0] - neg_suppress

                # 3. 抑制 Positive (解決 Neu→Pos，餐廳領域)
                if pos_suppress > 0:
                    adjusted_logits[:, :, 2] = adjusted_logits[:, :, 2] - pos_suppress

                preds = torch.argmax(adjusted_logits, dim=-1)
            else:
                preds = torch.argmax(logits, dim=-1)  # [batch, max_aspects]
            probs = torch.softmax(logits, dim=-1)  # [batch, max_aspects, num_classes] 用於 AUC

            # 直接展開為 aspect-level（過濾無效和虛擬 aspects）
            preds_cpu = preds.cpu()
            labels_cpu = labels.cpu()
            probs_cpu = probs.cpu()
            mask_cpu = aspect_mask.cpu()

            for i in range(preds_cpu.size(0)):
                for j in range(preds_cpu.size(1)):
                    if mask_cpu[i, j] and labels_cpu[i, j] != -100:
                        valid_preds.append(preds_cpu[i, j].item())
                        valid_labels.append(labels_cpu[i, j].item())
                        valid_probs.append(probs_cpu[i, j].numpy())  # [num_classes]

    # 計算指標
    accuracy = accuracy_score(valid_labels, valid_preds)
    f1_macro = f1_score(valid_labels, valid_preds, average='macro')
    precision = precision_score(valid_labels, valid_preds, average='macro', zero_division=0)
    recall = recall_score(valid_labels, valid_preds, average='macro', zero_division=0)

    # 每類 F1
    f1_per_class = f1_score(valid_labels, valid_preds, average=None, zero_division=0)

    # 計算 AUC (macro 和 weighted)
    auc_macro = None
    auc_weighted = None
    try:
        # 將概率轉換為 numpy array
        valid_probs_array = np.array(valid_probs)  # [num_samples, num_classes]
        valid_labels_array = np.array(valid_labels)

        # 計算每個類別的樣本數
        unique_labels = np.unique(valid_labels_array)

        # 只有當至少有 2 個類別時才計算 AUC
        if len(unique_labels) >= 2:
            # One-vs-Rest AUC
            auc_macro = roc_auc_score(
                valid_labels_array,
                valid_probs_array,
                multi_class='ovr',
                average='macro'
            )
            auc_weighted = roc_auc_score(
                valid_labels_array,
                valid_probs_array,
                multi_class='ovr',
                average='weighted'
            )
    except Exception as e:
        print(f"  警告: 無法計算 AUC ({str(e)})")

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
        'f1_per_class': f1_per_class.tolist(),
        'auc_macro': auc_macro,
        'auc_weighted': auc_weighted
    }

    if gate_analysis is not None:
        result['gate_stats'] = gate_analysis

    # 加入 layer attention 權重（HBL 特有）
    if layer_attention_weights is not None:
        result['layer_attention'] = layer_attention_weights.tolist()

    # 加入 AH-IARN 分支權重統計
    if all_branch_weights:
        # numpy 已經在文件開頭導入
        # 拼接所有批次的分支權重: [num_batches, batch, max_aspects, 3]
        # 需要計算平均權重
        all_weights = []
        for batch_weights in all_branch_weights:
            # batch_weights: [batch, max_aspects, 3]
            if isinstance(batch_weights, torch.Tensor):
                batch_weights = batch_weights.numpy()
            # 展平為 [num_valid_aspects, 3]
            for sample_weights in batch_weights:
                # sample_weights: [max_aspects, 3]
                for aspect_weights in sample_weights:
                    # 過濾掉全零的（無效 aspect）
                    if np.sum(aspect_weights) > 0:
                        all_weights.append(aspect_weights)

        if all_weights:
            all_weights = np.array(all_weights)  # [num_valid, 3]
            avg_weights = np.mean(all_weights, axis=0)  # [3]
            result['branch_weights'] = {
                'Hierarchical': float(avg_weights[0]),
                'HSA': float(avg_weights[1]),
                'IARN': float(avg_weights[2])
            }

    # 加入 HSA 層級權重
    if hsa_level_weights is not None:
        result['hsa_level_weights'] = {
            'Token': float(hsa_level_weights[0]),
            'Phrase': float(hsa_level_weights[1]),
            'Clause': float(hsa_level_weights[2])
        }

    # 加入混淆矩陣（用於視覺化）
    if return_confusion_matrix:
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(valid_labels, valid_preds, labels=[0, 1, 2])
        result['confusion_matrix'] = cm.tolist()

    return result


def train_multiaspect_model(args):
    """訓練 Multi-Aspect 模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 創建時間戳資料夾
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_root = Path(__file__).parent.parent

    # 實驗名稱（基於配置）
    if args.baseline:
        # Baseline 實驗
        exp_name = f"baseline_{args.baseline}"
        exp_name += f"_drop{args.dropout}_bs{args.batch_size}x{args.accumulation_steps}"
        exp_name += f"_{args.loss_type}"
    elif args.improved:
        # Improved 實驗
        exp_name = f"improved_{args.improved}"
        # 如果使用 LLRD，加入標記以區分
        if args.use_llrd:
            exp_name += "_llrd"
        exp_name += f"_drop{args.dropout}_bs{args.batch_size}x{args.accumulation_steps}"
        exp_name += f"_{args.loss_type}"
    else:
        # Improved 實驗 (沒有指定 --baseline 時)
        exp_name = f"improved_{args.improved}"
        exp_name += f"_drop{args.dropout}_bs{args.batch_size}x{args.accumulation_steps}"
        exp_name += f"_{args.loss_type}"

    # 時間戳實驗資料夾
    # Baseline 實驗保存在 results/baseline/{dataset}/ 下
    # Improved 實驗保存在 results/improved/{dataset}/ 下
    if args.baseline:
        exp_dir = project_root / 'results' / 'baseline' / args.dataset / f"{timestamp}_{exp_name}"
    else:
        # Improved 模型
        exp_dir = project_root / 'results' / 'improved' / args.dataset / f"{timestamp}_{exp_name}"

    checkpoints_dir = exp_dir / 'checkpoints'
    visualizations_dir = exp_dir / 'visualizations'
    reports_dir = exp_dir / 'reports'

    # 創建所有資料夾
    for dir_path in [checkpoints_dir, visualizations_dir, reports_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print(f"  Output: {exp_dir.name}")

    # ========== HKGAN v2.0: 處理 Confidence Gate 和 Domain 參數 ==========
    # 如果使用 --no_confidence_gate，覆蓋 use_confidence_gate
    if args.no_confidence_gate:
        args.use_confidence_gate = False

    # 自動推斷 domain（如果未指定）
    if args.domain is None and args.improved == 'hkgan':
        # 根據數據集名稱自動設置領域
        if args.dataset in ['laptops', 'lap16']:
            args.domain = 'laptops'
            print(f"  [HKGAN] Auto-detected domain: laptops")
        elif args.dataset in ['restaurants', 'rest16', 'mams']:
            args.domain = 'restaurants'
            print(f"  [HKGAN] Auto-detected domain: restaurants")
        # 其他數據集不設置 domain（不啟用領域過濾）

    # 加載數據

    # 根據 dataset 參數設置數據路徑
    if args.dataset == 'restaurants':
        train_xml = 'Restaurants_Train_v2.xml'
        test_xml = 'Restaurants_Test_Gold.xml'
        default_aug_dir = 'data/augmented_restaurants'
    elif args.dataset == 'laptops':
        train_xml = 'Laptop_Train_v2.xml'
        test_xml = 'Laptops_Test_Gold.xml'
        default_aug_dir = 'data/augmented_laptops'
    elif args.dataset == 'rest16':
        # SemEval-2016 Restaurants
        train_xml = 'restaurants16_train_sb1.xml'
        test_xml = 'restaurants16_test_sb1.xml'
        default_aug_dir = None
    elif args.dataset == 'lap16':
        # SemEval-2016 Laptops
        train_xml = 'Laptops16_train_sb1.xml'
        test_xml = 'Laptops16_test_sb1.xml'
        default_aug_dir = None
    elif args.dataset == 'mams':
        # MAMS 已經有分割好的 train/val/test
        train_xml = 'train.xml'
        val_xml = 'val.xml'
        test_xml = 'test.xml'
        default_aug_dir = None  # MAMS 暫不支持數據增強
    elif args.dataset.startswith('memd_'):
        # MEMD-ABSA 數據集 (Books, Clothing, Hotel, Laptop, Restaurant)
        memd_domain = args.dataset.replace('memd_', '').capitalize()
        default_aug_dir = None  # MEMD 暫不支持數據增強
    else:
        raise ValueError(f"不支援的數據集: {args.dataset}")

    # MAMS 數據集的特殊處理
    if args.dataset == 'mams':

        # MAMS 數據路徑
        train_path = project_root / 'data' / 'raw' / 'MAMS-ATSA' / train_xml
        val_path = project_root / 'data' / 'raw' / 'MAMS-ATSA' / val_xml
        test_path = project_root / 'data' / 'raw' / 'MAMS-ATSA' / test_xml

        train_samples, val_samples, test_samples = load_mams_data(
            train_path=str(train_path),
            val_path=str(val_path),
            test_path=str(test_path),
            min_aspects=args.min_aspects,
            max_aspects=args.max_aspects,
            include_single_aspect=args.include_single_aspect,
            virtual_aspect_mode=args.virtual_aspect_mode
        )
    elif args.dataset.startswith('memd_'):
        # MEMD-ABSA 數據集

        # MEMD 數據路徑
        memd_dir = project_root / 'data' / 'raw' / 'MEMD-ABSA'

        train_samples, val_samples, test_samples = load_memd_data(
            domain=memd_domain,
            data_dir=str(memd_dir),
            min_aspects=args.min_aspects,
            max_aspects=args.max_aspects,
            include_single_aspect=args.include_single_aspect,
            virtual_aspect_mode=args.virtual_aspect_mode
        )
    elif args.dataset in ['rest16', 'lap16']:
        # SemEval-2016 數據集處理
        train_path = project_root / 'data' / 'raw' / 'semeval2016' / train_xml
        test_path = project_root / 'data' / 'raw' / 'semeval2016' / test_xml

        train_samples, val_samples, test_samples = load_semeval2016_data(
            train_path=str(train_path),
            test_path=str(test_path),
            min_aspects=args.min_aspects,
            max_aspects=args.max_aspects,
            include_single_aspect=args.include_single_aspect,
            virtual_aspect_mode=args.virtual_aspect_mode
        )
    else:
        # SemEval-2014 數據集處理
        # 數據增強功能已移除（模組已封存）
        use_augmented = getattr(args, 'use_augmented', False)

        # 數據路徑
        train_path = project_root / 'data' / 'raw' / 'semeval2014' / train_xml
        test_path = project_root / 'data' / 'raw' / 'semeval2014' / test_xml

        # 增強數據路徑 (Neutral augmentation) - 支持 JSON 和 XML 格式
        augmented_neutral_path = None
        if use_augmented:
            # 優先使用 JSON 格式 (新版 Claude 增強)
            aug_path_json = project_root / 'data' / 'augmented' / f'{args.dataset}_neutral_augmented.json'
            aug_path_xml = project_root / 'data' / 'augmented' / f'{args.dataset}_neutral_augmented.xml'

            if aug_path_json.exists():
                augmented_neutral_path = str(aug_path_json)
                print(f"  [Augmentation] 找到增強數據: {aug_path_json.name}")
            elif aug_path_xml.exists():
                augmented_neutral_path = str(aug_path_xml)
                print(f"  [Augmentation] 找到增強數據: {aug_path_xml.name}")

        # Self-Training 偽標籤數據路徑
        pseudo_labeled_path = None
        use_self_training = getattr(args, 'use_self_training', False)
        if use_self_training:
            pseudo_path = project_root / 'data' / 'pseudo_labeled' / f'{args.dataset}_pseudo.json'
            if pseudo_path.exists():
                pseudo_labeled_path = str(pseudo_path)
                print(f"  [Self-Training] 找到偽標籤數據: {pseudo_path.name}")

        train_all_samples, test_samples = load_multiaspect_data(
            train_path=str(train_path),
            test_path=str(test_path),
            min_aspects=args.min_aspects,
            max_aspects=args.max_aspects,
            include_single_aspect=args.include_single_aspect,
            virtual_aspect_mode=args.virtual_aspect_mode,
            augmented_neutral_path=augmented_neutral_path,
            pseudo_labeled_path=pseudo_labeled_path
        )

        # 分割驗證集 (SemEval-2014 only)
        val_size = int(0.1 * len(train_all_samples))
        val_samples = train_all_samples[:val_size]
        train_samples = train_all_samples[val_size:]

    print(f"  Train: {len(train_samples)} | Val: {len(val_samples)} | Test: {len(test_samples)}")

    # 顯示類別分布
    def count_label_distribution(samples):
        """統計樣本的類別分布"""
        from collections import Counter
        label_counts = Counter()
        for sample in samples:
            for label in sample.labels:
                if label >= 0:  # 忽略 padding (-100)
                    label_counts[label] += 1
        return label_counts

    train_dist = count_label_distribution(train_samples)
    total = sum(train_dist.values())
    print(f"\n  [Class Distribution] Train set:")
    label_names = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    for label in sorted(train_dist.keys()):
        count = train_dist[label]
        pct = count / total * 100 if total > 0 else 0
        print(f"    {label_names.get(label, f'Label {label}'):10}: {count:5} ({pct:5.1f}%)")

    # 自動模型選擇
    if args.auto_select and not args.baseline and not args.improved:
        # 分析數據集特徵
        dataset_stats = analyze_dataset(train_samples)
        print_dataset_stats(dataset_stats, args.dataset.upper())

        # 自動選擇模型
        selected_model, reason = select_model(dataset_stats)
        print_selection_result(selected_model, reason)

        # 設定模型類型
        args.improved = selected_model

        # 應用推薦配置
        recommended_config = get_model_config(selected_model, args.dataset)
        if args.dropout == 0.1:  # 使用默認值時才覆蓋
            args.dropout = recommended_config['dropout']

    # 加載 tokenizer (DeBERTa-v3 需要使用 slow tokenizer 以避免兼容性問題)
    if 'deberta-v3' in args.bert_model.lower():
        tokenizer = DebertaV2Tokenizer.from_pretrained(args.bert_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # 如果是知識蒸餾模式，載入 soft labels
    soft_labels_dict = None
    if args.loss_type == 'distill':
        soft_labels_dict = load_soft_labels(
            soft_labels_file=getattr(args, 'soft_labels_file', None),
            dataset=args.dataset
        )
        if soft_labels_dict is None:
            print("  [Warning] Falling back to focal loss (no soft labels)")
            args.loss_type = 'focal'

    # 創建 DataLoaders
    train_loader, val_loader, test_loader = create_multiaspect_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_text_len=args.max_text_len,
        max_aspect_len=args.max_aspect_len,
        max_num_aspects=args.max_aspects,
        soft_labels_dict=soft_labels_dict
    )

    # 創建模型
    print("\n" + "-"*60)
    # 檢查是否啟用對比學習
    use_contrastive = getattr(args, 'contrastive_weight', 0.0) > 0

    if args.baseline:
        print(f"Creating Model: Baseline-{args.baseline}")
        model = create_baseline(
            baseline_type=args.baseline,
            bert_model_name=args.bert_model,
            freeze_bert=args.freeze_bert,
            hidden_dim=args.hidden_dim,
            num_classes=3,
            dropout=args.dropout
        ).to(device)
    else:
        print(f"Creating Model: {args.improved}")
        # 獲取配置中的參數 (如果有)
        multi_aspect_threshold = getattr(args, 'multi_aspect_threshold', 0.5)
        selection_mode = getattr(args, 'selection_mode', 'soft')
        num_attention_heads = getattr(args, 'num_attention_heads', 4)
        num_syntax_layers = getattr(args, 'num_syntax_layers', 2)
        router_type = getattr(args, 'router_type', 'confidence')
        freeze_experts = getattr(args, 'freeze_experts', False)

        model = create_improved_model(
            model_type=args.improved,
            args=args,
            num_classes=3
        ).to(device)

        # 如果是 Unified-HIARN，計算多面向比例並設置模式
        if args.improved == 'unified_hiarn':
            # 計算訓練集的多面向比例
            n_multi = sum(1 for s in train_samples if s.num_aspects >= 2)
            n_total = len(train_samples)
            multi_aspect_ratio = n_multi / n_total if n_total > 0 else 0.0
            model.set_mode(multi_aspect_ratio)

    print(f"  BERT: {args.bert_model} | Dropout: {args.dropout}")
    if use_contrastive:
        print(f"  Contrastive Learning: weight={args.contrastive_weight}, temp={args.contrastive_temperature}")
    if args.neutral_boost != 0.0:
        print(f"  Neutral Boost: +{args.neutral_boost} (logit offset for Neutral class)")

    # 優化器 (支援 LLRD)
    if args.use_llrd:
        optimizer = create_llrd_optimizer(
            model,
            base_lr=args.lr,
            weight_decay=args.weight_decay,
            llrd_decay=args.llrd_decay
        )
    else:
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
        print(f"  Scheduler: Warmup({warmup_steps}) + Cosine({total_steps - warmup_steps})")

    # 訓練
    print("\n" + "-"*60)
    # 混合精度訓練 (AMP) - DeBERTa 在 FP16/BF16 下都會溢出，必須禁用
    is_deberta = 'deberta' in args.bert_model.lower()

    if device.type == 'cuda' and is_deberta:
        # DeBERTa 的注意力機制在混合精度下會溢出，必須使用 FP32
        use_amp = False
        amp_dtype = torch.float32
        print("Training [FP32] (DeBERTa requires full precision)")
    elif device.type == 'cuda':
        use_amp = True
        amp_dtype = torch.float16
        print("Training [AMP-FP16]")
    else:
        use_amp = False
        amp_dtype = torch.float32
        print("Training [CPU]")

    print("-"*60)
    scaler = GradScaler(enabled=use_amp)

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
            # 使用新的 sentence-pair 格式
            pair_ids = batch['pair_input_ids'].to(device)
            pair_mask = batch['pair_attention_mask'].to(device)
            pair_type_ids = batch['pair_token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            aspect_mask = batch['aspect_mask'].to(device)
            is_virtual = batch['is_virtual'].to(device)

            # Forward + Loss (with AMP autocast)
            with autocast(device_type='cuda', dtype=amp_dtype, enabled=use_amp):
                logits, extras = model(pair_ids, pair_mask, pair_type_ids, aspect_mask)

                # Loss
                if args.loss_type == 'distill':
                    # 知識蒸餾損失
                    soft_labels = batch['soft_labels'].to(device)

                    if 'distill_loss_fn' not in locals():
                        # 決定 class weights
                        if args.auto_class_weights:
                            from utils.focal_loss import compute_dynamic_class_weights
                            class_weights = compute_dynamic_class_weights(train_samples)
                            print(f"  [Distill] Class weights: {class_weights}")
                        else:
                            class_weights = args.class_weights

                        distill_loss_fn = get_distillation_loss_function(
                            alpha=getattr(args, 'distill_alpha', 0.3),
                            temperature=getattr(args, 'distill_temperature', 2.0),
                            use_focal=True,
                            focal_gamma=args.focal_gamma,
                            class_weights=class_weights,
                            virtual_weight=args.virtual_weight
                        ).to(device)

                        # 創建對比損失函數（如果啟用）
                        if use_contrastive and 'contrastive_loss_fn' not in locals():
                            contrastive_loss_fn = MultiAspectContrastiveLoss(
                                temperature=args.contrastive_temperature,
                                contrastive_weight=args.contrastive_weight
                            ).to(device)

                    loss = distill_loss_fn(logits, labels, soft_labels, aspect_mask, is_virtual)

                    # 添加對比損失（如果啟用且模型提供特徵）
                    if use_contrastive and extras is not None and 'features' in extras:
                        contrastive_loss = contrastive_loss_fn(
                            extras['features'], labels, aspect_mask
                        )
                        loss = loss + contrastive_loss

                elif args.loss_type != 'ce':
                    # 使用 Focal Loss 或 Adaptive Loss
                    if 'loss_fn' not in locals():
                        # 決定 class weights
                        if args.auto_class_weights:
                            alpha = 'auto'
                        else:
                            alpha = args.class_weights

                        loss_fn = get_loss_function(
                            loss_type=args.loss_type,
                            alpha=alpha,
                            gamma=args.focal_gamma,
                            virtual_weight=args.virtual_weight,
                            train_samples=train_samples  # 傳入訓練樣本以計算動態權重
                        ).to(device)

                        # 創建對比損失函數（如果啟用）
                        if use_contrastive and 'contrastive_loss_fn' not in locals():
                            contrastive_loss_fn = MultiAspectContrastiveLoss(
                                temperature=args.contrastive_temperature,
                                contrastive_weight=args.contrastive_weight
                            ).to(device)

                    loss = loss_fn(logits, labels, aspect_mask, is_virtual)

                    # 添加對比損失（如果啟用且模型提供特徵）
                    if use_contrastive and extras is not None and 'features' in extras:
                        contrastive_loss = contrastive_loss_fn(
                            extras['features'], labels, aspect_mask
                        )
                        loss = loss + contrastive_loss
                else:
                    # 使用標準 CE Loss
                    loss = compute_multi_aspect_loss(
                        logits, labels, aspect_mask, is_virtual,
                        virtual_weight=args.virtual_weight
                    )

                    # 對 CE Loss 也支持對比學習
                    if use_contrastive:
                        if 'contrastive_loss_fn' not in locals():
                            contrastive_loss_fn = MultiAspectContrastiveLoss(
                                temperature=args.contrastive_temperature,
                                contrastive_weight=args.contrastive_weight
                            ).to(device)

                        if extras is not None and 'features' in extras:
                            contrastive_loss = contrastive_loss_fn(
                                extras['features'], labels, aspect_mask
                            )
                            loss = loss + contrastive_loss

                # 梯度累積：將 loss 除以累積步數
                loss = loss / args.accumulation_steps

            # Backward (with AMP scaler for FP16, direct backward for BF16/FP32)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 每 accumulation_steps 步更新一次參數
            if (batch_idx + 1) % args.accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
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

        # 關閉進度條，避免輸出混亂
        progress_bar.close()

        avg_train_loss = train_loss / train_steps

        # 驗證階段 - 傳入 loss_fn 和 args 以計算 validation loss
        # 對於 distillation，驗證時使用 focal loss (不使用 soft labels)
        if args.loss_type == 'distill':
            loss_fn_for_eval = distill_loss_fn if 'distill_loss_fn' in locals() else None
        elif args.loss_type != 'ce':
            loss_fn_for_eval = loss_fn if 'loss_fn' in locals() else None
        else:
            loss_fn_for_eval = None
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

        # 簡化的 epoch 輸出（AGG 風格）
        val_f1 = val_metrics['f1_macro']
        val_acc = val_metrics['accuracy']
        f1_per_class = val_metrics['f1_per_class']

        # Early stopping 檢查
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            save_path = checkpoints_dir / f'best_model_epoch{epoch+1}_f1_{best_val_f1:.4f}.pt'
            torch.save(model.state_dict(), save_path)
            status = f"★ Best"
        else:
            patience_counter += 1
            status = f"P:{patience_counter}/{args.patience}"

        # 單行輸出：Epoch | Loss | Acc | F1 | Per-class F1 | Status
        print(f"  [{epoch+1:02d}/{args.epochs}] Loss:{avg_train_loss:.4f} | Acc:{val_acc:.4f} | "
              f"F1:{val_f1:.4f} [N:{f1_per_class[0]:.2f} U:{f1_per_class[1]:.2f} P:{f1_per_class[2]:.2f}] | {status}")

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # 測試階段
    print("\n" + "-"*60)
    print(f"Testing (Best: Epoch {best_epoch}, Val F1: {best_val_f1:.4f})")
    print("-"*60)

    # 加載最佳模型
    best_model_path = checkpoints_dir / f'best_model_epoch{best_epoch}_f1_{best_val_f1:.4f}.pt'
    model.load_state_dict(torch.load(best_model_path))

    # 測試集評估（也計算 loss）
    loss_fn_for_eval = loss_fn if args.loss_type != 'ce' else None
    test_metrics = evaluate_multi_aspect(
        model, test_loader, device, loss_fn_for_eval, args,
        collect_gates=False,
        return_confusion_matrix=True  # 收集混淆矩陣用於視覺化
    )

    # 簡化的測試結果輸出
    test_f1_class = test_metrics['f1_per_class']
    print(f"  Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1_macro']:.4f} "
          f"[N:{test_f1_class[0]:.2f} U:{test_f1_class[1]:.2f} P:{test_f1_class[2]:.2f}]")

    # HBL layer attention 權重（如果有）
    if 'layer_attention' in test_metrics:
        w = test_metrics['layer_attention']
        print(f"  Layer Weights: Low={w[0]:.3f} Mid={w[1]:.3f} High={w[2]:.3f}")

    # AH-IARN 分支權重（如果有）
    if 'branch_weights' in test_metrics:
        bw = test_metrics['branch_weights']
        print(f"  Branch Weights: Hier={bw['Hierarchical']:.3f} HSA={bw['HSA']:.3f} IARN={bw['IARN']:.3f}")

    # 保存結果
    results = {
        'args': vars(args),
        'best_val_f1': float(best_val_f1),
        'history': history,
        'test_metrics': {k: float(v) if isinstance(v, (float, np.float32, np.float64)) else v
                        for k, v in test_metrics.items()}
    }

    # 單獨保存 layer_attention 到頂層，方便報告生成器讀取
    if 'layer_attention' in test_metrics:
        results['layer_attention'] = test_metrics['layer_attention']

    # 單獨保存 branch_weights 和 hsa_level_weights 到頂層
    if 'branch_weights' in test_metrics:
        results['branch_weights'] = test_metrics['branch_weights']
    if 'hsa_level_weights' in test_metrics:
        results['hsa_level_weights'] = test_metrics['hsa_level_weights']

    # 保存結果到時間戳資料夾
    results_path = reports_dir / 'experiment_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    # 保存實驗配置
    config_path = reports_dir / 'experiment_config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2, default=str)

    # 自動生成訓練曲線圖表（靜默模式）
    try:
        generate_training_visualizations(results, visualizations_dir)
    except Exception as e:
        print(f"  [WARNING] Visualization failed: {e}")

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
            f.write(f"  模型類型: Improved - {args.improved}\n")
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

    # 完成訊息
    print("\n" + "="*60)
    print(f"DONE | Test F1: {test_metrics['f1_macro']:.4f} | Saved: {exp_dir.name}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='訓練 Multi-Aspect HMAC-Net')

    # 數據參數
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['restaurants', 'laptops', 'mams',
                                 'rest16', 'lap16',
                                 'memd_books', 'memd_clothing', 'memd_hotel',
                                 'memd_laptop', 'memd_restaurant'],
                        help='數據集選擇 (restaurants, laptops, mams, rest16, lap16, 或 memd_* 系列)')
    parser.add_argument('--use_augmented', action='store_true', default=False,
                        help='使用增強數據集 (Claude Augmentation)')
    parser.add_argument('--augmented_dir', type=str, default=None,
                        help='增強數據目錄（若不指定，自動根據 dataset 設置）')
    parser.add_argument('--use_self_training', action='store_true', default=False,
                        help='使用 Self-Training 偽標籤數據')
    parser.add_argument('--min_aspects', type=int, default=2,
                        help='最小 aspect 數量（用於過濾）')
    parser.add_argument('--max_aspects', type=int, default=8,
                        help='最大 aspect 數量（超過則截斷）')
    parser.add_argument('--include_single_aspect', action='store_true', default=True,
                        help='是否包含單 aspect（帶虛擬 aspect）')
    parser.add_argument('--no_include_single_aspect', action='store_false', dest='include_single_aspect',
                        help='禁用單 aspect 樣本')
    parser.add_argument('--virtual_aspect_mode', type=str, default='overall',
                        choices=['overall', 'context', 'none'],
                        help='虛擬 aspect 模式')
    parser.add_argument('--max_text_len', type=int, default=128,
                        help='最大文本長度')
    parser.add_argument('--max_aspect_len', type=int, default=10,
                        help='最大 aspect 長度')

    # 模型類型選擇
    parser.add_argument('--baseline', type=str, default=None,
                        choices=['bert_cls', 'bert_only'],  # bert_only 為向後兼容
                        help='使用 baseline 模型 (bert_cls: 標準BERT-CLS baseline)')
    parser.add_argument('--improved', type=str, default=None,
                        choices=['hierarchical', 'hierarchical_layerattn', 'aspect_aware', 'iarn', 'vp_iarn', 'hsa', 'hkgan'],
                        help='使用改進模型:\n'
                             '  - hierarchical: Hierarchical BERT (基礎階層模型)\n'
                             '  - hierarchical_layerattn: Hierarchical BERT with Layer Attention\n'
                             '  - aspect_aware: Aspect-aware Hierarchical BERT\n'
                             '  - iarn: Inter-Aspect Relation Network\n'
                             '  - vp_iarn: VP-IARN (向量投影 + IARN)\n'
                             '  - hsa: Hierarchical Syntax Attention (aspect_aware 別名)\n'
                             '  - hkgan: Hierarchical Knowledge-enhanced GAT Network (推薦)')
    parser.add_argument('--auto_select', action='store_true',
                        help='根據數據集特徵自動選擇最佳模型 (Hierarchical BERT 或 IARN)')

    # 模型參數
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='BERT 模型名稱')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='是否凍結 BERT')
    parser.add_argument('--hidden_dim', type=int, default=768,
                        help='隱藏層維度（建議保持 768）')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout 比率')
    parser.add_argument('--multi_aspect_threshold', type=float, default=0.5,
                        help='Unified-HIARN: 多面向比例閾值 (>閾值使用IARN, <=閾值使用Hierarchical)')

    # Adaptive Hierarchical Net 參數
    parser.add_argument('--selection_mode', type=str, default='soft',
                        choices=['soft', 'hard'],
                        help='Adaptive Hierarchical: 分支選擇模式 (soft=加權組合, hard=Gumbel-Softmax硬選擇)')
    parser.add_argument('--num_attention_heads', type=int, default=4,
                        help='IARN/Adaptive: Multi-head attention 頭數')
    parser.add_argument('--num_syntax_layers', type=int, default=2,
                        help='HSA/Adaptive: 語法傳播層數')

    # HKGAN (Hierarchical Knowledge-enhanced GAT) 參數
    parser.add_argument('--gat_heads', type=int, default=4,
                        help='HKGAN: GAT 注意力頭數')
    parser.add_argument('--gat_layers', type=int, default=2,
                        help='HKGAN: GAT 層數')
    parser.add_argument('--knowledge_weight', type=float, default=0.1,
                        help='HKGAN: SenticNet 知識注入權重')
    parser.add_argument('--use_senticnet', action='store_true', default=True,
                        help='HKGAN: 是否使用 SenticNet 知識增強')

    # HKGAN v2.0 新增：Neutral 識別改進
    parser.add_argument('--use_confidence_gate', action='store_true', default=True,
                        help='HKGAN: 是否使用信心門控（動態調整 SenticNet 注入強度）')
    parser.add_argument('--no_confidence_gate', action='store_true', default=False,
                        help='HKGAN: 禁用信心門控')
    parser.add_argument('--domain', type=str, default=None,
                        help='HKGAN: 領域名稱，用於領域過濾（如不指定，將自動根據數據集推斷）')

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
                        choices=['ce', 'focal', 'adaptive', 'distill'],
                        help='Loss function type: ce (CrossEntropy), focal (FocalLoss), adaptive (AdaptiveWeighted), distill (Knowledge Distillation)')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Focal loss gamma parameter (default: 2.0, 0=CE)')
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (default: 0.0, range: 0.0-1.0)')
    parser.add_argument('--class_weights', type=float, nargs=3, default=None,
                        help='Class weights [neg, neu, pos], e.g., --class_weights 1.0 2.0 1.0')
    parser.add_argument('--auto_class_weights', action='store_true',
                        help='動態計算 class weights (根據數據集分布自動調整)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子（default: 42）')

    # 對比學習參數
    parser.add_argument('--contrastive_weight', type=float, default=0.0,
                        help='對比損失權重 (0.0=禁用, 推薦 0.1-0.3)')
    parser.add_argument('--contrastive_temperature', type=float, default=0.07,
                        help='對比學習溫度參數 (default: 0.07)')

    # Layer-wise Learning Rate Decay (LLRD)
    parser.add_argument('--use_llrd', action='store_true',
                        help='啟用 Layer-wise Learning Rate Decay')
    parser.add_argument('--llrd_decay', type=float, default=0.95,
                        help='LLRD 衰減係數 (default: 0.95，每往下一層 LR 乘上此係數)')

    # 非對稱 Logit 調整 (推理時)
    parser.add_argument('--neutral_boost', type=float, default=0.0,
                        help='Neutral logits 偏移量 (解決 Neu→Pos, 推薦 0.5~1.0)')
    parser.add_argument('--neg_suppress', type=float, default=0.0,
                        help='Negative logits 抑制量 (解決 Neu→Neg, 筆電領域推薦 0.3~0.8)')
    parser.add_argument('--pos_suppress', type=float, default=0.0,
                        help='Positive logits 抑制量 (解決 Neu→Pos, 餐廳領域推薦 0.5~1.0)')

    # 知識蒸餾參數 (Knowledge Distillation)
    parser.add_argument('--distill_alpha', type=float, default=0.3,
                        help='Soft label 權重 (0.0=純 hard label, 1.0=純 soft label, 推薦 0.2-0.4)')
    parser.add_argument('--distill_temperature', type=float, default=2.0,
                        help='蒸餾溫度 (較高值產生更平滑的分布, 推薦 2.0-4.0)')
    parser.add_argument('--soft_labels_file', type=str, default=None,
                        help='Soft labels JSON 文件路徑 (相對於 data/soft_labels/)')

    args = parser.parse_args()

    # 設置隨機種子（確保可重現性）
    import random
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # 確保 CUDA 的確定性行為
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 簡化的啟動資訊
    print("\n" + "="*60)
    print("Multi-Aspect ABSA Training")
    print("="*60)

    # 只顯示關鍵配置
    model_type = args.baseline if args.baseline else args.improved
    print(f"  Model: {model_type} | Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size}x{args.accumulation_steps} | LR: {args.lr}")
    print(f"  Loss: {args.loss_type} | Patience: {args.patience} | Seed: {args.seed}")

    train_multiaspect_model(args)


if __name__ == '__main__':
    main()
