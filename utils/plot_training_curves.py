"""
訓練曲線可視化工具
從訓練歷史數據生成訓練曲線圖表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# 設置中文字體和風格
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")


def load_results(results_path):
    """加載訓練結果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def plot_loss_curves(history, save_dir):
    """
    繪製訓練和驗證損失曲線
    """
    epochs = history['epochs']
    train_loss = history['train_loss']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-o', label='Training Loss', linewidth=2, markersize=6)

    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Loss', fontsize=14, fontweight='bold')
    plt.title('Training Loss Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = save_dir / 'training_loss_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Loss curve: {save_path}")
    plt.close()


def plot_accuracy_curve(history, save_dir):
    """
    繪製驗證準確率曲線
    """
    epochs = history['epochs']
    val_accuracy = history['val_accuracy']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_accuracy, 'g-o', label='Validation Accuracy', linewidth=2, markersize=6)

    # 標記最佳epoch
    best_epoch = np.argmax(val_accuracy) + 1
    best_acc = max(val_accuracy)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.text(best_epoch, best_acc, f' {best_acc:.4f}', fontsize=11,
             verticalalignment='bottom', color='red', fontweight='bold')

    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=14, fontweight='bold')
    plt.title('Validation Accuracy Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([min(val_accuracy) - 0.02, max(val_accuracy) + 0.02])
    plt.tight_layout()

    save_path = save_dir / 'validation_accuracy_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Accuracy curve: {save_path}")
    plt.close()


def plot_f1_curve(history, save_dir):
    """
    繪製驗證F1曲線
    """
    epochs = history['epochs']
    val_f1 = history['val_f1_macro']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_f1, 'm-o', label='Validation F1 (Macro)', linewidth=2, markersize=6)

    # 標記最佳epoch
    best_epoch = np.argmax(val_f1) + 1
    best_f1 = max(val_f1)
    plt.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
    plt.text(best_epoch, best_f1, f' {best_f1:.4f}', fontsize=11,
             verticalalignment='bottom', color='red', fontweight='bold')

    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('F1 Score (Macro)', fontsize=14, fontweight='bold')
    plt.title('Validation F1 Score Curve', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.ylim([min(val_f1) - 0.02, max(val_f1) + 0.02])
    plt.tight_layout()

    save_path = save_dir / 'validation_f1_curve.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] F1 curve: {save_path}")
    plt.close()


def plot_per_class_f1_curves(history, save_dir):
    """
    繪製每個類別的F1曲線
    """
    epochs = history['epochs']
    f1_per_class = np.array(history['val_f1_per_class'])  # [num_epochs, 3]

    class_names = ['Negative', 'Neutral', 'Positive']
    colors = ['#e74c3c', '#f39c12', '#27ae60']

    plt.figure(figsize=(12, 7))

    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        f1_scores = f1_per_class[:, i]
        plt.plot(epochs, f1_scores, marker='o', label=f'{class_name} F1',
                color=color, linewidth=2, markersize=6)

        # 標記最終值
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
    print(f"[SAVED] Per-class F1 curves: {save_path}")
    plt.close()


def plot_comprehensive_metrics(history, save_dir):
    """
    繪製綜合指標圖（2x2子圖）
    """
    epochs = history['epochs']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Training Metrics', fontsize=18, fontweight='bold', y=0.995)

    # 1. 訓練損失
    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', linewidth=2, markersize=5)
    axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 驗證準確率
    val_acc = history['val_accuracy']
    axes[0, 1].plot(epochs, val_acc, 'g-o', linewidth=2, markersize=5)
    best_epoch = np.argmax(val_acc) + 1
    axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Validation Accuracy (Best: Epoch {best_epoch})', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 驗證F1
    val_f1 = history['val_f1_macro']
    axes[1, 0].plot(epochs, val_f1, 'm-o', linewidth=2, markersize=5)
    best_epoch_f1 = np.argmax(val_f1) + 1
    axes[1, 0].axvline(x=best_epoch_f1, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'Validation F1 Macro (Best: Epoch {best_epoch_f1})', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 每類F1
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
    print(f"[SAVED] Comprehensive metrics: {save_path}")
    plt.close()


def plot_precision_recall_curves(history, save_dir):
    """
    繪製Precision和Recall曲線
    """
    epochs = history['epochs']
    val_precision = history['val_precision']
    val_recall = history['val_recall']

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_precision, 'b-o', label='Precision', linewidth=2, markersize=6)
    plt.plot(epochs, val_recall, 'r-s', label='Recall', linewidth=2, markersize=6)

    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Score', fontsize=14, fontweight='bold')
    plt.title('Precision and Recall Curves', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = save_dir / 'precision_recall_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"[SAVED] Precision/Recall curves: {save_path}")
    plt.close()


def generate_training_report(results, save_dir):
    """
    生成訓練報告文本文件
    """
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("Multi-Aspect HMAC-Net Training Report")
    report_lines.append("="*80)
    report_lines.append("")

    # 配置信息
    args = results['args']
    report_lines.append("Configuration:")
    report_lines.append(f"  Model: {args['bert_model']}")
    report_lines.append(f"  Hidden Dim: {args['hidden_dim']}")
    report_lines.append(f"  PMAC: {'Enabled' if args['use_pmac'] else 'Disabled'} ({args['pmac_mode']})")
    report_lines.append(f"  IARM: {'Enabled' if args['use_iarm'] else 'Disabled'} ({args['iarm_mode']})")
    report_lines.append(f"  Loss Type: {args.get('loss_type', 'ce')}")
    if args.get('loss_type') == 'focal':
        report_lines.append(f"  Focal Gamma: {args.get('focal_gamma', 2.0)}")
        report_lines.append(f"  Class Weights: {args.get('class_weights', [1.0, 1.0, 1.0])}")
    report_lines.append(f"  Batch Size: {args['batch_size']}")
    report_lines.append(f"  Learning Rate: {args['lr']}")
    report_lines.append(f"  Epochs: {args['epochs']}")
    report_lines.append("")

    # 訓練歷史
    history = results['history']
    report_lines.append("Training History:")
    report_lines.append(f"  Total Epochs Trained: {len(history['epochs'])}")
    report_lines.append("")

    # 每個epoch的詳細數據
    report_lines.append("Epoch-by-Epoch Metrics:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Epoch':<8} {'Train Loss':<12} {'Val Acc':<10} {'Val F1':<10} {'Neg F1':<10} {'Neu F1':<10} {'Pos F1':<10}")
    report_lines.append("-" * 80)

    for i, epoch in enumerate(history['epochs']):
        train_loss = history['train_loss'][i]
        val_acc = history['val_accuracy'][i]
        val_f1 = history['val_f1_macro'][i]
        f1_per_class = history['val_f1_per_class'][i]

        report_lines.append(
            f"{epoch:<8} {train_loss:<12.4f} {val_acc:<10.4f} {val_f1:<10.4f} "
            f"{f1_per_class[0]:<10.4f} {f1_per_class[1]:<10.4f} {f1_per_class[2]:<10.4f}"
        )

    report_lines.append("-" * 80)
    report_lines.append("")

    # 最佳驗證結果
    best_val_f1_idx = np.argmax(history['val_f1_macro'])
    best_epoch = history['epochs'][best_val_f1_idx]
    report_lines.append("Best Validation Results:")
    report_lines.append(f"  Best Epoch: {best_epoch}")
    report_lines.append(f"  Train Loss: {history['train_loss'][best_val_f1_idx]:.4f}")
    report_lines.append(f"  Val Accuracy: {history['val_accuracy'][best_val_f1_idx]:.4f}")
    report_lines.append(f"  Val F1 (Macro): {history['val_f1_macro'][best_val_f1_idx]:.4f}")
    report_lines.append(f"  Val Precision: {history['val_precision'][best_val_f1_idx]:.4f}")
    report_lines.append(f"  Val Recall: {history['val_recall'][best_val_f1_idx]:.4f}")
    best_f1_per_class = history['val_f1_per_class'][best_val_f1_idx]
    report_lines.append(f"  Val F1 per class (Neg/Neu/Pos): [{best_f1_per_class[0]:.4f}, {best_f1_per_class[1]:.4f}, {best_f1_per_class[2]:.4f}]")
    report_lines.append("")

    # 測試結果
    test_metrics = results['test_metrics']
    report_lines.append("Test Results:")
    report_lines.append(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    report_lines.append(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    report_lines.append(f"  Precision: {test_metrics['precision']:.4f}")
    report_lines.append(f"  Recall: {test_metrics['recall']:.4f}")
    test_f1_per_class = test_metrics['f1_per_class']
    report_lines.append(f"  F1 per class:")
    report_lines.append(f"    Negative: {test_f1_per_class[0]:.4f}")
    report_lines.append(f"    Neutral:  {test_f1_per_class[1]:.4f}")
    report_lines.append(f"    Positive: {test_f1_per_class[2]:.4f}")
    report_lines.append("")
    report_lines.append("="*80)

    # 保存報告
    report_path = save_dir / 'training_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"[SAVED] Training report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='繪製訓練曲線圖表')
    parser.add_argument('--results_path', type=str,
                       default='results/reports/multiaspect_results.json',
                       help='訓練結果JSON文件路徑')
    parser.add_argument('--save_dir', type=str,
                       default='results/visualizations',
                       help='圖表保存目錄')
    args = parser.parse_args()

    # 設置路徑
    project_root = Path(__file__).parent.parent
    results_path = project_root / args.results_path
    save_dir = project_root / args.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    # 加載結果
    print(f"\n加載訓練結果: {results_path}")
    results = load_results(results_path)

    # 檢查是否有history數據
    if 'history' not in results:
        print("[ERROR] No training history found in results!")
        print("Please retrain the model with the updated training script.")
        return

    history = results['history']

    print(f"\n生成訓練曲線圖表...")
    print(f"總共訓練了 {len(history['epochs'])} 個 epochs")
    print("")

    # 生成各種圖表
    plot_loss_curves(history, save_dir)
    plot_accuracy_curve(history, save_dir)
    plot_f1_curve(history, save_dir)
    plot_per_class_f1_curves(history, save_dir)
    plot_precision_recall_curves(history, save_dir)
    plot_comprehensive_metrics(history, save_dir)

    # 生成訓練報告
    generate_training_report(results, save_dir)

    print(f"\n[COMPLETE] All visualizations saved to: {save_dir}")
    print(f"\nGenerated files:")
    print(f"  - training_loss_curve.png")
    print(f"  - validation_accuracy_curve.png")
    print(f"  - validation_f1_curve.png")
    print(f"  - per_class_f1_curves.png")
    print(f"  - precision_recall_curves.png")
    print(f"  - comprehensive_training_metrics.png")
    print(f"  - training_report.txt")


if __name__ == '__main__':
    main()
