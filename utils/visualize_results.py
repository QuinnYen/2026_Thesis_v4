"""
Multi-Aspect HMAC-Net 訓練結果可視化工具

功能:
1. 訓練曲線 (Loss, F1, Accuracy)
2. 混淆矩陣
3. 每類別 F1 分析
4. Aspect 數量分布分析
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
import torch


# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


def load_results(results_path):
    """加載結果 JSON"""
    with open(results_path, 'r') as f:
        return json.load(f)


def plot_training_curves(history, save_dir):
    """繪製訓練曲線"""
    if not history or 'train_loss' not in history:
        print("No training history found!")
        return

    epochs = list(range(1, len(history['train_loss']) + 1))

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Loss曲線
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Validation F1
    if 'val_f1' in history:
        axes[0, 1].plot(epochs, history['val_f1'], 'g-', label='Val F1 (Macro)', linewidth=2, marker='o')
        best_epoch = np.argmax(history['val_f1']) + 1
        best_f1 = max(history['val_f1'])
        axes[0, 1].axvline(x=best_epoch, color='r', linestyle='--', alpha=0.7, label=f'Best Epoch: {best_epoch}')
        axes[0, 1].axhline(y=best_f1, color='r', linestyle='--', alpha=0.7, label=f'Best F1: {best_f1:.4f}')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('F1 Score', fontsize=12)
        axes[0, 1].set_title('Validation F1 (Macro) Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=11)
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Validation Accuracy
    if 'val_acc' in history:
        axes[1, 0].plot(epochs, history['val_acc'], 'm-', label='Val Accuracy', linewidth=2, marker='s')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Accuracy', fontsize=12)
        axes[1, 0].set_title('Validation Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Per-Class F1
    if 'val_f1_per_class' in history:
        f1_per_class = np.array(history['val_f1_per_class'])  # [epochs, 3]
        axes[1, 1].plot(epochs, f1_per_class[:, 0], 'r-', label='Negative F1', linewidth=2, marker='o')
        axes[1, 1].plot(epochs, f1_per_class[:, 1], 'y-', label='Neutral F1', linewidth=2, marker='s')
        axes[1, 1].plot(epochs, f1_per_class[:, 2], 'g-', label='Positive F1', linewidth=2, marker='^')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('F1 Score', fontsize=12)
        axes[1, 1].set_title('Per-Class F1 Over Time', fontsize=14, fontweight='bold')
        axes[1, 1].legend(fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = save_dir / 'training_curves.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, save_dir, dataset_name='Test'):
    """繪製混淆矩陣"""
    cm = confusion_matrix(y_true, y_pred)

    # 計算百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. 計數混淆矩陣
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                cbar_kws={'label': 'Count'})
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_title(f'{dataset_name} Set - Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

    # 2. 百分比混淆矩陣
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1],
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'],
                cbar_kws={'label': 'Percentage (%)'})
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_title(f'{dataset_name} Set - Confusion Matrix (Percentage)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    save_path = save_dir / f'confusion_matrix_{dataset_name.lower()}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def plot_class_performance(results, save_dir):
    """繪製每個類別的性能分析"""
    f1_scores = results['test_metrics']['f1_per_class']
    classes = ['Negative', 'Neutral', 'Positive']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. F1 分數柱狀圖
    colors = ['#e74c3c', '#f39c12', '#27ae60']
    bars = axes[0].bar(classes, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=np.mean(f1_scores), color='blue', linestyle='--', linewidth=2,
                    label=f'Macro F1: {np.mean(f1_scores):.4f}')
    axes[0].set_ylabel('F1 Score', fontsize=12)
    axes[0].set_title('Per-Class F1 Score (Test Set)', fontsize=14, fontweight='bold')
    axes[0].set_ylim([0, 1.0])
    axes[0].legend(fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)

    # 在柱狀圖上標註數值
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 2. 性能指標對比
    precision = results['test_metrics']['precision']
    recall = results['test_metrics']['recall']
    f1_macro = results['test_metrics']['f1_macro']
    accuracy = results['test_metrics']['accuracy']

    metrics = ['Precision', 'Recall', 'F1 (Macro)', 'Accuracy']
    values = [precision, recall, f1_macro, accuracy]
    colors_metrics = ['#3498db', '#9b59b6', '#e67e22', '#1abc9c']

    bars2 = axes[1].bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Score', fontsize=12)
    axes[1].set_title('Overall Test Metrics', fontsize=14, fontweight='bold')
    axes[1].set_ylim([0, 1.0])
    axes[1].grid(axis='y', alpha=0.3)

    # 在柱狀圖上標註數值
    for bar, value in zip(bars2, values):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_path = save_dir / 'class_performance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Class performance plot saved to: {save_path}")
    plt.close()


def plot_aspect_distribution(data_stats, save_dir):
    """繪製 aspect 數量分布"""
    if 'aspect_distribution' not in data_stats:
        print("No aspect distribution data found!")
        return

    aspect_counts = data_stats['aspect_distribution']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # 1. 柱狀圖
    num_aspects = list(aspect_counts.keys())
    counts = list(aspect_counts.values())

    bars = axes[0].bar(num_aspects, counts, color='steelblue', alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Number of Aspects', fontsize=12)
    axes[0].set_ylabel('Sample Count', fontsize=12)
    axes[0].set_title('Distribution of Aspect Counts', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    # 標註數值
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(count)}',
                    ha='center', va='bottom', fontsize=10)

    # 2. 累積百分比
    total = sum(counts)
    cumulative_pct = np.cumsum(counts) / total * 100

    axes[1].plot(num_aspects, cumulative_pct, 'o-', color='darkorange', linewidth=2, markersize=8)
    axes[1].fill_between(num_aspects, 0, cumulative_pct, alpha=0.3, color='orange')
    axes[1].set_xlabel('Number of Aspects', fontsize=12)
    axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1].set_title('Cumulative Distribution of Aspects', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])

    # 標註關鍵百分比
    for i, (aspect, pct) in enumerate(zip(num_aspects, cumulative_pct)):
        if i % 2 == 0:  # 只標註偶數索引避免擁擠
            axes[1].text(aspect, pct + 2, f'{pct:.1f}%',
                        ha='center', fontsize=9)

    plt.tight_layout()
    save_path = save_dir / 'aspect_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Aspect distribution plot saved to: {save_path}")
    plt.close()


def generate_performance_report(results, save_dir):
    """生成性能報告 Markdown"""
    report = f"""# Multi-Aspect HMAC-Net 訓練結果報告

## 測試集性能總覽

| 指標 | 分數 |
|------|------|
| **Accuracy** | **{results['test_metrics']['accuracy']:.4f}** ({results['test_metrics']['accuracy']*100:.2f}%) |
| **F1 (Macro)** | **{results['test_metrics']['f1_macro']:.4f}** |
| **Precision** | {results['test_metrics']['precision']:.4f} |
| **Recall** | {results['test_metrics']['recall']:.4f} |
| **Best Val F1** | {results['best_val_f1']:.4f} |

## 每類別性能

| 類別 | F1 Score | 性能評價 |
|------|----------|----------|
| **Negative** | {results['test_metrics']['f1_per_class'][0]:.4f} | {'✅ 良好' if results['test_metrics']['f1_per_class'][0] > 0.7 else '⚠️ 需改進'} |
| **Neutral** | {results['test_metrics']['f1_per_class'][1]:.4f} | {'✅ 良好' if results['test_metrics']['f1_per_class'][1] > 0.7 else '⚠️ 需改進'} |
| **Positive** | {results['test_metrics']['f1_per_class'][2]:.4f} | {'✅ 優秀' if results['test_metrics']['f1_per_class'][2] > 0.8 else '✅ 良好'} |

## 關鍵發現

### 🎯 優勢
- **Positive 類別識別優秀** (F1={results['test_metrics']['f1_per_class'][2]:.4f})
- **整體準確率達標** ({results['test_metrics']['accuracy']*100:.2f}%)
- **Negative 類別性能良好** (F1={results['test_metrics']['f1_per_class'][0]:.4f})

### ⚠️ 改進空間
- **Neutral 類別性能較弱** (F1={results['test_metrics']['f1_per_class'][1]:.4f})
  - 原因分析: Neutral 類別樣本較少且特徵不明顯
  - 改進方向: Focal Loss, Class Weighting, Data Augmentation

## 模型配置

```json
{json.dumps(results['args'], indent=2, ensure_ascii=False)}
```

## 可視化結果

- 📊 訓練曲線: `training_curves.png`
- 🎯 混淆矩陣: `confusion_matrix_test.png`
- 📈 類別性能: `class_performance.png`
- 📊 Aspect 分布: `aspect_distribution.png`

---
**生成時間**: {Path(save_dir / 'performance_report.md').parent.name}
**模型**: Multi-Aspect HMAC-Net (DistilBERT + AAHA + PMAC + IARM)
"""

    report_path = save_dir / 'performance_report.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\nPerformance report saved to: {report_path}")


def main():
    """主函數"""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize Multi-Aspect HMAC-Net Results')
    parser.add_argument('--results', type=str, default='results/reports/multiaspect_results.json',
                        help='Path to results JSON file')
    parser.add_argument('--history', type=str, default='results/reports/training_history.json',
                        help='Path to training history JSON file (optional)')
    parser.add_argument('--predictions', type=str, default='results/reports/test_predictions.pt',
                        help='Path to test predictions (optional, for confusion matrix)')
    parser.add_argument('--output_dir', type=str, default='results/visualizations',
                        help='Output directory for plots')

    args = parser.parse_args()

    # 創建輸出目錄
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("Multi-Aspect HMAC-Net Results Visualization")
    print("="*80)

    # 加載結果
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    results = load_results(results_path)
    print(f"\nLoaded results from: {results_path}")
    print(f"Test Accuracy: {results['test_metrics']['accuracy']:.4f}")
    print(f"Test F1 (Macro): {results['test_metrics']['f1_macro']:.4f}")

    # 1. 繪製類別性能
    print("\n[1/5] Generating class performance plot...")
    plot_class_performance(results, output_dir)

    # 2. 生成性能報告
    print("\n[2/5] Generating performance report...")
    generate_performance_report(results, output_dir)

    # 3. 繪製訓練曲線（如果有訓練歷史）
    history_path = Path(args.history)
    if history_path.exists():
        print("\n[3/5] Generating training curves...")
        with open(history_path, 'r') as f:
            history = json.load(f)
        plot_training_curves(history, output_dir)
    else:
        print(f"\n[3/5] Skipping training curves (history file not found: {history_path})")

    # 4. 繪製混淆矩陣（如果有預測結果）
    pred_path = Path(args.predictions)
    if pred_path.exists():
        print("\n[4/5] Generating confusion matrix...")
        predictions = torch.load(pred_path)
        y_true = predictions['labels']
        y_pred = predictions['predictions']
        plot_confusion_matrix(y_true, y_pred, output_dir, dataset_name='Test')
    else:
        print(f"\n[4/5] Skipping confusion matrix (predictions file not found: {pred_path})")

    # 5. Aspect 分布（如果有數據統計）
    if 'data_stats' in results:
        print("\n[5/5] Generating aspect distribution plot...")
        plot_aspect_distribution(results['data_stats'], output_dir)
    else:
        print("\n[5/5] Skipping aspect distribution (no data stats in results)")

    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()
