"""
生成 Unified-HIARN 獨立實驗報告

Method 4: Unified-HIARN (統一 Hierarchical + IARN)

核心特點:
- 數據集級別策略選擇: 根據多面向比例自動選擇 Hierarchical 或 IARN
- 多面向比例 > 50%: 使用 IARN 模式 (aspect 間交互)
- 多面向比例 <= 50%: 使用 Hierarchical 模式 (純階層特徵)
- 統一模型，無需手動選擇

報告內容:
1. 模型架構說明
2. 實驗配置
3. 實驗結果 (Acc, Macro-F1, Per-class F1)
4. 策略分析 (數據集級別選擇結果)
5. 與其他模型對比 (如果有)

使用方法:
    python experiments/generate_unified_hiarn_report.py --dataset restaurants
    python experiments/generate_unified_hiarn_report.py --dataset laptops
    python experiments/generate_unified_hiarn_report.py --dataset mams
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def find_unified_hiarn_experiment(results_dir, dataset):
    """查找該數據集下最新的 Unified-HIARN 實驗"""
    improved_dir = results_dir / "improved" / dataset

    if not improved_dir.exists():
        return None

    latest_exp = None
    latest_time = None

    for exp_dir in improved_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        dir_name = exp_dir.name
        if '_improved_unified_hiarn_' in dir_name:
            mtime = exp_dir.stat().st_mtime
            if latest_time is None or mtime > latest_time:
                latest_exp = exp_dir
                latest_time = mtime

    return latest_exp


def find_baseline_experiment(results_dir, dataset):
    """查找 Baseline 實驗"""
    baseline_dir = results_dir / "baseline" / dataset

    if not baseline_dir.exists():
        return None

    latest_exp = None
    latest_time = None

    for exp_dir in baseline_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        if '_baseline_bert_cls_' in exp_dir.name:
            mtime = exp_dir.stat().st_mtime
            if latest_time is None or mtime > latest_time:
                latest_exp = exp_dir
                latest_time = mtime

    return latest_exp


def read_metrics(exp_dir):
    """從實驗目錄讀取指標"""
    metrics = {
        'test_acc': None,
        'test_f1': None,
        'test_f1_neg': None,
        'test_f1_neu': None,
        'test_f1_pos': None,
        'val_f1': None,
        'best_epoch': None,
        'total_epochs': None,
        'patience': None,
        'class_weights': None,
        'focal_gamma': None,
        'timestamp': None,
        'exp_name': exp_dir.name,
        # Unified-HIARN 特有
        'avg_fusion_gate': None,
        'avg_gate_multi_aspect': None,
        'avg_gate_single_aspect': None,
        'n_multi_aspect_samples': None,
        'n_single_aspect_samples': None,
        'base_hierarchical_weight': None,
    }

    # 從 experiment_config.json 讀取訓練配置
    config_file = exp_dir / "reports" / "experiment_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                training_config = config.get('training', {})
                metrics['total_epochs'] = training_config.get('epochs')
                metrics['patience'] = training_config.get('patience')
                metrics['class_weights'] = training_config.get('class_weights')
                metrics['focal_gamma'] = training_config.get('focal_gamma')
        except Exception as e:
            print(f"    讀取配置錯誤: {e}")

    # 從 experiment_results.json 讀取
    results_file = exp_dir / "reports" / "experiment_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 讀取 test_metrics
                test_metrics = data.get('test_metrics', {})
                metrics['test_acc'] = test_metrics.get('accuracy')
                metrics['test_f1'] = test_metrics.get('f1_macro')

                # 讀取各類別的 F1 值
                f1_per_class = test_metrics.get('f1_per_class', [])
                if len(f1_per_class) >= 3:
                    metrics['test_f1_neg'] = f1_per_class[0]
                    metrics['test_f1_neu'] = f1_per_class[1]
                    metrics['test_f1_pos'] = f1_per_class[2]

                # 讀取 validation F1
                metrics['val_f1'] = data.get('best_val_f1')

                # 讀取 best epoch
                if 'best_val_f1' in data and 'history' in data:
                    val_f1_list = data['history'].get('val_f1_macro', [])
                    if val_f1_list:
                        best_f1 = data['best_val_f1']
                        for i, f1 in enumerate(val_f1_list):
                            if abs(f1 - best_f1) < 1e-6:
                                metrics['best_epoch'] = i + 1
                                break

                # Unified-HIARN 特有指標
                if 'avg_fusion_gate' in data:
                    metrics['avg_fusion_gate'] = data['avg_fusion_gate']
                if 'avg_gate_multi_aspect' in data:
                    metrics['avg_gate_multi_aspect'] = data['avg_gate_multi_aspect']
                if 'avg_gate_single_aspect' in data:
                    metrics['avg_gate_single_aspect'] = data['avg_gate_single_aspect']
                if 'n_multi_aspect_samples' in data:
                    metrics['n_multi_aspect_samples'] = data['n_multi_aspect_samples']
                if 'n_single_aspect_samples' in data:
                    metrics['n_single_aspect_samples'] = data['n_single_aspect_samples']
                if 'base_hierarchical_weight' in data:
                    metrics['base_hierarchical_weight'] = data['base_hierarchical_weight']

        except Exception as e:
            print(f"    讀取錯誤: {e}")

    # 獲取時間戳
    dir_name = exp_dir.name
    try:
        timestamp_str = dir_name.split('_')[0] + '_' + dir_name.split('_')[1]
        metrics['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except:
        metrics['timestamp'] = datetime.fromtimestamp(exp_dir.stat().st_mtime)

    return metrics


def generate_text_report(dataset, unified_metrics, baseline_metrics):
    """生成純文字報告"""
    report = []
    report.append("=" * 80)
    report.append(f"Unified-HIARN 實驗報告 - {dataset.upper()} Dataset")
    report.append("=" * 80)
    report.append(f"\n生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 模型架構說明
    report.append("-" * 80)
    report.append("模型架構: Unified-HIARN (Method 4)")
    report.append("-" * 80)
    report.append("  核心創新: 統一模型，根據**數據集**的多面向比例自動選擇策略")
    report.append("")
    report.append("  策略選擇 (數據集級別):")
    report.append("    - 多面向比例 > 50%: 使用 IARN 模式 (aspect 間交互)")
    report.append("    - 多面向比例 <= 50%: 使用 Hierarchical 模式 (純階層特徵)")
    report.append("")
    report.append("  Hierarchical 分支:")
    report.append("    - Low (layers 1-4): 詞彙特徵 → concat → fusion")
    report.append("    - Mid (layers 5-8): 語義特徵 → concat → fusion")
    report.append("    - High (layers 9-12): 任務特徵 → concat → fusion")
    report.append("    - 與獨立 HierarchicalBERT 架構完全一致")
    report.append("")
    report.append("  IARN 分支:")
    report.append("    - Aspect-to-Aspect Attention + Relation-aware Gating")
    report.append("")

    # 實驗配置
    report.append("-" * 80)
    report.append("實驗配置")
    report.append("-" * 80)
    # 使用 or 確保 None 值被替換為預設值
    epochs = unified_metrics.get('total_epochs') or 30
    patience = unified_metrics.get('patience') or 10
    focal_gamma = unified_metrics.get('focal_gamma') or 2.0
    class_weights = unified_metrics.get('class_weights')

    # 格式化 class_weights 顯示
    if class_weights is None or class_weights == 'auto':
        class_weights_str = "auto (動態計算)"
    elif isinstance(class_weights, list):
        class_weights_str = f"[{', '.join(f'{w:.1f}' for w in class_weights)}]"
    else:
        class_weights_str = str(class_weights)

    report.append(f"  Epochs:        {epochs}")
    report.append(f"  Patience:      {patience}")
    report.append("  Learning Rate: 2e-5")
    report.append("  BERT Model:    BERT-base-uncased")
    report.append("  Optimizer:     AdamW")
    report.append("  Scheduler:     Cosine Annealing with Warmup (10%)")
    report.append("  Loss Type:     Focal Loss")
    report.append(f"  Focal Gamma:   {focal_gamma}")
    report.append(f"  Class Weights: {class_weights_str}")
    report.append("  Dropout:       0.4")
    report.append("")

    # 主要結果
    report.append("-" * 80)
    report.append("實驗結果 (Main Results)")
    report.append("-" * 80)
    if unified_metrics['test_acc']:
        acc = unified_metrics['test_acc'] * 100
        f1 = unified_metrics['test_f1'] * 100
        best_ep = unified_metrics.get('best_epoch') or 'N/A'
        total_ep = epochs  # 使用上面已處理的 epochs 值
        report.append(f"  Accuracy:     {acc:.2f}%")
        report.append(f"  Macro-F1:     {f1:.2f}%")
        report.append(f"  Best Epoch:   {best_ep} / {total_ep}")
    else:
        report.append("  [ERROR] 未找到有效結果")
    report.append("")

    # Per-class F1
    if unified_metrics.get('test_f1_neg'):
        report.append("-" * 80)
        report.append("Per-class F1 Analysis")
        report.append("-" * 80)
        f1_neg = unified_metrics['test_f1_neg'] * 100
        f1_neu = unified_metrics['test_f1_neu'] * 100
        f1_pos = unified_metrics['test_f1_pos'] * 100
        report.append(f"  Negative F1:  {f1_neg:.2f}%")
        report.append(f"  Neutral F1:   {f1_neu:.2f}%")
        report.append(f"  Positive F1:  {f1_pos:.2f}%")
        report.append("")

    # 策略分析 (數據集級別)
    report.append("-" * 80)
    report.append("策略分析 (Dataset-level Strategy)")
    report.append("-" * 80)
    report.append("  Unified-HIARN 使用數據集級別策略選擇:")
    report.append("")
    report.append("  策略決定方式:")
    report.append("    1. 計算訓練集的多面向比例 (>=2 aspects 的樣本比例)")
    report.append("    2. 若比例 > 50%: 使用 IARN 模式 (aspect 間交互)")
    report.append("    3. 若比例 <= 50%: 使用 Hierarchical 模式 (純階層特徵)")
    report.append("")
    report.append("  各數據集預期策略:")
    report.append("    - Restaurants/Laptops: ~30-40% 多面向 → Hierarchical 模式")
    report.append("    - MAMS: 100% 多面向 → IARN 模式")
    report.append("")
    report.append("  優點:")
    report.append("    - 單一模型適應不同數據集特性")
    report.append("    - 低多面向數據集: 與獨立 Hierarchical 結果一致")
    report.append("    - 高多面向數據集: 獲得 aspect 間交互能力")
    report.append("")

    # 與 Baseline 對比
    report.append("-" * 80)
    report.append("與 Baseline 對比")
    report.append("-" * 80)

    report.append(f"{'Model':<30} {'Acc (%)':>10} {'Macro-F1 (%)':>14}")
    report.append("-" * 80)

    # Unified-HIARN
    if unified_metrics['test_acc']:
        report.append(f"{'>>> Unified-HIARN (Method 4)':<30} {unified_metrics['test_acc']*100:>10.2f} {unified_metrics['test_f1']*100:>14.2f}")

    # Baseline
    if baseline_metrics and baseline_metrics.get('test_acc'):
        report.append(f"{'    Baseline (BERT-CLS)':<30} {baseline_metrics['test_acc']*100:>10.2f} {baseline_metrics['test_f1']*100:>14.2f}")

    report.append("-" * 80)
    report.append("")

    # 改進分析
    if baseline_metrics and baseline_metrics.get('test_f1') and unified_metrics.get('test_f1'):
        report.append("-" * 80)
        report.append("改進分析 (vs Baseline)")
        report.append("-" * 80)

        # Accuracy 改進
        acc_improvement = (unified_metrics['test_acc'] - baseline_metrics['test_acc']) * 100
        report.append(f"  Accuracy 提升:  {acc_improvement:+.2f}%")

        # Macro-F1 改進
        f1_improvement = (unified_metrics['test_f1'] - baseline_metrics['test_f1']) * 100
        f1_improvement_pct = (f1_improvement / (baseline_metrics['test_f1'] * 100)) * 100
        report.append(f"  Macro-F1 提升:  {f1_improvement:+.2f}% ({f1_improvement_pct:+.2f}% relative)")

        # Per-class 改進
        if unified_metrics.get('test_f1_neg') and baseline_metrics.get('test_f1_neg'):
            neg_diff = (unified_metrics['test_f1_neg'] - baseline_metrics['test_f1_neg']) * 100
            neu_diff = (unified_metrics['test_f1_neu'] - baseline_metrics['test_f1_neu']) * 100
            pos_diff = (unified_metrics['test_f1_pos'] - baseline_metrics['test_f1_pos']) * 100
            report.append(f"  Negative F1:    {neg_diff:+.2f}%")
            report.append(f"  Neutral F1:     {neu_diff:+.2f}%")
            report.append(f"  Positive F1:    {pos_diff:+.2f}%")

        report.append("")

        # 總結
        if f1_improvement > 0:
            report.append(f"  >>> Unified-HIARN 優於 Baseline (+{f1_improvement:.2f}% Macro-F1)")
        elif f1_improvement < 0:
            report.append(f"  >>> Unified-HIARN 低於 Baseline ({f1_improvement:.2f}% Macro-F1)")
        else:
            report.append(f"  >>> Unified-HIARN 與 Baseline 持平")

        report.append("")
    elif not baseline_metrics or not baseline_metrics.get('test_f1'):
        report.append("  [INFO] 未找到 Baseline 結果，無法進行對比")
        report.append("  請先執行: python run_experiments.py --dataset <dataset>")
        report.append("")

    # 實驗目錄
    report.append("-" * 80)
    report.append("實驗目錄")
    report.append("-" * 80)
    report.append(f"  Unified-HIARN: {unified_metrics.get('exp_name', 'N/A')}")
    report.append("")

    # 結論
    report.append("-" * 80)
    report.append("結論")
    report.append("-" * 80)
    report.append("")
    report.append("Unified-HIARN 的設計目標:")
    report.append("  1. 統一模型: 無需針對不同數據集手動選擇模型")
    report.append("  2. 數據集自適應: 根據多面向比例自動選擇最佳策略")
    report.append("  3. 保持最優性能:")
    report.append("     - Restaurants/Laptops: 自動使用 Hierarchical，與獨立模型一致")
    report.append("     - MAMS: 自動使用 IARN，獲得 aspect 交互能力")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='生成 Unified-HIARN 獨立報告')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['restaurants', 'laptops', 'mams', 'rest16', 'lap16'],
                        help='數據集選擇')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    print(f"\n收集 {args.dataset.upper()} Unified-HIARN 實驗結果\n")

    # 查找 Unified-HIARN 實驗
    unified_exp = find_unified_hiarn_experiment(results_dir, args.dataset)
    if unified_exp is None:
        print(f"  [ERROR] 未找到 Unified-HIARN 實驗結果")
        print(f"  請先執行: python run_experiments.py --dataset {args.dataset} --unified")
        return

    print(f"  [OK] 找到: {unified_exp.name}")
    unified_metrics = read_metrics(unified_exp)
    if unified_metrics['test_acc']:
        print(f"       Test Acc: {unified_metrics['test_acc']:.4f}, F1: {unified_metrics['test_f1']:.4f}")

    # 查找 Baseline 實驗
    print("\n查找 Baseline 實驗...")
    baseline_exp = find_baseline_experiment(results_dir, args.dataset)
    baseline_metrics = None
    if baseline_exp:
        print(f"  [OK] Baseline: {baseline_exp.name}")
        baseline_metrics = read_metrics(baseline_exp)
    else:
        print(f"  [INFO] 未找到 Baseline 結果")

    # 生成報告
    text_report = generate_text_report(args.dataset, unified_metrics, baseline_metrics)

    # 保存報告
    report_path = results_dir / f"Unified-HIARN報告_{args.dataset}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text_report)

    print("\n" + "=" * 80)
    print(f"報告生成完成: {report_path}")
    print("=" * 80 + "\n")

    # 打印報告內容
    print(text_report)


if __name__ == "__main__":
    main()
