"""
批次執行實驗腳本

使用方法:
    # 全 HKGAN 模式: 對所有數據集執行 HKGAN
    python run_experiments.py --hkgan --full-run

    # 全 HKGAN 多種子模式: 對所有數據集執行多種子 HKGAN (種子: 42, 123, 2023, 999, 0)
    python run_experiments.py --hkgan --full-run --multi-seed

    # 全基線模式: 對所有數據集執行 Baseline
    python run_experiments.py --full-baseline

    # 只生成報告
    python run_experiments.py --report-only

    # 統計顯著性檢驗模式
    python run_experiments.py --significance-test
"""

import subprocess
import argparse
from pathlib import Path
import sys
import json
import numpy as np
from scipy import stats

# 多種子實驗用的種子列表
MULTI_SEED_LIST = [42, 123, 2023, 999, 0]

# 所有支援的數據集
ALL_DATASETS = ['restaurants', 'laptops', 'mams', 'rest16', 'lap16']

# 數據集顯示名稱
DISPLAY_NAMES = {
    'rest16': 'REST16',
    'restaurants': 'REST14',
    'laptops': 'LAP14',
    'lap16': 'LAP16',
    'mams': 'MAMS'
}


def get_display_name(dataset):
    """取得數據集的顯示名稱"""
    return DISPLAY_NAMES.get(dataset, dataset.upper())


def run_experiment(config_path, description, dataset, seed_override=None):
    """執行單個實驗

    Args:
        config_path: 配置文件路徑
        description: 實驗描述
        dataset: 數據集名稱
        seed_override: 覆蓋配置文件中的 seed（用於多種子實驗）
    """
    cmd = [
        sys.executable,
        "experiments/train_from_config.py",
        "--config", str(config_path),
        "--dataset", dataset
    ]

    # 如果指定了 seed_override，通過 --override 傳遞
    # 注意：--override 使用 nargs='*'，需要傳遞多個參數
    if seed_override is not None:
        cmd.extend(["--override", "--seed", str(seed_override)])

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def generate_hkgan_report(dataset):
    """生成 HKGAN 對比報告"""
    cmd = [
        sys.executable,
        "experiments/generate_hkgan_report.py",
        "--dataset", dataset
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ HKGAN Report generation FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def generate_thesis_figures():
    """生成論文圖表（包含 ROC 曲線）"""
    cmd = [
        sys.executable,
        "experiments/plot_thesis_figures.py",
        "--figure", "all",
        "--output", "results/figures/",
        "--no-show"
    ]

    try:
        print(f"\n{'='*60}")
        print("生成論文圖表（含 ROC 曲線）...")
        print(f"{'='*60}\n")
        subprocess.run(cmd, check=True)
        print(f"\n✓ 圖表已保存至 results/figures/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Figure generation FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def run_baseline_only(dataset, multi_seed=False):
    """執行基線實驗 (只有 Baseline)

    Args:
        dataset: 數據集名稱
        multi_seed: 是否執行多種子實驗
    """
    configs_dir = Path("configs")
    config_path = configs_dir / "unified_baseline.yaml"

    if not config_path.exists():
        print(f"  ERROR: unified_baseline.yaml not found")
        return False

    if multi_seed:
        return run_multi_seed_baseline(dataset, config_path)

    # 單種子模式
    print(f"\n{'='*60}")
    print(f"[Baseline Mode] Running on {get_display_name(dataset)}")
    print(f"{'='*60}\n")

    print(f"[1/1] Running Baseline: BERT-CLS...")
    success = run_experiment(config_path, "Baseline: BERT-CLS", dataset)

    if success:
        print(f"\n[Summary] 1/1 succeeded")

    return success


def run_multi_seed_baseline(dataset, config_path):
    """執行多種子 Baseline 實驗（用於統計顯著性檢驗）

    使用多個隨機種子執行 Baseline 實驗，
    以便與 HKGAN 進行 paired t-test 統計顯著性檢驗。
    """
    results_dir = Path("results/baseline") / dataset
    seeds = MULTI_SEED_LIST

    print(f"\n{'='*80}")
    print(f"[Multi-Seed Baseline Mode] Running on {get_display_name(dataset)}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}\n")

    results = []
    success_count = 0

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(seeds)}] Running Baseline with seed={seed}")
        print(f"{'='*60}\n")

        success = run_experiment(config_path, f"Baseline seed={seed}", dataset, seed_override=seed)

        if success:
            success_count += 1
            # 讀取結果
            exp_dirs = sorted(results_dir.glob("*_baseline_*"), key=lambda x: x.stat().st_mtime, reverse=True)
            if exp_dirs:
                rf = exp_dirs[0] / "reports" / "experiment_results.json"
                if rf.exists():
                    with open(rf, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    test_metrics = result.get('test_metrics', {})
                    results.append({
                        'seed': seed,
                        'accuracy': test_metrics.get('accuracy', 0),
                        'f1_macro': test_metrics.get('f1_macro', 0),
                        'f1_per_class': test_metrics.get('f1_per_class', [0, 0, 0]),
                        'auc_macro': test_metrics.get('auc_macro')
                    })

    print(f"\n{'='*80}")
    print(f"[Multi-Seed Baseline Summary] {success_count}/{len(seeds)} succeeded")
    print(f"{'='*80}\n")

    if results:
        generate_baseline_multiseed_report(dataset, results)

    return success_count == len(seeds)


def generate_baseline_multiseed_report(dataset, results):
    """生成 Baseline 多種子實驗報告"""
    reports_dir = Path("results")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 計算統計數據
    accuracies = [r['accuracy'] * 100 for r in results]
    f1_macros = [r['f1_macro'] * 100 for r in results]
    f1_neg = [r['f1_per_class'][0] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_neu = [r['f1_per_class'][1] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_pos = [r['f1_per_class'][2] * 100 for r in results if len(r['f1_per_class']) >= 3]
    auc_macros = [r['auc_macro'] * 100 for r in results if r.get('auc_macro') is not None]

    def mean_std(arr):
        if not arr:
            return 0, 0
        return float(np.mean(arr)), float(np.std(arr))

    acc_mean, acc_std = mean_std(accuracies)
    f1_mean, f1_std = mean_std(f1_macros)
    neg_mean, neg_std = mean_std(f1_neg)
    neu_mean, neu_std = mean_std(f1_neu)
    pos_mean, pos_std = mean_std(f1_pos)
    auc_mean, auc_std = mean_std(auc_macros)

    report = []
    report.append("=" * 80)
    report.append(f"Baseline 多種子實驗報告 - {get_display_name(dataset)} Dataset")
    report.append("=" * 80)
    report.append(f"Seeds: {[r['seed'] for r in results]}")
    report.append(f"Runs: {len(results)}")
    report.append("")

    report.append("-" * 80)
    report.append("Individual Results (Baseline)")
    report.append("-" * 80)
    report.append(f"{'Seed':<10} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'AUC (%)':<12} {'Neg F1 (%)':<12} {'Neu F1 (%)':<12} {'Pos F1 (%)':<12}")
    report.append("-" * 80)

    for r in results:
        f1_cls = r['f1_per_class']
        r_auc = r.get('auc_macro')
        r_auc_str = f"{r_auc*100:<12.2f}" if r_auc else "N/A         "
        report.append(f"{r['seed']:<10} {r['accuracy']*100:<12.2f} {r['f1_macro']*100:<15.2f} {r_auc_str} {f1_cls[0]*100:<12.2f} {f1_cls[1]*100:<12.2f} {f1_cls[2]*100:<12.2f}")

    report.append("-" * 80)
    report.append("")
    report.append("-" * 80)
    report.append("Aggregated Statistics (Mean ± Std)")
    report.append("-" * 80)
    report.append(f"  Accuracy:   {acc_mean:.2f}% ± {acc_std:.2f}%")
    report.append(f"  Macro-F1:   {f1_mean:.2f}% ± {f1_std:.2f}%")
    if auc_mean > 0:
        report.append(f"  AUC:        {auc_mean:.2f}% ± {auc_std:.2f}%")
    report.append(f"  Neg F1:     {neg_mean:.2f}% ± {neg_std:.2f}%")
    report.append(f"  Neu F1:     {neu_mean:.2f}% ± {neu_std:.2f}%")
    report.append(f"  Pos F1:     {pos_mean:.2f}% ± {pos_std:.2f}%")
    report.append("-" * 80)
    report.append("")
    report.append("=" * 80)

    # 保存報告
    report_text = "\n".join(report)
    output_file = reports_dir / f"Baseline_MultiSeed_{dataset}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\n報告已保存至: {output_file}")


def run_statistical_significance_test():
    """執行統計顯著性檢驗（Paired t-test）

    比較 HKGAN 與 Baseline 在相同 seed 下的配對結果，
    計算 paired t-test 以驗證改善是否具有統計顯著性。
    """
    results_dir = Path("results")
    improved_dir = Path("results/improved")
    baseline_dir = Path("results/baseline")

    print(f"\n{'='*90}")
    print("Statistical Significance Test (Paired t-test)")
    print(f"{'='*90}")
    print(f"Seeds: {MULTI_SEED_LIST}")
    print(f"Significance level: α = 0.05")
    print(f"{'='*90}\n")

    all_results = []

    for dataset in ALL_DATASETS:
        print(f"\n{'-'*70}")
        print(f"Dataset: {get_display_name(dataset)}")
        print(f"{'-'*70}")

        # 收集配對數據
        hkgan_f1_by_seed = {}
        baseline_f1_by_seed = {}
        hkgan_acc_by_seed = {}
        baseline_acc_by_seed = {}

        # 讀取 HKGAN 結果
        hkgan_dir = improved_dir / dataset
        if hkgan_dir.exists():
            for exp in hkgan_dir.glob("*_improved_hkgan_*"):
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    with open(rf, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    seed = result.get('args', {}).get('seed', 42)
                    test_metrics = result.get('test_metrics', {})
                    if seed in MULTI_SEED_LIST:
                        hkgan_f1_by_seed[seed] = test_metrics.get('f1_macro', 0) * 100
                        hkgan_acc_by_seed[seed] = test_metrics.get('accuracy', 0) * 100

        # 讀取 Baseline 結果
        bl_dir = baseline_dir / dataset
        if bl_dir.exists():
            for exp in bl_dir.glob("*_baseline_*"):
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    with open(rf, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    seed = result.get('args', {}).get('seed', 42)
                    test_metrics = result.get('test_metrics', {})
                    if seed in MULTI_SEED_LIST:
                        baseline_f1_by_seed[seed] = test_metrics.get('f1_macro', 0) * 100
                        baseline_acc_by_seed[seed] = test_metrics.get('accuracy', 0) * 100

        # 找出共同的 seeds
        common_seeds = sorted(set(hkgan_f1_by_seed.keys()) & set(baseline_f1_by_seed.keys()))

        if len(common_seeds) < 2:
            print(f"  警告: 配對數據不足 (需要至少 2 個共同 seed)")
            print(f"  HKGAN seeds: {sorted(hkgan_f1_by_seed.keys())}")
            print(f"  Baseline seeds: {sorted(baseline_f1_by_seed.keys())}")
            continue

        # 準備配對數據
        hkgan_f1 = [hkgan_f1_by_seed[s] for s in common_seeds]
        baseline_f1 = [baseline_f1_by_seed[s] for s in common_seeds]
        hkgan_acc = [hkgan_acc_by_seed[s] for s in common_seeds]
        baseline_acc = [baseline_acc_by_seed[s] for s in common_seeds]

        # Paired t-test for Macro-F1
        t_stat_f1, p_value_f1 = stats.ttest_rel(hkgan_f1, baseline_f1)
        # Paired t-test for Accuracy
        t_stat_acc, p_value_acc = stats.ttest_rel(hkgan_acc, baseline_acc)

        # Wilcoxon signed-rank test (非參數檢驗，作為補充)
        try:
            w_stat_f1, w_p_value_f1 = stats.wilcoxon(hkgan_f1, baseline_f1)
        except ValueError:
            w_stat_f1, w_p_value_f1 = None, None

        # 計算效果量 (Cohen's d)
        diff_f1 = np.array(hkgan_f1) - np.array(baseline_f1)
        cohens_d_f1 = np.mean(diff_f1) / np.std(diff_f1, ddof=1) if np.std(diff_f1, ddof=1) > 0 else 0

        # 打印結果
        print(f"\n  Paired Seeds: {common_seeds} (n={len(common_seeds)})")
        print(f"\n  {'Seed':<8} {'Baseline F1':<14} {'HKGAN F1':<14} {'Diff':<10}")
        print(f"  {'-'*46}")
        for s, bf, hf in zip(common_seeds, baseline_f1, hkgan_f1):
            diff = hf - bf
            print(f"  {s:<8} {bf:<14.2f} {hf:<14.2f} {diff:+.2f}")
        print(f"  {'-'*46}")
        print(f"  {'Mean':<8} {np.mean(baseline_f1):<14.2f} {np.mean(hkgan_f1):<14.2f} {np.mean(diff_f1):+.2f}")

        print(f"\n  Macro-F1 Statistical Tests:")
        print(f"    Paired t-test:  t = {t_stat_f1:.3f}, p = {p_value_f1:.4f}", end="")
        print(f"  {'✓ Significant' if p_value_f1 < 0.05 else '✗ Not Significant'} (α=0.05)")
        if w_p_value_f1 is not None:
            print(f"    Wilcoxon test:  W = {w_stat_f1:.1f}, p = {w_p_value_f1:.4f}", end="")
            print(f"  {'✓ Significant' if w_p_value_f1 < 0.05 else '✗ Not Significant'} (α=0.05)")
        print(f"    Cohen's d:      {cohens_d_f1:.3f}", end="")
        if abs(cohens_d_f1) >= 0.8:
            print(" (Large effect)")
        elif abs(cohens_d_f1) >= 0.5:
            print(" (Medium effect)")
        elif abs(cohens_d_f1) >= 0.2:
            print(" (Small effect)")
        else:
            print(" (Negligible effect)")

        print(f"\n  Accuracy Statistical Tests:")
        print(f"    Paired t-test:  t = {t_stat_acc:.3f}, p = {p_value_acc:.4f}", end="")
        print(f"  {'✓ Significant' if p_value_acc < 0.05 else '✗ Not Significant'} (α=0.05)")

        all_results.append({
            'dataset': dataset,
            'n_pairs': len(common_seeds),
            'baseline_f1_mean': np.mean(baseline_f1),
            'hkgan_f1_mean': np.mean(hkgan_f1),
            'f1_improvement': np.mean(diff_f1),
            't_stat_f1': t_stat_f1,
            'p_value_f1': p_value_f1,
            'cohens_d_f1': cohens_d_f1,
            'significant_f1': p_value_f1 < 0.05
        })

    # 生成總結報告
    if all_results:
        generate_significance_report(all_results)


def generate_significance_report(all_results):
    """生成統計顯著性檢驗報告"""
    reports_dir = Path("results")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = []
    report.append("=" * 100)
    report.append("Statistical Significance Test Report - HKGAN vs Baseline")
    report.append("=" * 100)
    report.append(f"Test Method: Paired t-test (two-tailed)")
    report.append(f"Seeds: {MULTI_SEED_LIST}")
    report.append(f"Significance Level: α = 0.05")
    report.append("")
    report.append("-" * 100)
    cohens_d_header = "Cohen's d"
    delta_f1_header = "Delta F1"
    report.append(f"{'Dataset':<15} {'n':<5} {'Baseline F1':<14} {'HKGAN F1':<14} {delta_f1_header:<10} {'t-stat':<10} {'p-value':<12} {cohens_d_header:<12} {'Significant':<12}")
    report.append("-" * 100)

    significant_count = 0
    for r in all_results:
        sig_str = "Yes ***" if r['p_value_f1'] < 0.001 else ("Yes **" if r['p_value_f1'] < 0.01 else ("Yes *" if r['p_value_f1'] < 0.05 else "No"))
        if r['significant_f1']:
            significant_count += 1
        delta_str = f"{r['f1_improvement']:+.2f}"
        report.append(f"{get_display_name(r['dataset']):<15} {r['n_pairs']:<5} {r['baseline_f1_mean']:<14.2f} {r['hkgan_f1_mean']:<14.2f} {delta_str:<10} {r['t_stat_f1']:<10.3f} {r['p_value_f1']:<12.4f} {r['cohens_d_f1']:<12.3f} {sig_str:<12}")

    report.append("-" * 100)
    report.append("")
    report.append("Legend:")
    report.append("  *** p < 0.001 (highly significant)")
    report.append("  **  p < 0.01  (very significant)")
    report.append("  *   p < 0.05  (significant)")
    report.append("")
    report.append("Cohen's d interpretation:")
    report.append("  |d| >= 0.8: Large effect")
    report.append("  |d| >= 0.5: Medium effect")
    report.append("  |d| >= 0.2: Small effect")
    report.append("  |d| <  0.2: Negligible effect")
    report.append("")
    report.append("-" * 100)
    report.append(f"Summary: {significant_count}/{len(all_results)} datasets show statistically significant improvement (p < 0.05)")
    report.append("=" * 100)

    # 保存報告
    report_text = "\n".join(report)
    output_file = reports_dir / "Statistical_Significance_Report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n\n{report_text}")
    print(f"\n報告已保存至: {output_file}")


def run_hkgan_experiments(dataset, multi_seed=False):
    """執行 HKGAN 實驗

    Args:
        dataset: 數據集名稱
        multi_seed: 是否執行多種子實驗（驗證魯棒性）
    """
    configs_dir = Path("configs")
    config_path = configs_dir / "unified_hkgan.yaml"

    if not config_path.exists():
        print(f"  ERROR: unified_hkgan.yaml not found")
        return False

    if multi_seed:
        return run_multi_seed_hkgan(dataset, config_path)

    # 單種子模式
    print(f"\n{'='*60}")
    print(f"[HKGAN Mode] Running on {get_display_name(dataset)}")
    print(f"{'='*60}\n")

    print(f"[1/1] Running HKGAN...")
    success = run_experiment(config_path, "HKGAN", dataset)

    if success:
        print(f"\n[Summary] 1/1 succeeded")
    generate_hkgan_report(dataset)

    return success


def run_multi_seed_hkgan(dataset, config_path):
    """執行多種子 HKGAN 實驗（驗證魯棒性）

    使用多個隨機種子執行實驗，計算平均值和標準差，
    以證明模型性能的穩定性和可重複性。
    """
    results_dir = Path("results/improved") / dataset
    seeds = MULTI_SEED_LIST

    print(f"\n{'='*80}")
    print(f"[Multi-Seed HKGAN Mode] Running on {get_display_name(dataset)}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}\n")

    results = []
    success_count = 0

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(seeds)}] Running HKGAN with seed={seed}")
        print(f"{'='*60}\n")

        success = run_experiment(config_path, f"HKGAN (seed={seed})", dataset, seed_override=seed)

        if success:
            success_count += 1
            # 讀取最新實驗結果
            result = get_latest_experiment_result(results_dir)
            if result:
                results.append({
                    'seed': seed,
                    'accuracy': result.get('accuracy', 0),
                    'f1_macro': result.get('f1_macro', 0),
                    'f1_per_class': result.get('f1_per_class', [0, 0, 0]),
                    'auc_macro': result.get('auc_macro'),  # 新增 AUC
                    'auc_weighted': result.get('auc_weighted')
                })
                auc_str = f", AUC={result.get('auc_macro', 0)*100:.2f}%" if result.get('auc_macro') else ""
                print(f"  ✓ seed={seed}: Acc={result.get('accuracy', 0)*100:.2f}%, F1={result.get('f1_macro', 0)*100:.2f}%{auc_str}")
            else:
                print(f"  ⚠ seed={seed}: Could not read results")
        else:
            print(f"  ✗ seed={seed}: FAILED")

    # 生成多種子報告
    if results:
        generate_multi_seed_report(dataset, results)

    print(f"\n{'='*60}")
    print(f"[Multi-Seed Summary] {success_count}/{len(seeds)} experiments succeeded")
    print(f"{'='*60}\n")

    return success_count == len(seeds)


def get_latest_experiment_result(results_dir):
    """獲取最新實驗的測試結果"""
    if not results_dir.exists():
        return None

    # 找到最新的實驗目錄
    exp_dirs = sorted(results_dir.glob("*_improved_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not exp_dirs:
        return None

    latest_dir = exp_dirs[0]
    result_file = latest_dir / "reports" / "experiment_results.json"

    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('test_metrics', {})
        except Exception:
            pass

    return None


def get_baseline_result(dataset):
    """獲取 Baseline 實驗的測試結果"""
    baseline_dir = Path("results/baseline") / dataset

    if not baseline_dir.exists():
        return None

    # 找到最新的 baseline 實驗目錄
    exp_dirs = sorted(baseline_dir.glob("*_baseline_*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not exp_dirs:
        return None

    latest_dir = exp_dirs[0]
    result_file = latest_dir / "reports" / "experiment_results.json"

    if result_file.exists():
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('test_metrics', {})
        except Exception:
            pass

    return None


def generate_multi_seed_report(dataset, results):
    """生成多種子實驗報告（含 Baseline 對比）"""
    reports_dir = Path("results")
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 獲取 Baseline 結果
    baseline = get_baseline_result(dataset)

    # 計算統計數據
    accuracies = [r['accuracy'] * 100 for r in results]
    f1_macros = [r['f1_macro'] * 100 for r in results]
    f1_neg = [r['f1_per_class'][0] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_neu = [r['f1_per_class'][1] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_pos = [r['f1_per_class'][2] * 100 for r in results if len(r['f1_per_class']) >= 3]
    # 新增 AUC 統計
    auc_macros = [r['auc_macro'] * 100 for r in results if r.get('auc_macro') is not None]

    def mean_std(arr):
        if not arr:
            return 0, 0
        return float(np.mean(arr)), float(np.std(arr))

    acc_mean, acc_std = mean_std(accuracies)
    f1_mean, f1_std = mean_std(f1_macros)
    neg_mean, neg_std = mean_std(f1_neg)
    neu_mean, neu_std = mean_std(f1_neu)
    pos_mean, pos_std = mean_std(f1_pos)
    auc_mean, auc_std = mean_std(auc_macros)  # 新增 AUC 統計

    report = []
    report.append("=" * 80)
    report.append(f"HKGAN 多種子實驗報告 - {get_display_name(dataset)} Dataset")
    report.append("=" * 80)
    report.append(f"Seeds: {[r['seed'] for r in results]}")
    report.append(f"Runs: {len(results)}")
    report.append("")

    # Baseline 對比區塊
    report.append("-" * 80)
    report.append("Baseline vs HKGAN (Mean)")
    report.append("-" * 80)
    report.append(f"{'Model':<20} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'AUC (%)':<12} {'Neg F1 (%)':<12} {'Neu F1 (%)':<12} {'Pos F1 (%)':<12}")
    report.append("-" * 80)

    if baseline:
        b_acc = baseline.get('accuracy', 0) * 100
        b_f1 = baseline.get('f1_macro', 0) * 100
        b_auc = baseline.get('auc_macro', 0) * 100 if baseline.get('auc_macro') else 0
        b_per_class = baseline.get('f1_per_class', [0, 0, 0])
        b_neg = b_per_class[0] * 100 if len(b_per_class) >= 3 else 0
        b_neu = b_per_class[1] * 100 if len(b_per_class) >= 3 else 0
        b_pos = b_per_class[2] * 100 if len(b_per_class) >= 3 else 0
        b_auc_str = f"{b_auc:<12.2f}" if b_auc > 0 else "N/A         "
        report.append(f"{'Baseline':<20} {b_acc:<12.2f} {b_f1:<15.2f} {b_auc_str} {b_neg:<12.2f} {b_neu:<12.2f} {b_pos:<12.2f}")
    else:
        report.append(f"{'Baseline':<20} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    auc_str = f"{auc_mean:<12.2f}" if auc_mean > 0 else "N/A         "
    report.append(f"{'HKGAN (Mean)':<20} {acc_mean:<12.2f} {f1_mean:<15.2f} {auc_str} {neg_mean:<12.2f} {neu_mean:<12.2f} {pos_mean:<12.2f}")
    report.append("-" * 80)

    # 改進分析
    if baseline:
        report.append("")
        report.append("改進分析 (HKGAN vs Baseline):")
        acc_diff = acc_mean - b_acc
        f1_diff = f1_mean - b_f1
        auc_diff = auc_mean - b_auc if auc_mean > 0 and b_auc > 0 else None
        neg_diff = neg_mean - b_neg
        neu_diff = neu_mean - b_neu
        pos_diff = pos_mean - b_pos

        def diff_str(val):
            return f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%"

        report.append(f"  Accuracy:  {diff_str(acc_diff)}")
        report.append(f"  Macro-F1:  {diff_str(f1_diff)}")
        if auc_diff is not None:
            report.append(f"  AUC:       {diff_str(auc_diff)}")
        report.append(f"  Neg F1:    {diff_str(neg_diff)}")
        report.append(f"  Neu F1:    {diff_str(neu_diff)}")
        report.append(f"  Pos F1:    {diff_str(pos_diff)}")

        if f1_diff > 0:
            report.append(f"\n  ✓ HKGAN 在 {get_display_name(dataset)} 上超越 Baseline {f1_diff:.2f}% (Macro-F1)")
        else:
            report.append(f"\n  ✗ HKGAN 在 {get_display_name(dataset)} 上未能超越 Baseline")

    report.append("")
    report.append("-" * 80)
    report.append("Individual Results (HKGAN)")
    report.append("-" * 80)
    report.append(f"{'Seed':<10} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'AUC (%)':<12} {'Neg F1 (%)':<12} {'Neu F1 (%)':<12} {'Pos F1 (%)':<12}")
    report.append("-" * 80)

    for r in results:
        f1_cls = r['f1_per_class']
        r_auc = r.get('auc_macro')
        r_auc_str = f"{r_auc*100:<12.2f}" if r_auc else "N/A         "
        report.append(f"{r['seed']:<10} {r['accuracy']*100:<12.2f} {r['f1_macro']*100:<15.2f} {r_auc_str} {f1_cls[0]*100:<12.2f} {f1_cls[1]*100:<12.2f} {f1_cls[2]*100:<12.2f}")

    report.append("-" * 80)
    report.append("")
    report.append("-" * 80)
    report.append("Aggregated Statistics (Mean ± Std)")
    report.append("-" * 80)

    report.append(f"  Accuracy:   {acc_mean:.2f}% ± {acc_std:.2f}%")
    report.append(f"  Macro-F1:   {f1_mean:.2f}% ± {f1_std:.2f}%")
    if auc_mean > 0:
        report.append(f"  AUC:        {auc_mean:.2f}% ± {auc_std:.2f}%")
    report.append(f"  Neg F1:     {neg_mean:.2f}% ± {neg_std:.2f}%")
    report.append(f"  Neu F1:     {neu_mean:.2f}% ± {neu_std:.2f}%")
    report.append(f"  Pos F1:     {pos_mean:.2f}% ± {pos_std:.2f}%")
    report.append("-" * 80)
    report.append("")

    # 評估穩定性
    report.append("-" * 80)
    report.append("Robustness Assessment")
    report.append("-" * 80)

    if f1_std < 1.0:
        stability = "Excellent (σ < 1%)"
    elif f1_std < 2.0:
        stability = "Good (1% ≤ σ < 2%)"
    elif f1_std < 3.0:
        stability = "Fair (2% ≤ σ < 3%)"
    else:
        stability = "Poor (σ ≥ 3%)"

    report.append(f"  Macro-F1 Stability: {stability}")
    if f1_mean > 0:
        report.append(f"  Coefficient of Variation: {f1_std/f1_mean*100:.2f}%")
    report.append("")
    report.append("=" * 80)

    # 保存報告
    report_text = "\n".join(report)
    output_file = reports_dir / f"HKGAN_MultiSeed_{dataset}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\n報告已保存至: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='批次執行實驗')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['restaurants', 'mams', 'laptops', 'rest16', 'lap16'],
                        help='數據集選擇 (restaurants, mams, laptops, rest16, lap16)')
    parser.add_argument('--hkgan', action='store_true',
                        help='HKGAN 模式: 執行 HKGAN 實驗')
    parser.add_argument('--baseline', action='store_true',
                        help='Baseline 模式: 執行 Baseline 實驗')
    parser.add_argument('--multi-seed', action='store_true',
                        help='多種子模式: 使用 5 個種子驗證魯棒性 (42, 123, 2023, 999, 0)')
    parser.add_argument('--full-run', action='store_true',
                        help='全數據集模式: 對所有數據集執行實驗')
    parser.add_argument('--full-baseline', action='store_true',
                        help='全基線模式: 對所有數據集只執行 Baseline')
    parser.add_argument('--report-only', action='store_true',
                        help='只生成報告，不執行實驗')
    parser.add_argument('--significance-test', action='store_true',
                        help='執行統計顯著性檢驗 (Paired t-test)')

    args = parser.parse_args()

    # 統計顯著性檢驗模式
    if args.significance_test:
        run_statistical_significance_test()
        return

    # 只生成報告模式
    if args.report_only:
        # 生成各數據集報告
        if args.dataset:
            generate_hkgan_report(args.dataset)
        else:
            for dataset in ALL_DATASETS:
                generate_hkgan_report(dataset)
        # 生成論文圖表（包含 ROC 曲線）
        generate_thesis_figures()
        return

    # 全基線模式
    if args.full_baseline:
        print(f"\n{'='*80}")
        mode_str = "Multi-Seed Baseline" if args.multi_seed else "Baseline"
        print(f"[Full {mode_str} Mode] Running on {len(ALL_DATASETS)} datasets")
        print(f"Datasets: {', '.join(ALL_DATASETS)}")
        if args.multi_seed:
            print(f"Seeds: {MULTI_SEED_LIST}")
        print(f"{'='*80}\n")

        results = {}
        for dataset in ALL_DATASETS:
            print(f"\n{'#'*80}")
            print(f"# Dataset: {get_display_name(dataset)}")
            print(f"{'#'*80}")

            success = run_baseline_only(dataset, multi_seed=args.multi_seed)
            results[dataset] = 'Success' if success else 'Failed'

        # 總結報告
        print(f"\n{'='*80}")
        print(f"[Full {mode_str} Summary] All datasets completed")
        print(f"{'='*80}")
        for dataset, status in results.items():
            print(f"  {get_display_name(dataset):12s}: {mode_str} {status}")
        print(f"{'='*80}\n")
        return

    # 全 HKGAN 模式
    if args.full_run and args.hkgan:
        print(f"\n{'='*80}")
        mode_str = "Multi-Seed HKGAN" if args.multi_seed else "HKGAN"
        print(f"[Full {mode_str} Mode] Running on {len(ALL_DATASETS)} datasets")
        print(f"Datasets: {', '.join(ALL_DATASETS)}")
        if args.multi_seed:
            print(f"Seeds: {MULTI_SEED_LIST}")
        print(f"{'='*80}\n")

        results = {}
        for dataset in ALL_DATASETS:
            print(f"\n{'#'*80}")
            print(f"# Dataset: {get_display_name(dataset)}")
            print(f"{'#'*80}")

            success = run_hkgan_experiments(dataset, multi_seed=args.multi_seed)
            results[dataset] = 'Success' if success else 'Failed'

        # 總結報告
        print(f"\n{'='*80}")
        print(f"[Full {mode_str} Summary] All datasets completed")
        print(f"{'='*80}")
        for dataset, status in results.items():
            print(f"  {get_display_name(dataset):12s}: {mode_str} {status}")
        print(f"{'='*80}\n")
        return

    # 檢查 dataset 參數
    if args.dataset is None:
        parser.error("--dataset is required unless using --full-run, --full-baseline, or --report-only")

    # HKGAN 模式
    if args.hkgan:
        run_hkgan_experiments(args.dataset, multi_seed=args.multi_seed)
        return

    # Baseline 模式
    if args.baseline:
        run_baseline_only(args.dataset, multi_seed=args.multi_seed)
        return

    # 預設: HKGAN 模式
    print("提示: 未指定模式，預設使用 HKGAN 模式。使用 --baseline 執行基線實驗。")
    run_hkgan_experiments(args.dataset, multi_seed=args.multi_seed)


if __name__ == "__main__":
    main()
