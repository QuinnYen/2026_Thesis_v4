"""
批次執行實驗腳本

支持的數據集:
    - restaurants: SemEval-2014 Restaurants
    - laptops: SemEval-2014 Laptops
    - mams: MAMS (100% multi-aspect)
    - rest16: SemEval-2016 Restaurants
    - lap16: SemEval-2016 Laptops

使用方法:
    # HKGAN 模式 (推薦): 執行 HKGAN 實驗
    python run_experiments.py --hkgan --dataset restaurants
    python run_experiments.py --hkgan --dataset laptops
    python run_experiments.py --hkgan --dataset mams
    python run_experiments.py --hkgan --dataset rest16
    python run_experiments.py --hkgan --dataset lap16

    # 多種子實驗模式 (驗證魯棒性):
    python run_experiments.py --hkgan --dataset laptops --multi-seed
    python run_experiments.py --hkgan --dataset restaurants --multi-seed
    python run_experiments.py --hkgan --dataset mams --multi-seed
    python run_experiments.py --hkgan --dataset rest16 --multi-seed
    python run_experiments.py --hkgan --dataset lap16 --multi-seed
    # 使用種子: 42, 123, 2023, 999, 0

    # 全 HKGAN 模式: 對所有數據集執行 HKGAN
    python run_experiments.py --hkgan --full-run

    # 全 HKGAN 多種子模式: 對所有數據集執行多種子 HKGAN
    python run_experiments.py --hkgan --full-run --multi-seed

    # Baseline 模式: 執行 Baseline 實驗
    python run_experiments.py --baseline --dataset restaurants
    python run_experiments.py --baseline --dataset laptops

    # 全基線模式: 對所有數據集執行 Baseline
    python run_experiments.py --full-baseline

    # 只生成報告
    python run_experiments.py --report-only --dataset restaurants

數據集說明:
    - SemEval-2014: 標準 ABSA 格式 (aspectTerm)
    - SemEval-2016 Rest16: 有明確目標屬性
    - SemEval-2016 Lap16: 只有 category 屬性 (LAPTOP#GENERAL 格式)
    - MAMS: 100% 多面向句子，每個面向可能有不同情感
"""

import subprocess
import argparse
from pathlib import Path
import sys
import json
import numpy as np

# 多種子實驗用的種子列表
MULTI_SEED_LIST = [42, 123, 2023, 999, 0]

# 所有支援的數據集
ALL_DATASETS = ['restaurants', 'laptops', 'mams', 'rest16', 'lap16']


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


def run_baseline_only(dataset):
    """執行基線實驗 (只有 Baseline)"""
    configs_dir = Path("configs")
    config_path = configs_dir / "unified_baseline.yaml"

    print(f"\n{'='*60}")
    print(f"[Baseline Mode] Running on {dataset.upper()}")
    print(f"{'='*60}\n")

    if not config_path.exists():
        print(f"  ERROR: unified_baseline.yaml not found")
        return False

    print(f"[1/1] Running Baseline: BERT-CLS...")
    success = run_experiment(config_path, "Baseline: BERT-CLS", dataset)

    if success:
        print(f"\n[Summary] 1/1 succeeded")

    return success


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
    print(f"[HKGAN Mode] Running on {dataset.upper()}")
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
    print(f"[Multi-Seed HKGAN Mode] Running on {dataset.upper()}")
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
                    'f1_per_class': result.get('f1_per_class', [0, 0, 0])
                })
                print(f"  ✓ seed={seed}: Acc={result.get('accuracy', 0)*100:.2f}%, F1={result.get('f1_macro', 0)*100:.2f}%")
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

    def mean_std(arr):
        if not arr:
            return 0, 0
        return float(np.mean(arr)), float(np.std(arr))

    acc_mean, acc_std = mean_std(accuracies)
    f1_mean, f1_std = mean_std(f1_macros)
    neg_mean, neg_std = mean_std(f1_neg)
    neu_mean, neu_std = mean_std(f1_neu)
    pos_mean, pos_std = mean_std(f1_pos)

    report = []
    report.append("=" * 80)
    report.append(f"HKGAN 多種子實驗報告 - {dataset.upper()} Dataset")
    report.append("=" * 80)
    report.append(f"Seeds: {[r['seed'] for r in results]}")
    report.append(f"Runs: {len(results)}")
    report.append("")

    # Baseline 對比區塊
    report.append("-" * 80)
    report.append("Baseline vs HKGAN (Mean)")
    report.append("-" * 80)
    report.append(f"{'Model':<20} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'Neg F1 (%)':<12} {'Neu F1 (%)':<12} {'Pos F1 (%)':<12}")
    report.append("-" * 80)

    if baseline:
        b_acc = baseline.get('accuracy', 0) * 100
        b_f1 = baseline.get('f1_macro', 0) * 100
        b_per_class = baseline.get('f1_per_class', [0, 0, 0])
        b_neg = b_per_class[0] * 100 if len(b_per_class) >= 3 else 0
        b_neu = b_per_class[1] * 100 if len(b_per_class) >= 3 else 0
        b_pos = b_per_class[2] * 100 if len(b_per_class) >= 3 else 0
        report.append(f"{'Baseline':<20} {b_acc:<12.2f} {b_f1:<15.2f} {b_neg:<12.2f} {b_neu:<12.2f} {b_pos:<12.2f}")
    else:
        report.append(f"{'Baseline':<20} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    report.append(f"{'HKGAN (Mean)':<20} {acc_mean:<12.2f} {f1_mean:<15.2f} {neg_mean:<12.2f} {neu_mean:<12.2f} {pos_mean:<12.2f}")
    report.append("-" * 80)

    # 改進分析
    if baseline:
        report.append("")
        report.append("改進分析 (HKGAN vs Baseline):")
        acc_diff = acc_mean - b_acc
        f1_diff = f1_mean - b_f1
        neg_diff = neg_mean - b_neg
        neu_diff = neu_mean - b_neu
        pos_diff = pos_mean - b_pos

        def diff_str(val):
            return f"+{val:.2f}%" if val >= 0 else f"{val:.2f}%"

        report.append(f"  Accuracy:  {diff_str(acc_diff)}")
        report.append(f"  Macro-F1:  {diff_str(f1_diff)}")
        report.append(f"  Neg F1:    {diff_str(neg_diff)}")
        report.append(f"  Neu F1:    {diff_str(neu_diff)}")
        report.append(f"  Pos F1:    {diff_str(pos_diff)}")

        if f1_diff > 0:
            report.append(f"\n  ✓ HKGAN 在 {dataset.upper()} 上超越 Baseline {f1_diff:.2f}% (Macro-F1)")
        else:
            report.append(f"\n  ✗ HKGAN 在 {dataset.upper()} 上未能超越 Baseline")

    report.append("")
    report.append("-" * 80)
    report.append("Individual Results (HKGAN)")
    report.append("-" * 80)
    report.append(f"{'Seed':<10} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'Neg F1 (%)':<12} {'Neu F1 (%)':<12} {'Pos F1 (%)':<12}")
    report.append("-" * 80)

    for r in results:
        f1_cls = r['f1_per_class']
        report.append(f"{r['seed']:<10} {r['accuracy']*100:<12.2f} {r['f1_macro']*100:<15.2f} {f1_cls[0]*100:<12.2f} {f1_cls[1]*100:<12.2f} {f1_cls[2]*100:<12.2f}")

    report.append("-" * 80)
    report.append("")
    report.append("-" * 80)
    report.append("Aggregated Statistics (Mean ± Std)")
    report.append("-" * 80)

    report.append(f"  Accuracy:   {acc_mean:.2f}% ± {acc_std:.2f}%")
    report.append(f"  Macro-F1:   {f1_mean:.2f}% ± {f1_std:.2f}%")
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

    args = parser.parse_args()

    # 只生成報告模式
    if args.report_only:
        if args.dataset:
            generate_hkgan_report(args.dataset)
        else:
            for dataset in ALL_DATASETS:
                generate_hkgan_report(dataset)
        return

    # 全基線模式
    if args.full_baseline:
        print(f"\n{'='*80}")
        print(f"[Full Baseline Mode] Running Baseline on {len(ALL_DATASETS)} datasets")
        print(f"Datasets: {', '.join(ALL_DATASETS)}")
        print(f"{'='*80}\n")

        results = {}
        for dataset in ALL_DATASETS:
            print(f"\n{'#'*80}")
            print(f"# Dataset: {dataset.upper()}")
            print(f"{'#'*80}")

            success = run_baseline_only(dataset)
            results[dataset] = 'Success' if success else 'Failed'

        # 總結報告
        print(f"\n{'='*80}")
        print(f"[Full Baseline Summary] All datasets completed")
        print(f"{'='*80}")
        for dataset, status in results.items():
            print(f"  {dataset.upper():12s}: Baseline {status}")
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
            print(f"# Dataset: {dataset.upper()}")
            print(f"{'#'*80}")

            success = run_hkgan_experiments(dataset, multi_seed=args.multi_seed)
            results[dataset] = 'Success' if success else 'Failed'

        # 總結報告
        print(f"\n{'='*80}")
        print(f"[Full {mode_str} Summary] All datasets completed")
        print(f"{'='*80}")
        for dataset, status in results.items():
            print(f"  {dataset.upper():12s}: {mode_str} {status}")
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
        run_baseline_only(args.dataset)
        return

    # 預設: HKGAN 模式
    print("提示: 未指定模式，預設使用 HKGAN 模式。使用 --baseline 執行基線實驗。")
    run_hkgan_experiments(args.dataset, multi_seed=args.multi_seed)


if __name__ == "__main__":
    main()
