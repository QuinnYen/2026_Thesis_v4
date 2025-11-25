"""
批次執行實驗腳本

支持的數據集:
    - mams: MAMS (100% multi-aspect)
    - restaurants: SemEval-2014 Restaurants
    - laptops: SemEval-2014 Laptops
    - rest16: SemEval-2016 Restaurants (Subtask 1, explicit targets)
    - lap16: SemEval-2016 Laptops (Slot 1, category-based implicit aspects)

使用方法:
    # 標準模式: 執行所有基礎方法 (Baseline + Method 1-3)
    python run_experiments.py --dataset mams
    python run_experiments.py --dataset restaurants
    python run_experiments.py --dataset laptops
    python run_experiments.py --dataset rest16
    python run_experiments.py --dataset lap16

    # Unified-HIARN 模式: 執行統一模型 (Method 4)
    python run_experiments.py --dataset mams --unified
    python run_experiments.py --dataset restaurants --unified
    python run_experiments.py --dataset laptops --unified
    python run_experiments.py --dataset rest16 --unified
    python run_experiments.py --dataset lap16 --unified

    # 完整模式: 執行所有方法 (標準 + Unified-HIARN)
    python run_experiments.py --dataset mams --all
    python run_experiments.py --dataset restaurants --all
    python run_experiments.py --dataset laptops --all
    python run_experiments.py --dataset rest16 --all
    python run_experiments.py --dataset lap16 --all

    # 全數據集完整模式: 一條命令執行所有數據集的完整實驗
    python run_experiments.py --full-run

執行順序 (標準模式):
    1. Baseline: BERT-CLS
    2. Method 1: Hierarchical BERT (階層式BERT)
    3. Method 2: IARN (Aspect 間交互)
    4. Method 3: HSA - Hierarchical Syntax Attention (階層式語法注意力)
    5. 生成綜合報告

Unified-HIARN 模式:
    1. Method 4: Unified-HIARN (動態融合 Hierarchical + IARN)
    2. 生成 Unified-HIARN 獨立報告

模型設計理念:
    - Method 1-3: 各有專長，需根據數據集選擇
    - Method 4 (Unified-HIARN): 統一模型，自適應處理單/多 aspect 場景

數據集說明:
    - SemEval-2014: 標準 ABSA 格式 (aspectTerm)
    - SemEval-2016 Rest16: 有明確 target 屬性
    - SemEval-2016 Lap16: 只有 category 屬性 (LAPTOP#GENERAL 格式)
"""

import subprocess
import argparse
from pathlib import Path
import sys


def run_experiment(config_path, description, dataset):
    """執行單個實驗"""
    cmd = [
        sys.executable,
        "experiments/train_from_config.py",
        "--config", str(config_path),
        "--dataset", dataset
    ]

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def generate_comprehensive_report(dataset):
    """生成綜合報告 (Method 1-3)"""
    cmd = [
        sys.executable,
        "experiments/generate_comprehensive_report.py",
        "--dataset", dataset
    ]

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Report generation FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def generate_unified_hiarn_report(dataset):
    """生成 Unified-HIARN 獨立報告 (Method 4)"""
    cmd = [
        sys.executable,
        "experiments/generate_unified_hiarn_report.py",
        "--dataset", dataset
    ]

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Unified-HIARN report generation FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def run_standard_experiments(dataset):
    """執行標準模式實驗 (Method 1-3)"""
    configs_dir = Path("configs")
    success_count = 0
    total_count = 0

    experiments = [
        (configs_dir / "unified_baseline.yaml", "Baseline: BERT-CLS"),
        (configs_dir / "unified_hierarchical.yaml", "Method 1: Hierarchical BERT"),
        (configs_dir / "unified_iarn.yaml", "Method 2: IARN"),
        (configs_dir / "unified_hsa.yaml", "Method 3: HSA"),
    ]

    print(f"\n{'='*60}")
    print(f"[Standard Mode] {len(experiments)} experiments on {dataset.upper()}")
    print(f"{'='*60}\n")

    for config_path, description in experiments:
        if not config_path.exists():
            print(f"  SKIP: {description} (config not found)")
            continue

        total_count += 1
        print(f"\n[{total_count}/{len(experiments)}] Running {description}...")
        if run_experiment(config_path, description, dataset):
            success_count += 1

    print(f"\n[Summary] {success_count}/{total_count} succeeded")
    generate_comprehensive_report(dataset)

    return success_count, total_count


def run_unified_hiarn_experiment(dataset):
    """執行 Unified-HIARN 實驗 (Method 4)"""
    configs_dir = Path("configs")
    config_path = configs_dir / "unified_hiarn.yaml"

    print(f"\n{'='*60}")
    print(f"[Unified-HIARN Mode] Running on {dataset.upper()}")
    print(f"{'='*60}\n")

    if not config_path.exists():
        print(f"  ERROR: unified_hiarn.yaml not found")
        return False

    print(f"[1/1] Running Method 4: Unified-HIARN...")
    success = run_experiment(config_path, "Method 4: Unified-HIARN", dataset)

    if success:
        print(f"\n[Summary] 1/1 succeeded")
        generate_unified_hiarn_report(dataset)

    return success


def main():
    parser = argparse.ArgumentParser(description='批次執行實驗')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['restaurants', 'mams', 'laptops', 'rest16', 'lap16'],
                        help='數據集選擇 (restaurants, mams, laptops, rest16, lap16)')
    parser.add_argument('--unified', action='store_true',
                        help='Unified-HIARN 模式: 只執行 Method 4 (統一模型)')
    parser.add_argument('--all', action='store_true',
                        help='完整模式: 執行所有方法 (標準 + Unified-HIARN)')
    parser.add_argument('--full-run', action='store_true',
                        help='全數據集完整模式: 對所有數據集執行完整實驗')

    args = parser.parse_args()

    # 全數據集完整模式
    if args.full_run:
        all_datasets = ['restaurants', 'laptops', 'mams']
        total_results = {}

        print(f"\n{'='*80}")
        print(f"[Full Run Mode] Running all experiments on {len(all_datasets)} datasets")
        print(f"Datasets: {', '.join(all_datasets)}")
        print(f"{'='*80}\n")

        for dataset in all_datasets:
            print(f"\n{'#'*80}")
            print(f"# Dataset: {dataset.upper()}")
            print(f"{'#'*80}")

            # 執行標準模式
            standard_success, standard_total = run_standard_experiments(dataset)

            # 執行 Unified-HIARN
            unified_success = run_unified_hiarn_experiment(dataset)

            total_results[dataset] = {
                'standard': (standard_success, standard_total),
                'unified': unified_success
            }

        # 總結報告
        print(f"\n{'='*80}")
        print(f"[Full Run Summary] All datasets completed")
        print(f"{'='*80}")
        for dataset, results in total_results.items():
            std_success, std_total = results['standard']
            unified = 'Success' if results['unified'] else 'Failed'
            print(f"  {dataset.upper():12s}: Standard {std_success}/{std_total}, Unified-HIARN {unified}")
        print(f"{'='*80}\n")
        return

    # 檢查 dataset 參數
    if args.dataset is None:
        parser.error("--dataset is required unless using --full-run")

    # Unified-HIARN 模式
    if args.unified:
        run_unified_hiarn_experiment(args.dataset)
        return

    # 完整模式
    if args.all:
        # 先執行標準模式
        standard_success, standard_total = run_standard_experiments(args.dataset)

        # 再執行 Unified-HIARN
        unified_success = run_unified_hiarn_experiment(args.dataset)

        print(f"\n{'='*60}")
        print(f"[Final Summary] All experiments completed")
        print(f"  Standard (Method 1-3): {standard_success}/{standard_total}")
        print(f"  Unified-HIARN (Method 4): {'Success' if unified_success else 'Failed'}")
        print(f"{'='*60}\n")
        return

    # 標準模式（預設）
    run_standard_experiments(args.dataset)


if __name__ == "__main__":
    main()
