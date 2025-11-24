"""
批次執行實驗腳本

使用方法:
    # 標準模式: 執行所有方法 (Baseline + Hierarchical + IARN + HSA)
    python run_experiments.py --dataset mams
    python run_experiments.py --dataset restaurants
    python run_experiments.py --dataset laptops

    # 自動模式: 根據數據集特徵自動選擇最佳模型
    python run_experiments.py --dataset mams --auto
    python run_experiments.py --dataset restaurants --auto
    python run_experiments.py --dataset laptops --auto

執行順序 (標準模式):
    1. Baseline: BERT-CLS
    2. Method 1: Hierarchical BERT (BERT 層級特徵)
    3. Method 2: IARN (Aspect 間交互)
    4. Method 3: HSA - Hierarchical Syntax Attention (階層式語法注意力)
    5. 生成綜合報告

自動模式選擇邏輯:
    - 多面向比例 > 50% → IARN
    - 多面向比例 ≤ 50% → Hierarchical BERT
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
    """生成綜合報告"""
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


def run_auto_experiment(dataset):
    """執行自動選擇模式實驗"""
    configs_dir = Path("configs")
    config_path = configs_dir / "adaptive_dual.yaml"

    if not config_path.exists():
        print(f"  ERROR: adaptive_dual.yaml not found")
        return False

    cmd = [
        sys.executable,
        "experiments/train_from_config.py",
        "--config", str(config_path),
        "--dataset", dataset
    ]

    print(f"\n[Auto] Analyzing {dataset.upper()} and selecting best model...\n")

    try:
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n  Auto experiment FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def main():
    parser = argparse.ArgumentParser(description='批次執行所有實驗')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['restaurants', 'mams', 'laptops'],
                        help='數據集選擇 (restaurants, mams, 或 laptops)')
    parser.add_argument('--auto', action='store_true',
                        help='自動選擇模式: 根據數據集特徵選擇最佳模型')

    args = parser.parse_args()

    configs_dir = Path("configs")

    # 自動選擇模式
    if args.auto:
        success = run_auto_experiment(args.dataset)
        if success:
            print(f"\n[Auto] Experiment completed successfully")
        return

    # 標準模式: 執行所有方法
    success_count = 0
    total_count = 0

    # 使用統一配置文件（所有數據集共用）
    experiments = [
        (configs_dir / "unified_baseline.yaml", "Baseline: BERT-CLS"),
        (configs_dir / "unified_hierarchical.yaml", "Method 1: Hierarchical BERT"),
        (configs_dir / "unified_iarn.yaml", "Method 2: IARN"),
        (configs_dir / "unified_hsa.yaml", "Method 3: HSA"),
    ]

    print(f"\n[Batch] {len(experiments)} experiments on {args.dataset.upper()}\n")

    for config_path, description in experiments:
        if not config_path.exists():
            print(f"  SKIP: {description} (config not found)")
            continue

        total_count += 1
        if run_experiment(config_path, description, args.dataset):
            success_count += 1

    # 總結並生成報告
    print(f"\n[Summary] {success_count}/{total_count} succeeded")
    generate_comprehensive_report(args.dataset)


if __name__ == "__main__":
    main()
