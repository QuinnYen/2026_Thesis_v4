"""
批次執行實驗腳本

使用方法:
    python run_experiments.py --all                    # 執行所有實驗
    python run_experiments.py --baselines              # 只執行 baseline（含報告生成）
    python run_experiments.py --full                   # 只執行完整模型
    python run_experiments.py --ablation               # 執行消融實驗
    python run_experiments.py --report                 # 只生成 baseline 報告（不執行訓練）
"""

import subprocess
import argparse
from pathlib import Path
import sys

def run_experiment(config_path, description):
    """執行單個實驗"""
    print(f"\n{'='*80}")
    print(f"開始實驗: {description}")
    print(f"配置文件: {config_path}")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "experiments/train_from_config.py",
        "--config", str(config_path)
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ {description} 完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} 失敗 (錯誤碼: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\n中斷執行")
        return False


def generate_baseline_report():
    """生成 baseline 比較報告"""
    print(f"\n{'='*80}")
    print("生成 Baseline 比較報告")
    print(f"{'='*80}\n")

    cmd = [
        sys.executable,
        "experiments/generate_baseline_report.py"
    ]

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✓ 報告生成完成\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ 報告生成失敗 (錯誤碼: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\n中斷執行")
        return False


def main():
    parser = argparse.ArgumentParser(description='批次執行實驗')
    parser.add_argument('--all', action='store_true', help='執行所有實驗')
    parser.add_argument('--baselines', action='store_true', help='執行 baseline 實驗（含報告生成）')
    parser.add_argument('--full', action='store_true', help='執行完整模型')
    parser.add_argument('--ablation', action='store_true', help='執行消融實驗')
    parser.add_argument('--report', action='store_true', help='只生成 baseline 報告（不執行訓練）')
    parser.add_argument('--config', type=str, help='單獨執行指定配置文件')

    args = parser.parse_args()

    # 只生成報告模式
    if args.report:
        generate_baseline_report()
        return

    configs_dir = Path("configs")
    success_count = 0
    total_count = 0

    experiments = []

    # 定義實驗列表
    if args.all or args.baselines:
        experiments.extend([
            (configs_dir / "baseline_bert_only.yaml", "Baseline: BERT Only"),
            (configs_dir / "baseline_bert_aaha.yaml", "Baseline: BERT + AAHA"),
            (configs_dir / "baseline_bert_mean.yaml", "Baseline: BERT + Mean Pooling"),
        ])

    if args.all or args.ablation:
        experiments.extend([
            (configs_dir / "pmac_only.yaml", "Ablation: PMAC Only (without IARM)"),
            # (configs_dir / "iarm_only.yaml", "Ablation: IARM Only (without PMAC)"),  # 需要創建
        ])

    if args.all or args.full:
        experiments.append(
            (configs_dir / "full_model_optimized.yaml", "Full Model: PMAC + IARM (Optimized)")
        )

    if args.config:
        experiments = [(Path(args.config), f"Custom: {args.config}")]

    if not experiments:
        print("請指定要執行的實驗:")
        print("  --all         執行所有實驗")
        print("  --baselines   執行 baseline 實驗（含報告生成）")
        print("  --full        執行完整模型")
        print("  --ablation    執行消融實驗")
        print("  --report      只生成 baseline 報告")
        print("  --config PATH 執行指定配置")
        return

    print(f"\n{'='*80}")
    print(f"準備執行 {len(experiments)} 個實驗")
    print(f"{'='*80}\n")

    # 記錄是否為 baseline 模式
    is_baseline_mode = args.baselines or args.all

    for config_path, description in experiments:
        if not config_path.exists():
            print(f"⚠️  跳過: {description} (配置文件不存在: {config_path})")
            continue

        total_count += 1
        if run_experiment(config_path, description):
            success_count += 1

    # 總結
    print(f"\n{'='*80}")
    print("實驗總結")
    print(f"{'='*80}")
    print(f"總計: {total_count} 個實驗")
    print(f"成功: {success_count} 個")
    print(f"失敗: {total_count - success_count} 個")
    print(f"{'='*80}\n")

    # 如果執行了 baseline 實驗，自動生成報告
    if is_baseline_mode and success_count > 0:
        print("\n開始生成 Baseline 比較報告...\n")
        generate_baseline_report()


if __name__ == "__main__":
    main()
