"""
消融實驗執行腳本

支援的消融變體:
    - full: 完整 HKGAN 模型（基準）
    - no_senticnet: 移除 SenticNet 知識增強
    - no_dynamic_gate: 移除動態知識門控 (v3.0)
    - no_confidence_gate: 移除信心門控 (v2.0)
    - no_contrastive: 移除對比學習
    - no_logit_adjust: 移除非對稱 Logit 調整
    - no_focal_loss: 改用 CrossEntropy
    - no_llrd: 移除逐層學習率衰減

使用方法:
    # 執行單個消融實驗
    python run_ablation.py --ablation no_senticnet --dataset laptops

    # 執行多種子消融實驗（驗證穩定性）
    python run_ablation.py --ablation no_senticnet --dataset laptops --multi-seed

    # 執行所有消融實驗（單一資料集）
    python run_ablation.py --all --dataset laptops

    # 執行完整消融研究（所有變體 × 所有資料集）
    python run_ablation.py --full-study

    # 只生成消融報告
    python run_ablation.py --report-only --dataset laptops

    # 列出所有可用的消融變體
    python run_ablation.py --list
"""

import subprocess
import argparse
from pathlib import Path
import sys
import json
import numpy as np
from datetime import datetime

# 消融變體對應的配置檔案
ABLATION_CONFIGS = {
    'full': 'configs/ablation/base_hkgan.yaml',
    'no_senticnet': 'configs/ablation/no_senticnet.yaml',
    'no_dynamic_gate': 'configs/ablation/no_dynamic_gate.yaml',
    'no_confidence_gate': 'configs/ablation/no_confidence_gate.yaml',
    'no_contrastive': 'configs/ablation/no_contrastive.yaml',
    'no_logit_adjust': 'configs/ablation/no_logit_adjust.yaml',
    'no_focal_loss': 'configs/ablation/no_focal_loss.yaml',
    'no_llrd': 'configs/ablation/no_llrd.yaml',
}

# 消融變體的中文描述
ABLATION_DESCRIPTIONS = {
    'full': '完整 HKGAN 模型（基準）',
    'no_senticnet': '移除 SenticNet 知識增強',
    'no_dynamic_gate': '移除動態知識門控 (v3.0)',
    'no_confidence_gate': '移除信心門控 (v2.0)',
    'no_contrastive': '移除對比學習',
    'no_logit_adjust': '移除非對稱 Logit 調整',
    'no_focal_loss': '改用 CrossEntropy Loss',
    'no_llrd': '移除逐層學習率衰減 (LLRD)',
}

# 多種子實驗用的種子列表
MULTI_SEED_LIST = [42, 123, 2023, 999, 0]

# 所有支援的資料集
ALL_DATASETS = ['restaurants', 'laptops', 'mams', 'rest16', 'lap16']

# 消融實驗推薦的執行順序（按重要性）
ABLATION_ORDER = [
    'full',              # 基準
    'no_senticnet',      # 核心創新 1
    'no_dynamic_gate',   # 核心創新 2 (v3.0)
    'no_confidence_gate', # 核心創新 3 (v2.0)
    'no_contrastive',    # 輔助機制 1
    'no_logit_adjust',   # 輔助機制 2
    'no_focal_loss',     # 訓練技巧 1
    'no_llrd',           # 訓練技巧 2
]


def run_ablation_experiment(ablation_type, dataset, seed_override=None):
    """執行單個消融實驗

    Args:
        ablation_type: 消融變體名稱
        dataset: 資料集名稱
        seed_override: 覆蓋配置文件中的 seed（用於多種子實驗）

    Returns:
        bool: 是否成功
    """
    config_path = ABLATION_CONFIGS.get(ablation_type)
    if not config_path:
        print(f"錯誤: 未知的消融變體 '{ablation_type}'")
        return False

    if not Path(config_path).exists():
        print(f"錯誤: 配置檔案不存在 '{config_path}'")
        return False

    cmd = [
        sys.executable,
        "experiments/train_from_config.py",
        "--config", config_path,
        "--dataset", dataset
    ]

    # 如果指定了 seed_override，通過 --override 傳遞
    if seed_override is not None:
        cmd.extend(["--override", "--seed", str(seed_override)])

    description = f"Ablation: {ablation_type} on {dataset.upper()}"
    if seed_override is not None:
        description += f" (seed={seed_override})"

    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\n中斷執行")
        return False


def run_multi_seed_ablation(ablation_type, dataset):
    """執行多種子消融實驗

    Args:
        ablation_type: 消融變體名稱
        dataset: 資料集名稱

    Returns:
        bool: 所有種子是否都成功
    """
    # 消融實驗結果存放在 results/ablation/{dataset}/ 下
    results_dir = Path("results/ablation") / dataset
    seeds = MULTI_SEED_LIST

    print(f"\n{'='*80}")
    print(f"[Multi-Seed Ablation] {ablation_type} on {dataset.upper()}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}\n")

    results = []
    success_count = 0

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(seeds)}] Running {ablation_type} with seed={seed}")
        print(f"{'='*60}\n")

        success = run_ablation_experiment(ablation_type, dataset, seed_override=seed)

        if success:
            success_count += 1
            # 讀取最新實驗結果
            result = get_latest_experiment_result(results_dir, ablation_type)
            if result:
                results.append({
                    'seed': seed,
                    'accuracy': result.get('accuracy', 0),
                    'f1_macro': result.get('f1_macro', 0),
                    'f1_per_class': result.get('f1_per_class', [0, 0, 0]),
                })
                print(f"  [OK] seed={seed}: Acc={result.get('accuracy', 0)*100:.2f}%, F1={result.get('f1_macro', 0)*100:.2f}%")
            else:
                print(f"  [??] seed={seed}: 無法讀取結果")
        else:
            print(f"  [FAIL] seed={seed}: 失敗")

    # 生成多種子報告
    if results:
        generate_multi_seed_ablation_report(ablation_type, dataset, results)

    print(f"\n{'='*60}")
    print(f"[Multi-Seed Summary] {success_count}/{len(seeds)} experiments succeeded")
    print(f"{'='*60}\n")

    return success_count == len(seeds)


def get_latest_experiment_result(results_dir, ablation_type):
    """獲取最新實驗的測試結果"""
    if not results_dir.exists():
        return None

    # 尋找對應消融實驗的目錄
    exp_pattern = f"*ablation_{ablation_type}*" if ablation_type != 'full' else "*ablation_full*"
    exp_dirs = sorted(results_dir.glob(exp_pattern), key=lambda x: x.stat().st_mtime, reverse=True)

    if not exp_dirs:
        # 如果找不到特定模式，嘗試找最新的
        exp_dirs = sorted(results_dir.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True)

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


def generate_multi_seed_ablation_report(ablation_type, dataset, results):
    """生成多種子消融實驗報告"""
    # 報告存到 results/ablation/{dataset}/ 下
    reports_dir = Path("results/ablation") / dataset
    reports_dir.mkdir(parents=True, exist_ok=True)

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
    report.append(f"消融實驗報告: {ablation_type} - {dataset.upper()}")
    report.append(f"描述: {ABLATION_DESCRIPTIONS.get(ablation_type, 'N/A')}")
    report.append("=" * 80)
    report.append(f"Seeds: {[r['seed'] for r in results]}")
    report.append(f"Runs: {len(results)}")
    report.append("")

    report.append("-" * 80)
    report.append("Individual Results")
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
    report.append("=" * 80)

    # 保存報告
    report_text = "\n".join(report)
    output_file = reports_dir / f"ablation_{ablation_type}_{dataset}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\n報告已保存至: {output_file}")

    # 同時保存 JSON 格式方便後續分析
    json_output = {
        'ablation_type': ablation_type,
        'description': ABLATION_DESCRIPTIONS.get(ablation_type, 'N/A'),
        'dataset': dataset,
        'seeds': [r['seed'] for r in results],
        'individual_results': results,
        'aggregated': {
            'accuracy': {'mean': acc_mean, 'std': acc_std},
            'f1_macro': {'mean': f1_mean, 'std': f1_std},
            'f1_neg': {'mean': neg_mean, 'std': neg_std},
            'f1_neu': {'mean': neu_mean, 'std': neu_std},
            'f1_pos': {'mean': pos_mean, 'std': pos_std},
        }
    }

    json_file = reports_dir / f"ablation_{ablation_type}_{dataset}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)


def run_all_ablations(dataset, multi_seed=False):
    """執行所有消融實驗（單一資料集）

    Args:
        dataset: 資料集名稱
        multi_seed: 是否執行多種子實驗
    """
    print(f"\n{'='*80}")
    print(f"[All Ablations] Running on {dataset.upper()}")
    print(f"Variants: {len(ABLATION_ORDER)}")
    print(f"Multi-seed: {multi_seed}")
    print(f"{'='*80}\n")

    results = {}

    for ablation_type in ABLATION_ORDER:
        print(f"\n{'#'*80}")
        print(f"# Ablation: {ablation_type}")
        print(f"# {ABLATION_DESCRIPTIONS.get(ablation_type, 'N/A')}")
        print(f"{'#'*80}")

        if multi_seed:
            success = run_multi_seed_ablation(ablation_type, dataset)
        else:
            success = run_ablation_experiment(ablation_type, dataset)

        results[ablation_type] = 'Success' if success else 'Failed'

    # 總結報告
    print(f"\n{'='*80}")
    print(f"[All Ablations Summary] {dataset.upper()}")
    print(f"{'='*80}")
    for ablation_type in ABLATION_ORDER:
        status = results.get(ablation_type, 'N/A')
        desc = ABLATION_DESCRIPTIONS.get(ablation_type, 'N/A')
        print(f"  {ablation_type:<20s}: {status:<10s} ({desc})")
    print(f"{'='*80}\n")


def run_full_study(multi_seed=False):
    """執行完整消融研究（所有變體 × 所有資料集）"""
    print(f"\n{'='*80}")
    print(f"[Full Ablation Study]")
    print(f"Datasets: {len(ALL_DATASETS)}")
    print(f"Variants: {len(ABLATION_ORDER)}")
    print(f"Total experiments: {len(ALL_DATASETS) * len(ABLATION_ORDER)}")
    if multi_seed:
        print(f"Seeds per experiment: {len(MULTI_SEED_LIST)}")
        print(f"Total runs: {len(ALL_DATASETS) * len(ABLATION_ORDER) * len(MULTI_SEED_LIST)}")
    print(f"{'='*80}\n")

    all_results = {}

    for dataset in ALL_DATASETS:
        print(f"\n{'#'*80}")
        print(f"# Dataset: {dataset.upper()}")
        print(f"{'#'*80}")

        run_all_ablations(dataset, multi_seed=multi_seed)
        all_results[dataset] = 'Completed'

    # 最終總結
    print(f"\n{'='*80}")
    print(f"[Full Study Complete]")
    print(f"{'='*80}")
    for dataset, status in all_results.items():
        print(f"  {dataset.upper():<12s}: {status}")
    print(f"{'='*80}\n")


def generate_ablation_summary_report():
    """生成消融實驗總結報告（從已有的 JSON 結果）"""
    reports_dir = Path("results/ablation")
    if not reports_dir.exists():
        print("錯誤: 沒有找到消融實驗結果")
        return

    # 收集所有 JSON 結果
    all_results = {}

    for json_file in reports_dir.glob("ablation_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                ablation_type = data['ablation_type']
                dataset = data['dataset']

                if dataset not in all_results:
                    all_results[dataset] = {}

                all_results[dataset][ablation_type] = {
                    'acc_mean': data['aggregated']['accuracy']['mean'],
                    'acc_std': data['aggregated']['accuracy']['std'],
                    'f1_mean': data['aggregated']['f1_macro']['mean'],
                    'f1_std': data['aggregated']['f1_macro']['std'],
                    'neu_mean': data['aggregated']['f1_neu']['mean'],
                    'neu_std': data['aggregated']['f1_neu']['std'],
                }
        except Exception as e:
            print(f"警告: 無法讀取 {json_file}: {e}")

    if not all_results:
        print("錯誤: 沒有找到有效的消融實驗結果")
        return

    # 生成總結報告
    report = []
    report.append("=" * 100)
    report.append("消融實驗總結報告 (Ablation Study Summary)")
    report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)
    report.append("")

    for dataset in ALL_DATASETS:
        if dataset not in all_results:
            continue

        results = all_results[dataset]

        report.append("-" * 100)
        report.append(f"Dataset: {dataset.upper()}")
        report.append("-" * 100)
        report.append(f"{'Variant':<25} {'Acc (%)':<15} {'Macro-F1 (%)':<15} {'Neu F1 (%)':<15} {'Δ F1':<10}")
        report.append("-" * 100)

        # 獲取基準（full）的 F1
        baseline_f1 = results.get('full', {}).get('f1_mean', 0)

        for ablation_type in ABLATION_ORDER:
            if ablation_type not in results:
                continue

            r = results[ablation_type]
            delta = r['f1_mean'] - baseline_f1 if ablation_type != 'full' else 0
            delta_str = f"{delta:+.2f}" if ablation_type != 'full' else "-"

            report.append(
                f"{ablation_type:<25} "
                f"{r['acc_mean']:.2f}±{r['acc_std']:.2f}    "
                f"{r['f1_mean']:.2f}±{r['f1_std']:.2f}    "
                f"{r['neu_mean']:.2f}±{r['neu_std']:.2f}    "
                f"{delta_str}"
            )

        report.append("")

    report.append("=" * 100)
    report.append("說明:")
    report.append("  - Δ F1: 相對於 full (完整模型) 的 Macro-F1 變化")
    report.append("  - 負值表示移除該組件後性能下降（該組件有正面貢獻）")
    report.append("=" * 100)

    # 保存報告
    report_text = "\n".join(report)
    output_file = reports_dir / "ablation_summary.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n報告已保存至: {output_file}")


def list_ablations():
    """列出所有可用的消融變體"""
    print("\n可用的消融變體:")
    print("=" * 60)
    for ablation_type in ABLATION_ORDER:
        config = ABLATION_CONFIGS.get(ablation_type, 'N/A')
        desc = ABLATION_DESCRIPTIONS.get(ablation_type, 'N/A')
        exists = "[O]" if Path(config).exists() else "[X]"
        print(f"  {exists} {ablation_type:<20s}: {desc}")
        print(f"      Config: {config}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='消融實驗執行腳本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
    # 執行單個消融實驗
    python run_ablation.py --ablation no_senticnet --dataset laptops

    # 執行多種子消融實驗
    python run_ablation.py --ablation no_senticnet --dataset laptops --multi-seed

    # 執行所有消融實驗（單一資料集）
    python run_ablation.py --all --dataset laptops

    # 執行完整消融研究
    python run_ablation.py --full-study

    # 生成消融報告
    python run_ablation.py --report-only

    # 列出所有可用的消融變體
    python run_ablation.py --list
        """
    )

    parser.add_argument('--ablation', type=str, default=None,
                        choices=list(ABLATION_CONFIGS.keys()),
                        help='消融變體名稱')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=ALL_DATASETS,
                        help='資料集選擇')
    parser.add_argument('--multi-seed', action='store_true',
                        help='多種子模式: 使用 5 個種子驗證穩定性')
    parser.add_argument('--all', action='store_true',
                        help='執行所有消融變體（需指定 --dataset）')
    parser.add_argument('--full-study', action='store_true',
                        help='執行完整消融研究（所有變體 × 所有資料集）')
    parser.add_argument('--report-only', action='store_true',
                        help='只生成消融報告（不執行實驗）')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的消融變體')

    args = parser.parse_args()

    # 列出消融變體
    if args.list:
        list_ablations()
        return

    # 只生成報告（調用專門的報告生成腳本）
    if args.report_only:
        import subprocess
        cmd = [sys.executable, "experiments/generate_ablation_report.py", "--all", "--use-multi-seed"]
        if args.dataset:
            cmd = [sys.executable, "experiments/generate_ablation_report.py", "--dataset", args.dataset, "--use-multi-seed"]
        subprocess.run(cmd)
        return

    # 完整消融研究
    if args.full_study:
        run_full_study(multi_seed=args.multi_seed)
        return

    # 執行所有消融變體
    if args.all:
        if args.dataset is None:
            parser.error("--all 需要指定 --dataset")
        run_all_ablations(args.dataset, multi_seed=args.multi_seed)
        return

    # 執行單個消融實驗
    if args.ablation:
        if args.dataset is None:
            parser.error("--ablation 需要指定 --dataset")

        if args.multi_seed:
            run_multi_seed_ablation(args.ablation, args.dataset)
        else:
            run_ablation_experiment(args.ablation, args.dataset)
        return

    # 沒有指定任何動作
    parser.print_help()


if __name__ == "__main__":
    main()
