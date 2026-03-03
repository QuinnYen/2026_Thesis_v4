"""
Knowledge Weight 驗證實驗腳本

目的：驗證 knowledge_weight=0.1 是否為知識模組貢獻不顯著的根本原因

實驗矩陣（相對消融實驗中的 full 基準）：
    A（已有）: use_gates=True,  knowledge_weight=0.1  → 消融基準
    B        : use_gates=True,  knowledge_weight=0.3  → 驗證訊號強度
    C        : use_gates=True,  knowledge_weight=0.5  → 更強訊號
    D        : use_gates=False, knowledge_weight=0.3  → 硬性注入+提高強度
    E        : use_gates=False, knowledge_weight=0.5  → 硬性注入+最強強度

判讀標準：
    若 B/C/D/E 中，full 明顯優於 no_all_knowledge（Delta > 1.0%）
        → 確認 knowledge_weight 是根本原因
    若 B/C ≈ full(0.1) 但 D/E 明顯提升
        → Gate 過度壓制是主因
    若 B > C (0.3 比 0.5 好)
        → knowledge_weight 在 0.3 附近有最佳點

使用方法：
    # 在 REST16 + LAP16 上執行驗證實驗（各 3 seeds）
    python run_validation.py --datasets rest16 lap16

    # 只執行特定變體
    python run_validation.py --variants kw_03 kw_05 --datasets rest16

    # 只生成比較報告（從已有結果）
    python run_validation.py --report-only --datasets rest16 lap16

    # 也在 REST14/LAP14 上執行（確認是否有同樣問題）
    python run_validation.py --datasets restaurants laptops rest16 lap16
"""

import subprocess
import argparse
from pathlib import Path
import sys
import json
import numpy as np
from datetime import datetime

# =============================================================================
# 驗證實驗配置
# =============================================================================

VALIDATION_CONFIGS = {
    'kw_03':        'configs/validation/kw_03.yaml',
    'kw_05':        'configs/validation/kw_05.yaml',
    'kw_03_no_gate': 'configs/validation/kw_03_no_gate.yaml',
    'kw_05_no_gate': 'configs/validation/kw_05_no_gate.yaml',
}

VALIDATION_DESCRIPTIONS = {
    'kw_03':         '[實驗B] knowledge_weight=0.3, Gates=開啟',
    'kw_05':         '[實驗C] knowledge_weight=0.5, Gates=開啟',
    'kw_03_no_gate': '[實驗D] knowledge_weight=0.3, Gates=關閉（硬性注入）',
    'kw_05_no_gate': '[實驗E] knowledge_weight=0.5, Gates=關閉（硬性注入）',
}

VALIDATION_ORDER = ['kw_03', 'kw_05', 'kw_03_no_gate', 'kw_05_no_gate']

# 驗證實驗用 3 seeds（快速確認方向）
VALIDATION_SEEDS = [42, 123, 2023]

# 資料集顯示名稱
DATASET_DISPLAY_NAMES = {
    'restaurants': 'Rest14',
    'laptops':     'Lap14',
    'mams':        'MAMS',
    'rest16':      'Rest16',
    'lap16':       'Lap16',
}

# 對應消融實驗的 experiment_name（用於尋找結果目錄）
VALIDATION_EXPERIMENT_NAMES = {
    'kw_03':         'ablation_kw_03',
    'kw_05':         'ablation_kw_05',
    'kw_03_no_gate': 'ablation_kw_03_no_gate',
    'kw_05_no_gate': 'ablation_kw_05_no_gate',
}


def run_validation_experiment(variant, dataset, seed_override=None):
    """執行單個驗證實驗

    Args:
        variant: 驗證變體名稱
        dataset: 資料集名稱
        seed_override: 覆蓋配置文件中的 seed

    Returns:
        bool: 是否成功
    """
    config_path = VALIDATION_CONFIGS.get(variant)
    if not config_path:
        print(f"錯誤: 未知的驗證變體 '{variant}'")
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

    if seed_override is not None:
        cmd.extend(["--override", "--seed", str(seed_override)])

    description = f"Validation: {variant} on {DATASET_DISPLAY_NAMES.get(dataset, dataset.upper())}"
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
        print(f"\n[X] {description} FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print("\n中斷執行")
        return False


def get_latest_experiment_result(results_dir, experiment_name):
    """獲取最新實驗的測試結果

    Args:
        results_dir: 結果目錄（results/ablation/{dataset}/）
        experiment_name: 實驗名稱前綴（如 ablation_kw_03）

    Returns:
        dict or None
    """
    if not results_dir.exists():
        return None

    exp_dirs = sorted(
        results_dir.glob(f"*{experiment_name}*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

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


def run_multi_seed_validation(variant, dataset, seeds=None):
    """執行多種子驗證實驗

    Returns:
        list: 各 seed 的結果
    """
    if seeds is None:
        seeds = VALIDATION_SEEDS
    results_dir = Path("results/ablation") / dataset
    exp_name = VALIDATION_EXPERIMENT_NAMES[variant]

    print(f"\n{'='*80}")
    print(f"[Multi-Seed Validation] {variant} on {DATASET_DISPLAY_NAMES.get(dataset, dataset.upper())}")
    print(f"Seeds: {seeds}")
    print(f"{'='*80}\n")

    results = []
    success_count = 0

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(seeds)}] {variant} seed={seed}")
        print(f"{'='*60}\n")

        success = run_validation_experiment(variant, dataset, seed_override=seed)

        if success:
            success_count += 1
            result = get_latest_experiment_result(results_dir, exp_name)
            if result:
                results.append({
                    'seed': seed,
                    'accuracy': result.get('accuracy', 0),
                    'f1_macro': result.get('f1_macro', 0),
                    'f1_per_class': result.get('f1_per_class', [0, 0, 0]),
                })
                print(f"  [OK] seed={seed}: F1={result.get('f1_macro', 0)*100:.2f}%")
            else:
                print(f"  [??] seed={seed}: 無法讀取結果")
        else:
            print(f"  [FAIL] seed={seed}")

    print(f"\n[Summary] {success_count}/{len(seeds)} succeeded")
    return results


def load_ablation_baseline(dataset):
    """從消融實驗結果讀取基準（full 模型 + no_all_knowledge）

    Returns:
        dict: {'full': {'f1_mean': ..., 'f1_std': ...}, 'no_all_knowledge': {...}}
    """
    baselines = {}
    ablation_dir = Path("results/ablation") / dataset

    for key in ['full', 'no_all_knowledge']:
        json_path = ablation_dir / f"ablation_{key}_{dataset}.json"
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                agg = data.get('aggregated', {})
                baselines[key] = {
                    'f1_mean': agg.get('f1_macro', {}).get('mean', 0),
                    'f1_std':  agg.get('f1_macro', {}).get('std', 0),
                }
            except Exception:
                pass

    return baselines


def compute_stats(results):
    """計算統計數據"""
    if not results:
        return None

    f1_macros = [r['f1_macro'] * 100 for r in results]
    accuracies = [r['accuracy'] * 100 for r in results]
    f1_neg = [r['f1_per_class'][0] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_neu = [r['f1_per_class'][1] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_pos = [r['f1_per_class'][2] * 100 for r in results if len(r['f1_per_class']) >= 3]

    return {
        'f1_mean':  float(np.mean(f1_macros)),
        'f1_std':   float(np.std(f1_macros)),
        'acc_mean': float(np.mean(accuracies)),
        'neg_mean': float(np.mean(f1_neg)) if f1_neg else 0,
        'neu_mean': float(np.mean(f1_neu)) if f1_neu else 0,
        'pos_mean': float(np.mean(f1_pos)) if f1_pos else 0,
        'n_seeds':  len(results),
    }


def generate_comparison_report(dataset, all_results):
    """生成驗證實驗比較報告

    Args:
        dataset: 資料集名稱
        all_results: {variant: [results]}
    """
    display = DATASET_DISPLAY_NAMES.get(dataset, dataset.upper())
    baselines = load_ablation_baseline(dataset)

    full_f1    = baselines.get('full', {}).get('f1_mean', None)
    no_know_f1 = baselines.get('no_all_knowledge', {}).get('f1_mean', None)

    report = []
    report.append("=" * 80)
    report.append(f"Knowledge Weight 驗證實驗報告 - {display}")
    report.append(f"時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)

    # 基準資訊
    report.append("\n[基準（來自消融實驗，5 seeds）]")
    if full_f1 is not None:
        report.append(f"  A - full (kw=0.1, gates=開): {full_f1:.2f}%")
    else:
        report.append(f"  A - full (kw=0.1, gates=開): 尚無結果")
    if no_know_f1 is not None:
        baseline_delta = no_know_f1 - (full_f1 or 0)
        report.append(f"  no_all_knowledge:             {no_know_f1:.2f}% (Delta={baseline_delta:+.2f}%)")
    report.append("")

    # 驗證實驗結果
    report.append("[驗證實驗結果（3 seeds）]")
    report.append("-" * 80)
    header = f"{'變體':<22} {'Macro-F1':>10} {'±Std':>8} {'Delta vs A':>12} {'Delta vs Δ(nk)':>14} {'判讀'}"
    report.append(header)
    report.append("-" * 80)

    summary = {}
    for variant in VALIDATION_ORDER:
        if variant not in all_results or not all_results[variant]:
            report.append(f"  {variant:<20} {'(無結果)'}")
            continue

        stats = compute_stats(all_results[variant])
        if not stats:
            continue

        summary[variant] = stats

        delta_a  = stats['f1_mean'] - (full_f1 or stats['f1_mean'])
        delta_nk = (no_know_f1 - stats['f1_mean']) if no_know_f1 else 0

        # 判讀：full 與 no_all_knowledge 的 delta 是否擴大
        if full_f1 and no_know_f1:
            # 在此配置下，full 相對 no_all_knowledge 的差距
            orig_delta = full_f1 - no_know_f1
            # 若新配置的 full 比 no_all_knowledge 差距更大 → 知識模組更有效
            verdict = "✅ 知識有效" if delta_a > 0.5 else ("⚠️ 略有改善" if delta_a > 0.1 else "❌ 無明顯改善")
        else:
            verdict = "（無基準可比）"

        report.append(
            f"  {variant:<20} {stats['f1_mean']:>10.2f}% {stats['f1_std']:>7.2f}% "
            f"{delta_a:>+12.2f}% {-delta_nk:>+14.2f}%  {verdict}"
        )

    report.append("-" * 80)

    # 結論
    report.append("")
    report.append("[結論]")
    if summary:
        best_variant = max(summary, key=lambda v: summary[v]['f1_mean'])
        best_f1 = summary[best_variant]['f1_mean']
        best_delta = best_f1 - (full_f1 or best_f1)

        if best_delta > 1.0:
            report.append(f"✅ 確認根本原因：提高 knowledge_weight 有顯著效果")
            report.append(f"   最佳變體: {best_variant} ({best_f1:.2f}%, Delta={best_delta:+.2f}%)")
            report.append(f"   建議：將 base_hkgan.yaml 中 knowledge_weight 改為對應值並重跑完整消融")
        elif best_delta > 0.3:
            report.append(f"⚠️ 輕微改善：提高 knowledge_weight 有幫助但效果不如預期")
            report.append(f"   最佳變體: {best_variant} ({best_f1:.2f}%, Delta={best_delta:+.2f}%)")
            report.append(f"   建議：考慮同時加入 Gate 正則化")
        else:
            report.append(f"❌ 無顯著改善：knowledge_weight 可能不是唯一問題")
            report.append(f"   建議：檢查知識注入位置或架構設計")

    report.append("")
    report.append("=" * 80)

    report_text = "\n".join(report)

    # 儲存報告
    output_dir = Path("results/validation") / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"kw_validation_{dataset}_{timestamp}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(report_text)
    print(f"\n報告已保存至: {output_file}")

    # 同時保存 JSON
    json_data = {
        'dataset': dataset,
        'timestamp': timestamp,
        'baselines': baselines,
        'validation_results': {
            v: compute_stats(r) for v, r in all_results.items() if r
        }
    }
    json_file = output_dir / f"kw_validation_{dataset}_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description='Knowledge Weight 驗證實驗（確認知識模組貢獻問題的根本原因）'
    )
    parser.add_argument(
        '--datasets', nargs='+',
        default=['rest16', 'lap16'],
        choices=['restaurants', 'laptops', 'mams', 'rest16', 'lap16'],
        help='要執行的資料集（預設: rest16 lap16）'
    )
    parser.add_argument(
        '--variants', nargs='+',
        default=VALIDATION_ORDER,
        choices=list(VALIDATION_CONFIGS.keys()),
        help='要執行的驗證變體（預設: 全部 4 個）'
    )
    parser.add_argument(
        '--report-only', action='store_true',
        help='只生成比較報告，不執行新實驗'
    )
    parser.add_argument(
        '--seeds', nargs='+', type=int,
        default=VALIDATION_SEEDS,
        help=f'使用的種子列表（預設: {VALIDATION_SEEDS}）'
    )

    args = parser.parse_args()

    # 使用 args.seeds（不修改全域變數）
    seeds = args.seeds

    print("\n" + "=" * 80)
    print("Knowledge Weight 驗證實驗")
    print(f"資料集: {[DATASET_DISPLAY_NAMES.get(d, d) for d in args.datasets]}")
    print(f"變體:   {args.variants}")
    print(f"Seeds:  {seeds}")
    print("=" * 80 + "\n")

    for dataset in args.datasets:
        all_results = {}

        if not args.report_only:
            for variant in args.variants:
                results = run_multi_seed_validation(variant, dataset, seeds)
                all_results[variant] = results
        else:
            # 從已有結果讀取
            results_dir = Path("results/ablation") / dataset
            for variant in args.variants:
                exp_name = VALIDATION_EXPERIMENT_NAMES[variant]
                # 嘗試讀取已存在的單次結果（只讀最新的幾個）
                exp_dirs = sorted(
                    results_dir.glob(f"*{exp_name}*"),
                    key=lambda x: x.stat().st_mtime,
                    reverse=True
                )[:len(seeds)]

                results = []
                for seed_idx, exp_dir in enumerate(exp_dirs):
                    result_file = exp_dir / "reports" / "experiment_results.json"
                    if result_file.exists():
                        with open(result_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        metrics = data.get('test_metrics', {})
                        results.append({
                            'seed': seeds[seed_idx] if seed_idx < len(seeds) else seed_idx,
                            'accuracy': metrics.get('accuracy', 0),
                            'f1_macro': metrics.get('f1_macro', 0),
                            'f1_per_class': metrics.get('f1_per_class', [0, 0, 0]),
                        })
                all_results[variant] = results

        # 生成比較報告
        generate_comparison_report(dataset, all_results)

    print("\n✅ 驗證實驗完成")


if __name__ == "__main__":
    main()
