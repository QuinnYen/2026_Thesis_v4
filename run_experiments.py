"""
批次執行實驗腳本

用法：
    # ── 單資料集：HKGAN ─────────────────────────────────────────
    python run_experiments.py --dataset restaurants --hkgan [--multi-seed]
    python run_experiments.py --dataset laptops     --hkgan [--multi-seed]
    python run_experiments.py --dataset mams        --hkgan [--multi-seed]
    python run_experiments.py --dataset rest16      --hkgan [--multi-seed]
    python run_experiments.py --dataset lap16       --hkgan [--multi-seed]

    # ── 單資料集：Baseline ──────────────────────────────────────
    python run_experiments.py --dataset restaurants --baseline [--multi-seed]
    python run_experiments.py --dataset laptops     --baseline [--multi-seed]
    python run_experiments.py --dataset mams        --baseline [--multi-seed]
    python run_experiments.py --dataset rest16      --baseline [--multi-seed]
    python run_experiments.py --dataset lap16       --baseline [--multi-seed]

    # ── 全資料集 ────────────────────────────────────────────────
    python run_experiments.py --hkgan --full-run [--multi-seed] [--auto-cleanup]
    python run_experiments.py --full-baseline    [--multi-seed]

    # ── 論文全流程（Baseline → HKGAN → 消融，5 資料集）─────────
    python run_experiments.py --full-thesis --multi-seed --auto-cleanup

    # ── 報告與統計（HKGAN 對比 + 統計顯著性 + 論文圖表）────────
    python run_experiments.py --report-only          # 全資料集，自動跳過無資料者

    # ── Checkpoint 清理 ─────────────────────────────────────────
    python run_experiments.py --cleanup-only           # dry-run，列出將刪除的內容
    python run_experiments.py --cleanup-only --execute  # 備份 txt → 刪除實驗資料夾
"""

import subprocess
import argparse
from pathlib import Path
import sys
import io
import json
import shutil
import datetime
import numpy as np
from scipy import stats
from utils.checkpoint_cleaner import run_cleanup, print_cleanup_summary

# Windows cp950 終端機無法顯示 Unicode 特殊符號，強制使用 utf-8 輸出
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

MULTI_SEED_LIST = [42, 123, 2023, 999, 0]
ALL_DATASETS    = ['restaurants', 'laptops', 'mams', 'rest16', 'lap16']
DISPLAY_NAMES   = {
    'restaurants': 'REST14',
    'laptops':     'LAP14',
    'mams':        'MAMS',
    'rest16':      'REST16',
    'lap16':       'LAP16',
}


def get_display_name(dataset):
    return DISPLAY_NAMES.get(dataset, dataset.upper())


# ──────────────────────────────────────────────────────────────
# 基礎執行單元
# ──────────────────────────────────────────────────────────────

def run_experiment(config_path, description, dataset, seed_override=None):
    """呼叫 train_from_config.py 執行單次實驗。"""
    cmd = [
        sys.executable,
        "experiments/train_from_config.py",
        "--config", str(config_path),
        "--dataset", dataset
    ]
    if seed_override is not None:
        cmd.extend(["--override", "--seed", str(seed_override)])

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {description} FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def generate_hkgan_report(dataset):
    """呼叫 generate_hkgan_report.py 生成對比報告。"""
    import os
    cmd = [sys.executable, "experiments/generate_hkgan_report.py", "--dataset", dataset]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] HKGAN Report generation FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


def generate_thesis_figures():
    """呼叫 plot_thesis_figures.py 生成論文圖表（含 ROC 曲線）。"""
    cmd = [
        sys.executable,
        "experiments/plot_thesis_figures.py",
        "--figure", "all",
        "--output", "results/figures/",
        "--no-show"
    ]
    import os
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    try:
        print(f"\n{'='*60}\n生成論文圖表（含 ROC 曲線）...\n{'='*60}\n")
        subprocess.run(cmd, check=True, env=env)
        print(f"\n[OK] 圖表已保存至 results/figures/")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] Figure generation FAILED (code: {e.returncode})\n")
        return False
    except KeyboardInterrupt:
        print(f"\nInterrupted")
        return False


# ──────────────────────────────────────────────────────────────
# Baseline 實驗
# ──────────────────────────────────────────────────────────────

def run_baseline_only(dataset, multi_seed=False):
    """執行 Baseline（單種子或多種子）。"""
    config_path = Path("configs/unified_baseline.yaml")
    if not config_path.exists():
        print(f"  ERROR: unified_baseline.yaml not found")
        return False

    if multi_seed:
        return run_multi_seed_baseline(dataset, config_path)

    print(f"\n{'='*60}\n[Baseline] {get_display_name(dataset)}\n{'='*60}\n")
    success = run_experiment(config_path, "Baseline: BERT-CLS", dataset)
    if success:
        print(f"\n[Summary] 1/1 succeeded")
    return success


def run_multi_seed_baseline(dataset, config_path):
    """以 MULTI_SEED_LIST 中所有種子執行 Baseline，供統計顯著性檢驗使用。"""
    results_dir = Path("results/baseline") / dataset

    print(f"\n{'='*80}\n[Multi-Seed Baseline] {get_display_name(dataset)}  seeds={MULTI_SEED_LIST}\n{'='*80}\n")

    results = []
    success_count = 0

    for i, seed in enumerate(MULTI_SEED_LIST):
        print(f"\n{'='*60}\n[{i+1}/{len(MULTI_SEED_LIST)}] Baseline seed={seed}\n{'='*60}\n")
        ok = run_experiment(config_path, f"Baseline seed={seed}", dataset, seed_override=seed)

        if ok:
            success_count += 1
            exp_dirs = sorted(results_dir.glob("*_baseline_*"),
                              key=lambda x: x.stat().st_mtime, reverse=True)
            if exp_dirs:
                rf = exp_dirs[0] / "reports" / "experiment_results.json"
                if rf.exists():
                    result = json.load(open(rf))
                    m = result.get('test_metrics', {})
                    results.append({
                        'seed':        seed,
                        'accuracy':    m.get('accuracy', 0),
                        'f1_macro':    m.get('f1_macro', 0),
                        'f1_per_class':m.get('f1_per_class', [0, 0, 0]),
                        'auc_macro':   m.get('auc_macro'),
                    })

    print(f"\n{'='*80}\n[Multi-Seed Baseline] {success_count}/{len(MULTI_SEED_LIST)} succeeded\n{'='*80}\n")
    if results:
        generate_baseline_multiseed_report(dataset, results)
    return success_count == len(MULTI_SEED_LIST)


def generate_baseline_multiseed_report(dataset, results):
    """將 Baseline 多種子結果寫成 txt 報告並印出。"""
    reports_dir = Path("results")
    reports_dir.mkdir(parents=True, exist_ok=True)

    def mean_std(arr):
        return (float(np.mean(arr)), float(np.std(arr))) if arr else (0, 0)

    accuracies  = [r['accuracy']  * 100 for r in results]
    f1_macros   = [r['f1_macro']  * 100 for r in results]
    f1_neg      = [r['f1_per_class'][0] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_neu      = [r['f1_per_class'][1] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_pos      = [r['f1_per_class'][2] * 100 for r in results if len(r['f1_per_class']) >= 3]
    auc_macros  = [r['auc_macro'] * 100 for r in results if r.get('auc_macro') is not None]

    acc_mean, acc_std = mean_std(accuracies)
    f1_mean,  f1_std  = mean_std(f1_macros)
    neg_mean, neg_std = mean_std(f1_neg)
    neu_mean, neu_std = mean_std(f1_neu)
    pos_mean, pos_std = mean_std(f1_pos)
    auc_mean, auc_std = mean_std(auc_macros)

    report = [
        "=" * 80,
        f"Baseline 多種子實驗報告 - {get_display_name(dataset)} Dataset",
        "=" * 80,
        f"Seeds: {[r['seed'] for r in results]}  Runs: {len(results)}",
        "",
        "-" * 80,
        "Individual Results",
        "-" * 80,
        f"{'Seed':<10} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'AUC (%)':<12} {'Neg F1':<10} {'Neu F1':<10} {'Pos F1':<10}",
        "-" * 80,
    ]
    for r in results:
        fc = r['f1_per_class']
        auc_s = f"{r['auc_macro']*100:<12.2f}" if r.get('auc_macro') else "N/A         "
        report.append(f"{r['seed']:<10} {r['accuracy']*100:<12.2f} {r['f1_macro']*100:<15.2f} {auc_s} {fc[0]*100:<10.2f} {fc[1]*100:<10.2f} {fc[2]*100:<10.2f}")

    report += [
        "-" * 80,
        "",
        "-" * 80,
        "Aggregated Statistics (Mean ± Std)",
        "-" * 80,
        f"  Accuracy:  {acc_mean:.2f}% ± {acc_std:.2f}%",
        f"  Macro-F1:  {f1_mean:.2f}% ± {f1_std:.2f}%",
    ]
    if auc_mean > 0:
        report.append(f"  AUC:       {auc_mean:.2f}% ± {auc_std:.2f}%")
    report += [
        f"  Neg F1:    {neg_mean:.2f}% ± {neg_std:.2f}%",
        f"  Neu F1:    {neu_mean:.2f}% ± {neu_std:.2f}%",
        f"  Pos F1:    {pos_mean:.2f}% ± {pos_std:.2f}%",
        "-" * 80,
        "=" * 80,
    ]

    report_text = "\n".join(report)
    output_file = reports_dir / f"Baseline_MultiSeed_{dataset}.txt"
    output_file.write_text(report_text, encoding='utf-8')
    print(f"\n{report_text}\n報告已保存至: {output_file}")


# ──────────────────────────────────────────────────────────────
# HKGAN 實驗
# ──────────────────────────────────────────────────────────────

def run_hkgan_experiments(dataset, multi_seed=False):
    """執行 HKGAN（單種子或多種子）。"""
    config_path = Path("configs/unified_hkgan.yaml")
    if not config_path.exists():
        print(f"  ERROR: unified_hkgan.yaml not found")
        return False

    if multi_seed:
        return run_multi_seed_hkgan(dataset, config_path)

    print(f"\n{'='*60}\n[HKGAN] {get_display_name(dataset)}\n{'='*60}\n")
    success = run_experiment(config_path, "HKGAN", dataset)
    if success:
        print(f"\n[Summary] 1/1 succeeded")
    generate_hkgan_report(dataset)
    return success


def run_multi_seed_hkgan(dataset, config_path):
    """以 MULTI_SEED_LIST 中所有種子執行 HKGAN，計算 mean±std 並儲存報告。"""
    results_dir = Path("results/improved") / dataset

    print(f"\n{'='*80}\n[Multi-Seed HKGAN] {get_display_name(dataset)}  seeds={MULTI_SEED_LIST}\n{'='*80}\n")

    results = []
    success_count = 0

    for i, seed in enumerate(MULTI_SEED_LIST):
        print(f"\n{'='*60}\n[{i+1}/{len(MULTI_SEED_LIST)}] HKGAN seed={seed}\n{'='*60}\n")
        ok = run_experiment(config_path, f"HKGAN (seed={seed})", dataset, seed_override=seed)

        if ok:
            success_count += 1
            result = get_latest_experiment_result(results_dir)
            if result:
                results.append({
                    'seed':        seed,
                    'accuracy':    result.get('accuracy', 0),
                    'f1_macro':    result.get('f1_macro', 0),
                    'f1_per_class':result.get('f1_per_class', [0, 0, 0]),
                    'auc_macro':   result.get('auc_macro'),
                    'auc_weighted':result.get('auc_weighted'),
                    'logit_adj':   result.get('_logit_adj', {}),
                })
                auc_s = f", AUC={result.get('auc_macro', 0)*100:.2f}%" if result.get('auc_macro') else ""
                print(f"  ✓ seed={seed}: Acc={result.get('accuracy',0)*100:.2f}%, F1={result.get('f1_macro',0)*100:.2f}%{auc_s}")
            else:
                print(f"  ⚠ seed={seed}: Could not read results")
        else:
            print(f"  ✗ seed={seed}: FAILED")

    if results:
        generate_multi_seed_report(dataset, results)

    print(f"\n{'='*60}\n[Multi-Seed HKGAN] {success_count}/{len(MULTI_SEED_LIST)} succeeded\n{'='*60}\n")

    if success_count > 0:
        generate_hkgan_report(dataset)

    return success_count == len(MULTI_SEED_LIST)


def get_latest_experiment_result(results_dir):
    """讀取最新 improved 實驗目錄的 test_metrics（含 logit_adj）。"""
    if not results_dir.exists():
        return None
    exp_dirs = sorted(results_dir.glob("*_hkgan*"),
                      key=lambda x: x.stat().st_mtime, reverse=True)
    if not exp_dirs:
        return None
    result_file = exp_dirs[0] / "reports" / "experiment_results.json"
    if result_file.exists():
        try:
            data = json.load(open(result_file))
            metrics = data.get('test_metrics', {})
            metrics['_logit_adj'] = data.get('logit_adj_grid_search', {})
            return metrics
        except Exception:
            pass
    return None


def get_baseline_result(dataset):
    """讀取最新 baseline 實驗目錄的 test_metrics。"""
    baseline_dir = Path("results/baseline") / dataset
    if not baseline_dir.exists():
        return None
    exp_dirs = sorted(baseline_dir.glob("*_baseline_*"),
                      key=lambda x: x.stat().st_mtime, reverse=True)
    if not exp_dirs:
        return None
    result_file = exp_dirs[0] / "reports" / "experiment_results.json"
    if result_file.exists():
        try:
            return json.load(open(result_file)).get('test_metrics', {})
        except Exception:
            pass
    return None


def generate_multi_seed_report(dataset, results):
    """將 HKGAN 多種子結果寫成 txt 報告（含 Baseline 對比與穩定性評估）。"""
    reports_dir = Path("results")
    reports_dir.mkdir(parents=True, exist_ok=True)

    baseline = get_baseline_result(dataset)

    def mean_std(arr):
        return (float(np.mean(arr)), float(np.std(arr))) if arr else (0, 0)

    accuracies  = [r['accuracy']  * 100 for r in results]
    f1_macros   = [r['f1_macro']  * 100 for r in results]
    f1_neg      = [r['f1_per_class'][0] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_neu      = [r['f1_per_class'][1] * 100 for r in results if len(r['f1_per_class']) >= 3]
    f1_pos      = [r['f1_per_class'][2] * 100 for r in results if len(r['f1_per_class']) >= 3]
    auc_macros  = [r['auc_macro'] * 100 for r in results if r.get('auc_macro') is not None]

    acc_mean, acc_std = mean_std(accuracies)
    f1_mean,  f1_std  = mean_std(f1_macros)
    neg_mean, neg_std = mean_std(f1_neg)
    neu_mean, neu_std = mean_std(f1_neu)
    pos_mean, pos_std = mean_std(f1_pos)
    auc_mean, auc_std = mean_std(auc_macros)

    report = [
        "=" * 80,
        f"HKGAN 多種子實驗報告 - {get_display_name(dataset)} Dataset",
        "=" * 80,
        f"Seeds: {[r['seed'] for r in results]}  Runs: {len(results)}",
        "",
        # ── Baseline vs HKGAN 對比 ──
        "-" * 80,
        "Baseline vs HKGAN (Mean)",
        "-" * 80,
        f"{'Model':<20} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'AUC (%)':<12} {'Neg F1':<10} {'Neu F1':<10} {'Pos F1':<10}",
        "-" * 80,
    ]

    if baseline:
        b_acc = baseline.get('accuracy', 0) * 100
        b_f1  = baseline.get('f1_macro',  0) * 100
        b_auc = baseline.get('auc_macro',  0) * 100 if baseline.get('auc_macro') else 0
        b_cls = baseline.get('f1_per_class', [0, 0, 0])
        b_neg, b_neu, b_pos = (b_cls[i] * 100 if len(b_cls) > i else 0 for i in range(3))
        b_auc_s = f"{b_auc:<12.2f}" if b_auc > 0 else "N/A         "
        report.append(f"{'Baseline':<20} {b_acc:<12.2f} {b_f1:<15.2f} {b_auc_s} {b_neg:<10.2f} {b_neu:<10.2f} {b_pos:<10.2f}")
    else:
        report.append(f"{'Baseline':<20} {'N/A':<12} {'N/A':<15} {'N/A':<12} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        b_acc = b_f1 = b_auc = b_neg = b_neu = b_pos = 0

    auc_s = f"{auc_mean:<12.2f}" if auc_mean > 0 else "N/A         "
    report.append(f"{'HKGAN (Mean)':<20} {acc_mean:<12.2f} {f1_mean:<15.2f} {auc_s} {neg_mean:<10.2f} {neu_mean:<10.2f} {pos_mean:<10.2f}")
    report.append("-" * 80)

    # ── 改進分析 ──
    if baseline:
        def diff_str(v): return f"+{v:.2f}%" if v >= 0 else f"{v:.2f}%"
        report += [
            "",
            "改進分析 (HKGAN vs Baseline):",
            f"  Accuracy:  {diff_str(acc_mean - b_acc)}",
            f"  Macro-F1:  {diff_str(f1_mean  - b_f1)}",
        ]
        if auc_mean > 0 and b_auc > 0:
            report.append(f"  AUC:       {diff_str(auc_mean - b_auc)}")
        report += [
            f"  Neg F1:    {diff_str(neg_mean - b_neg)}",
            f"  Neu F1:    {diff_str(neu_mean - b_neu)}",
            f"  Pos F1:    {diff_str(pos_mean - b_pos)}",
            "",
            f"  {'✓' if f1_mean > b_f1 else '✗'} HKGAN {'超越' if f1_mean > b_f1 else '未能超越'} Baseline {abs(f1_mean - b_f1):.2f}% (Macro-F1)",
        ]

    # ── 各 seed 明細 ──
    report += [
        "",
        "-" * 80,
        "Individual Results (HKGAN)",
        "-" * 80,
        f"{'Seed':<10} {'Acc (%)':<12} {'Macro-F1 (%)':<15} {'AUC (%)':<12} {'Neg F1':<10} {'Neu F1':<10} {'Pos F1':<10}",
        "-" * 80,
    ]
    for r in results:
        fc    = r['f1_per_class']
        auc_s = f"{r['auc_macro']*100:<12.2f}" if r.get('auc_macro') else "N/A         "
        report.append(f"{r['seed']:<10} {r['accuracy']*100:<12.2f} {r['f1_macro']*100:<15.2f} {auc_s} {fc[0]*100:<10.2f} {fc[1]*100:<10.2f} {fc[2]*100:<10.2f}")

    # ── 彙總統計 ──
    report += [
        "-" * 80,
        "",
        "-" * 80,
        "Aggregated Statistics (Mean ± Std)",
        "-" * 80,
        f"  Accuracy:  {acc_mean:.2f}% ± {acc_std:.2f}%",
        f"  Macro-F1:  {f1_mean:.2f}% ± {f1_std:.2f}%",
    ]
    if auc_mean > 0:
        report.append(f"  AUC:       {auc_mean:.2f}% ± {auc_std:.2f}%")
    report += [
        f"  Neg F1:    {neg_mean:.2f}% ± {neg_std:.2f}%",
        f"  Neu F1:    {neu_mean:.2f}% ± {neu_std:.2f}%",
        f"  Pos F1:    {pos_mean:.2f}% ± {pos_std:.2f}%",
        "-" * 80,
    ]

    # ── 穩定性評估 ──
    if f1_std < 1.0:   stability = "Excellent (σ < 1%)"
    elif f1_std < 2.0: stability = "Good (1% ≤ σ < 2%)"
    elif f1_std < 3.0: stability = "Fair (2% ≤ σ < 3%)"
    else:              stability = "Poor (σ ≥ 3%)"

    report += [
        "",
        "-" * 80,
        "Robustness Assessment",
        "-" * 80,
        f"  Macro-F1 Stability: {stability}",
    ]
    if f1_mean > 0:
        report.append(f"  Coefficient of Variation: {f1_std/f1_mean*100:.2f}%")
    report.append("")

    # ── Logit Adjustment 統計 ──
    adj_records = [r.get('logit_adj', {}) for r in results if r.get('logit_adj')]
    if adj_records:
        from collections import Counter
        report += [
            "-" * 80,
            "Logit Adjustment (Val-Set Grid Search)",
            "-" * 80,
            f"{'Seed':<10} {'neutral_boost':<16} {'neg_suppress':<14} {'pos_suppress':<14} {'val F1':<10}",
            "-" * 80,
        ]
        for r in results:
            adj = r.get('logit_adj', {})
            if adj:
                report.append(f"{r['seed']:<10} {adj.get('neutral_boost',0):<16.1f} {adj.get('neg_suppress',0):<14.1f} {adj.get('pos_suppress',0):<14.1f} {adj.get('val_f1',0):<10.4f}")
        report.append("-" * 80)
        nb_c = Counter(a.get('neutral_boost', 0) for a in adj_records)
        ns_c = Counter(a.get('neg_suppress',  0) for a in adj_records)
        ps_c = Counter(a.get('pos_suppress',  0) for a in adj_records)
        report.append(f"  neutral_boost 分佈: { {f'{k:.1f}':v for k,v in sorted(nb_c.items())} }")
        report.append(f"  neg_suppress  分佈: { {f'{k:.1f}':v for k,v in sorted(ns_c.items())} }")
        report.append(f"  pos_suppress  分佈: { {f'{k:.1f}':v for k,v in sorted(ps_c.items())} }")
        report.append("")

    report.append("=" * 80)

    report_text = "\n".join(report)
    output_file = reports_dir / f"HKGAN_MultiSeed_{dataset}.txt"
    output_file.write_text(report_text, encoding='utf-8')
    print(f"\n{report_text}\n報告已保存至: {output_file}")


# ──────────────────────────────────────────────────────────────
# 統計顯著性檢驗
# ──────────────────────────────────────────────────────────────

def run_statistical_significance_test(datasets=None):
    """對各資料集執行 HKGAN vs Baseline 的 Paired t-test 與 Wilcoxon 檢驗。"""
    if datasets is None:
        datasets = ALL_DATASETS
    improved_dir = Path("results/improved")
    baseline_dir = Path("results/baseline")

    print(f"\n{'='*90}")
    print("Statistical Significance Test (Paired t-test)")
    print(f"Seeds: {MULTI_SEED_LIST}  |  α = 0.05")
    print(f"{'='*90}")

    all_results = []

    for dataset in datasets:
        print(f"\n{'-'*70}\nDataset: {get_display_name(dataset)}\n{'-'*70}")

        hkgan_f1_by_seed    = {}
        baseline_f1_by_seed = {}
        hkgan_acc_by_seed   = {}
        baseline_acc_by_seed= {}

        # 讀取 HKGAN 結果
        hkgan_dir = improved_dir / dataset
        if hkgan_dir.exists():
            for exp in hkgan_dir.glob("*_hkgan*"):
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    data = json.load(open(rf))
                    seed = data.get('args', {}).get('seed', 42)
                    m    = data.get('test_metrics', {})
                    if seed in MULTI_SEED_LIST:
                        hkgan_f1_by_seed[seed]  = m.get('f1_macro', 0) * 100
                        hkgan_acc_by_seed[seed] = m.get('accuracy', 0) * 100

        # 讀取 Baseline 結果
        bl_dir = baseline_dir / dataset
        if bl_dir.exists():
            for exp in bl_dir.glob("*_baseline_*"):
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    data = json.load(open(rf))
                    seed = data.get('args', {}).get('seed', 42)
                    m    = data.get('test_metrics', {})
                    if seed in MULTI_SEED_LIST:
                        baseline_f1_by_seed[seed]  = m.get('f1_macro', 0) * 100
                        baseline_acc_by_seed[seed] = m.get('accuracy', 0) * 100

        common_seeds = sorted(set(hkgan_f1_by_seed) & set(baseline_f1_by_seed))
        if len(common_seeds) < 2:
            print(f"  警告: 配對數據不足（需要至少 2 個共同 seed）")
            print(f"  HKGAN seeds: {sorted(hkgan_f1_by_seed)}  Baseline seeds: {sorted(baseline_f1_by_seed)}")
            continue

        hkgan_f1    = [hkgan_f1_by_seed[s]    for s in common_seeds]
        baseline_f1 = [baseline_f1_by_seed[s] for s in common_seeds]
        hkgan_acc   = [hkgan_acc_by_seed[s]   for s in common_seeds]
        baseline_acc= [baseline_acc_by_seed[s] for s in common_seeds]

        t_stat_f1,  p_value_f1  = stats.ttest_rel(hkgan_f1, baseline_f1)
        t_stat_acc, p_value_acc = stats.ttest_rel(hkgan_acc, baseline_acc)

        try:
            w_stat_f1, w_p_value_f1 = stats.wilcoxon(hkgan_f1, baseline_f1)
        except ValueError:
            w_stat_f1 = w_p_value_f1 = None

        diff_f1     = np.array(hkgan_f1) - np.array(baseline_f1)
        cohens_d_f1 = (np.mean(diff_f1) / np.std(diff_f1, ddof=1)
                       if np.std(diff_f1, ddof=1) > 0 else 0)

        print(f"\n  Paired Seeds: {common_seeds} (n={len(common_seeds)})")
        print(f"\n  {'Seed':<8} {'Baseline F1':<14} {'HKGAN F1':<14} {'Diff':<10}")
        print(f"  {'-'*46}")
        for s, bf, hf in zip(common_seeds, baseline_f1, hkgan_f1):
            print(f"  {s:<8} {bf:<14.2f} {hf:<14.2f} {hf-bf:+.2f}")
        print(f"  {'-'*46}")
        print(f"  {'Mean':<8} {np.mean(baseline_f1):<14.2f} {np.mean(hkgan_f1):<14.2f} {np.mean(diff_f1):+.2f}")

        print(f"\n  Macro-F1 Statistical Tests:")
        sig_f1 = '✓ Significant' if p_value_f1 < 0.05 else '✗ Not Significant'
        print(f"    Paired t-test:  t={t_stat_f1:.3f}, p={p_value_f1:.4f}  {sig_f1} (α=0.05)")
        if w_p_value_f1 is not None:
            sig_w = '✓ Significant' if w_p_value_f1 < 0.05 else '✗ Not Significant'
            print(f"    Wilcoxon test:  W={w_stat_f1:.1f}, p={w_p_value_f1:.4f}  {sig_w} (α=0.05)")
        effect = ("Large" if abs(cohens_d_f1) >= 0.8 else
                  "Medium" if abs(cohens_d_f1) >= 0.5 else
                  "Small"  if abs(cohens_d_f1) >= 0.2 else "Negligible")
        print(f"    Cohen's d:      {cohens_d_f1:.3f} ({effect} effect)")

        print(f"\n  Accuracy Statistical Tests:")
        sig_acc = '✓ Significant' if p_value_acc < 0.05 else '✗ Not Significant'
        print(f"    Paired t-test:  t={t_stat_acc:.3f}, p={p_value_acc:.4f}  {sig_acc} (α=0.05)")

        all_results.append({
            'dataset':          dataset,
            'n_pairs':          len(common_seeds),
            'baseline_f1_mean': np.mean(baseline_f1),
            'hkgan_f1_mean':    np.mean(hkgan_f1),
            'f1_improvement':   np.mean(diff_f1),
            't_stat_f1':        t_stat_f1,
            'p_value_f1':       p_value_f1,
            'cohens_d_f1':      cohens_d_f1,
            'significant_f1':   p_value_f1 < 0.05,
        })

    if all_results:
        generate_significance_report(all_results)


def generate_significance_report(all_results):
    """將統計顯著性結果寫成 txt 報告並印出。"""
    reports_dir = Path("results")
    reports_dir.mkdir(parents=True, exist_ok=True)

    report = [
        "=" * 100,
        "Statistical Significance Test Report - HKGAN vs Baseline",
        "=" * 100,
        "Test Method: Paired t-test (two-tailed)",
        f"Seeds: {MULTI_SEED_LIST}  |  Significance Level: α = 0.05",
        "",
        "-" * 100,
        f"{'Dataset':<15} {'n':<5} {'Baseline F1':<14} {'HKGAN F1':<14} {'Delta F1':<10} {'t-stat':<10} {'p-value':<12} {'Cohen d':<10} {'Significant':<12}",
        "-" * 100,
    ]

    significant_count = 0
    for r in all_results:
        if r['p_value_f1'] < 0.001:   sig_str = "Yes ***"
        elif r['p_value_f1'] < 0.01:  sig_str = "Yes **"
        elif r['p_value_f1'] < 0.05:  sig_str = "Yes *"
        else:                          sig_str = "No"
        if r['significant_f1']:
            significant_count += 1
        report.append(
            f"{get_display_name(r['dataset']):<15} {r['n_pairs']:<5} "
            f"{r['baseline_f1_mean']:<14.2f} {r['hkgan_f1_mean']:<14.2f} "
            f"{r['f1_improvement']:+<10.2f} {r['t_stat_f1']:<10.3f} "
            f"{r['p_value_f1']:<12.4f} {r['cohens_d_f1']:<10.3f} {sig_str:<12}"
        )

    report += [
        "-" * 100,
        "",
        "Legend:  *** p<0.001  ** p<0.01  * p<0.05",
        "Cohen's d:  |d|≥0.8 Large  |d|≥0.5 Medium  |d|≥0.2 Small  else Negligible",
        "",
        "-" * 100,
        f"Summary: {significant_count}/{len(all_results)} datasets show statistically significant improvement (p < 0.05)",
        "=" * 100,
    ]

    report_text = "\n".join(report)
    output_file = reports_dir / "Statistical_Significance_Report.txt"
    output_file.write_text(report_text, encoding='utf-8')
    print(f"\n\n{report_text}\n報告已保存至: {output_file}")


# ──────────────────────────────────────────────────────────────
# 論文全流程
# ──────────────────────────────────────────────────────────────

def _run_full_thesis(multi_seed: bool = False, auto_cleanup: bool = False) -> None:
    """依序執行：Stage 1 Baseline → Stage 2 HKGAN → Stage 3 消融。"""
    seed_tag = "Multi-Seed" if multi_seed else "Single-Seed"
    print(f"\n{'#'*80}")
    print(f"# 論文全流程 ({seed_tag})")
    print(f"# 資料集: {', '.join(ALL_DATASETS)}")
    if multi_seed:
        print(f"# Seeds : {MULTI_SEED_LIST}")
    print(f"{'#'*80}\n")

    stage_results = {}

    # Stage 1: Baseline
    print(f"\n{'='*80}\n[Stage 1/3] Baseline × {len(ALL_DATASETS)} 資料集\n{'='*80}")
    stage_results['baseline'] = {}
    for dataset in ALL_DATASETS:
        print(f"\n{'#'*60}\n# [Baseline] {get_display_name(dataset)}\n{'#'*60}")
        ok = run_baseline_only(dataset, multi_seed=multi_seed)
        stage_results['baseline'][dataset] = 'OK' if ok else 'FAIL'

    # Stage 2: HKGAN
    print(f"\n{'='*80}\n[Stage 2/3] HKGAN × {len(ALL_DATASETS)} 資料集\n{'='*80}")
    stage_results['hkgan'] = {}
    for dataset in ALL_DATASETS:
        print(f"\n{'#'*60}\n# [HKGAN] {get_display_name(dataset)}\n{'#'*60}")
        ok = run_hkgan_experiments(dataset, multi_seed=multi_seed)
        stage_results['hkgan'][dataset] = 'OK' if ok else 'FAIL'

    # Stage 3: 消融實驗（委派給 run_ablation.py）
    print(f"\n{'='*80}\n[Stage 3/3] 消融實驗 × {len(ALL_DATASETS)} 資料集 × 5 變體\n{'='*80}")
    ablation_cmd = [sys.executable, "run_ablation.py", "--full-study"]
    if multi_seed:
        ablation_cmd.append("--multi-seed")
    try:
        subprocess.run(ablation_cmd, check=True)
        stage_results['ablation'] = 'OK'
    except subprocess.CalledProcessError as e:
        print(f"\n[!] 消融實驗失敗 (code: {e.returncode})")
        stage_results['ablation'] = 'FAIL'
    except KeyboardInterrupt:
        print(f"\n[!] 消融實驗被中斷")
        stage_results['ablation'] = 'INTERRUPTED'

    # 全流程總結
    print(f"\n{'='*80}\n  論文全流程完成摘要\n{'='*80}")
    for stage, res in stage_results.items():
        if isinstance(res, dict):
            ok_count = sum(1 for v in res.values() if v == 'OK')
            print(f"  {stage:<12}: {ok_count}/{len(res)} 成功")
            for ds, status in res.items():
                print(f"    {'✓' if status=='OK' else '✗'} {get_display_name(ds)}")
        else:
            print(f"  {stage:<12}: {'✓' if res=='OK' else '✗'} {res}")
    print(f"{'='*80}\n")

    # Stage 4: 報告輸出（HKGAN 對比報告 + 統計顯著性 + 圖表）
    print(f"\n{'='*80}\n[Stage 4/4] 報告輸出\n{'='*80}")
    for dataset in ALL_DATASETS:
        if stage_results.get('hkgan', {}).get(dataset) == 'OK':
            generate_hkgan_report(dataset)
    run_statistical_significance_test()
    generate_thesis_figures()

    if auto_cleanup:
        _maybe_cleanup(True)


# ──────────────────────────────────────────────────────────────
# 清理工具
# ──────────────────────────────────────────────────────────────

def _backup_and_cleanup(execute: bool) -> None:
    """
    備份 results/ 下所有 .txt，再刪除整個 results/ 內容。

    dry-run（預設）：列出將備份的 txt 和將刪除的資料夾，不實際執行。
    --execute：備份後刪除。
    """
    project_root = Path(__file__).parent
    results_root = project_root / "results"
    date_str     = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path  = project_root / "backup" / date_str

    top_txt = sorted(results_root.glob("*.txt"))
    exp_dirs = [
        exp_dir
        for category in ["ablation", "baseline", "improved"]
        for dataset_dir in sorted((results_root / category).iterdir())
        if dataset_dir.is_dir()
        for exp_dir in sorted(dataset_dir.iterdir())
        if exp_dir.is_dir()
        if (results_root / category).exists()
    ]
    exp_txt  = [f for ed in exp_dirs for f in sorted(ed.rglob("*.txt"))]
    all_txt  = top_txt + exp_txt

    total_bytes = sum(
        f.stat().st_size
        for ed in exp_dirs for f in ed.rglob("*") if f.is_file()
    )

    def fmt(b):
        for unit in ("B", "KB", "MB", "GB"):
            if b < 1024: return f"{b:.1f} {unit}"
            b /= 1024
        return f"{b:.1f} TB"

    mode = "執行模式" if execute else "Dry-Run 模式"
    print(f"\n{'='*70}\n  結果備份與清理（{mode}）\n{'='*70}")
    print(f"\n  實驗資料夾：{len(exp_dirs)} 個   總大小：{fmt(total_bytes)}")
    print(f"  將備份 txt：{len(all_txt)} 個（頂層 {len(top_txt)} + 子資料夾 {len(exp_txt)}）")
    print(f"  備份目標：  {backup_path}")
    print(f"\n  備份清單（txt）：")
    for txt in all_txt:
        print(f"    {txt.relative_to(project_root)}")
    print(f"\n  刪除清單（results/ 全部內容）：")
    for item in sorted(results_root.iterdir()):
        size = sum(f.stat().st_size for f in item.rglob("*") if f.is_file()) if item.is_dir() else item.stat().st_size
        print(f"    {'(dir) ' if item.is_dir() else ''}{item.name}  [{fmt(size)}]")

    if not execute:
        print(f"\n  [!] Dry-Run，未執行任何操作。確認後加上 --execute 實際執行。\n{'='*70}\n")
        return

    print(f"\n  正在備份 txt...")
    ok = fail = 0
    for txt in all_txt:
        dest = backup_path / txt.relative_to(project_root)
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(txt, dest)
            ok += 1
        except Exception as e:
            print(f"    [錯誤] {txt.relative_to(project_root)}: {e}")
            fail += 1
    print(f"  備份完成：{ok} 成功，{fail} 失敗")

    print(f"\n  正在清空 results/...")
    ok = fail = 0
    for item in results_root.iterdir():
        try:
            shutil.rmtree(item) if item.is_dir() else item.unlink()
            ok += 1
        except Exception as e:
            print(f"    [錯誤] {item.name}: {e}")
            fail += 1
    print(f"  清空完成：{ok} 項刪除成功，{fail} 失敗")
    print(f"{'='*70}\n")


def _maybe_cleanup(auto_cleanup: bool) -> None:
    """實驗完成後觸發 checkpoint 清理（若啟用）。"""
    if not auto_cleanup:
        return
    print("\n[自動清理] 開始清理多餘 checkpoint...")
    print_cleanup_summary(run_cleanup(execute=True))


# ──────────────────────────────────────────────────────────────
# 主程式
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='批次執行實驗')
    parser.add_argument('--dataset', choices=ALL_DATASETS,
                        help='目標資料集（full-run 模式不需要指定）')
    parser.add_argument('--hkgan',            action='store_true', help='執行 HKGAN 實驗')
    parser.add_argument('--baseline',         action='store_true', help='執行 Baseline 實驗')
    parser.add_argument('--multi-seed',       action='store_true', help='多種子模式（seeds: 42,123,2023,999,0）')
    parser.add_argument('--full-run',         action='store_true', help='對所有資料集執行（搭配 --hkgan）')
    parser.add_argument('--full-baseline',    action='store_true', help='對所有資料集執行 Baseline')
    parser.add_argument('--full-thesis',      action='store_true', help='論文全流程：Baseline → HKGAN → 消融')
    parser.add_argument('--report-only',      action='store_true', help='只生成報告（HKGAN 對比 + 統計顯著性 + 圖表），不訓練')
    parser.add_argument('--auto-cleanup',     action='store_true', help='實驗後自動清理多餘 checkpoint')
    parser.add_argument('--cleanup-only',     action='store_true', help='獨立清理模式（預設 dry-run）')
    parser.add_argument('--execute',          action='store_true', help='搭配 --cleanup-only，實際執行刪除')
    args = parser.parse_args()

    # 各模式分派
    if args.cleanup_only:
        _backup_and_cleanup(execute=args.execute)
        return

    if args.report_only:
        # 永遠跑全部資料集，generate_hkgan_report / run_statistical_significance_test 內部自動跳過無資料者
        for ds in ALL_DATASETS:
            generate_hkgan_report(ds)
        run_statistical_significance_test()
        generate_thesis_figures()
        return

    if args.full_thesis:
        _run_full_thesis(multi_seed=args.multi_seed, auto_cleanup=args.auto_cleanup)
        return

    if args.full_baseline:
        mode_str = "Multi-Seed Baseline" if args.multi_seed else "Baseline"
        print(f"\n{'='*80}\n[Full {mode_str}] {len(ALL_DATASETS)} datasets\n{'='*80}\n")
        res = {}
        for ds in ALL_DATASETS:
            print(f"\n{'#'*80}\n# {get_display_name(ds)}\n{'#'*80}")
            res[ds] = 'Success' if run_baseline_only(ds, args.multi_seed) else 'Failed'
        print(f"\n{'='*80}\n[Full {mode_str} Summary]\n{'='*80}")
        for ds, status in res.items():
            print(f"  {get_display_name(ds):12s}: {status}")
        print(f"{'='*80}\n")
        _maybe_cleanup(args.auto_cleanup)
        return

    if args.full_run and args.hkgan:
        mode_str = "Multi-Seed HKGAN" if args.multi_seed else "HKGAN"
        print(f"\n{'='*80}\n[Full {mode_str}] {len(ALL_DATASETS)} datasets\n{'='*80}\n")
        res = {}
        for ds in ALL_DATASETS:
            print(f"\n{'#'*80}\n# {get_display_name(ds)}\n{'#'*80}")
            res[ds] = 'Success' if run_hkgan_experiments(ds, args.multi_seed) else 'Failed'
        print(f"\n{'='*80}\n[Full {mode_str} Summary]\n{'='*80}")
        for ds, status in res.items():
            print(f"  {get_display_name(ds):12s}: {status}")
        print(f"{'='*80}\n")
        _maybe_cleanup(args.auto_cleanup)
        return

    if args.dataset is None:
        parser.error("--dataset 為必要參數（除非使用 --full-run / --full-baseline / --full-thesis / --report-only）")

    if args.hkgan:
        run_hkgan_experiments(args.dataset, args.multi_seed)
    elif args.baseline:
        run_baseline_only(args.dataset, args.multi_seed)
    else:
        print("提示: 未指定模式，預設使用 HKGAN 模式。")
        run_hkgan_experiments(args.dataset, args.multi_seed)

    _maybe_cleanup(args.auto_cleanup)


if __name__ == "__main__":
    main()
