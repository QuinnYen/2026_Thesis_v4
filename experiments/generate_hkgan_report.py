"""
HKGAN vs Baseline 對比報告生成器

專門比較 HKGAN 和 Baseline 的實驗結果

使用方法:
    python experiments/generate_hkgan_report.py --dataset restaurants
    python experiments/generate_hkgan_report.py --dataset laptops
    python experiments/generate_hkgan_report.py --dataset mams
    python experiments/generate_hkgan_report.py --all
"""

import json
import argparse
from pathlib import Path
from datetime import datetime

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


def find_experiments(results_dir, dataset):
    """查找 Baseline 和 HKGAN 實驗（收集所有 seed 目錄）"""
    experiments = {
        'baseline': None,   # 最新單筆（備用）
        'hkgan': None,      # 最新單筆（備用）
        'baseline_dirs': [],
        'hkgan_dirs': [],
    }

    MULTI_SEED_LIST = [42, 123, 2023, 999, 0]

    # 查找 baseline 實驗（全部 seed）
    baseline_dir = results_dir / "baseline" / dataset
    if baseline_dir.exists():
        for exp_dir in sorted(baseline_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not exp_dir.is_dir():
                continue
            dir_name = exp_dir.name
            if '_baseline_bert_cls' in dir_name or '_baseline_bert_only' in dir_name or '_ablation_bert_only' in dir_name:
                rf = exp_dir / "reports" / "experiment_results.json"
                if rf.exists():
                    try:
                        data = json.load(open(rf))
                        seed = data.get('args', {}).get('seed', -1)
                        if seed in MULTI_SEED_LIST:
                            experiments['baseline_dirs'].append(exp_dir)
                    except Exception:
                        pass
                    if experiments['baseline'] is None:
                        experiments['baseline'] = exp_dir

    # 查找 HKGAN 實驗（全部 seed）
    improved_dir = results_dir / "improved" / dataset
    if improved_dir.exists():
        for exp_dir in sorted(improved_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not exp_dir.is_dir():
                continue
            dir_name = exp_dir.name
            if '_hkgan' in dir_name:
                rf = exp_dir / "reports" / "experiment_results.json"
                if rf.exists():
                    try:
                        data = json.load(open(rf))
                        seed = data.get('args', {}).get('seed', -1)
                        if seed in MULTI_SEED_LIST:
                            experiments['hkgan_dirs'].append(exp_dir)
                    except Exception:
                        pass
                if experiments['hkgan'] is None:
                    experiments['hkgan'] = exp_dir

    return experiments


def read_metrics_multiseed(exp_dirs):
    """從多個 seed 目錄讀取並聚合指標（平均值）。"""
    import numpy as np
    if not exp_dirs:
        return None

    accs, f1s, f1_negs, f1_neus, f1_poss, aucs = [], [], [], [], [], []
    best_epoch_list = []
    confusion_sum = None
    sample = None

    for exp_dir in exp_dirs:
        rf = exp_dir / "reports" / "experiment_results.json"
        if not rf.exists():
            continue
        try:
            data = json.load(open(rf))
            tm = data.get('test_metrics', {})
            acc = tm.get('accuracy')
            f1  = tm.get('f1_macro')
            fpc = tm.get('f1_per_class', [])
            auc = tm.get('auc_macro')
            cm  = tm.get('confusion_matrix')

            if acc is not None: accs.append(acc)
            if f1  is not None: f1s.append(f1)
            if len(fpc) >= 3:
                f1_negs.append(fpc[0]); f1_neus.append(fpc[1]); f1_poss.append(fpc[2])
            if auc is not None: aucs.append(auc)

            # 混淆矩陣加總
            if cm is not None:
                cm_arr = np.array(cm)
                confusion_sum = cm_arr if confusion_sum is None else confusion_sum + cm_arr

            # best_epoch（從 history 推算）
            val_f1_list = data.get('history', {}).get('val_f1_macro', [])
            best_val = data.get('best_val_f1')
            if val_f1_list and best_val is not None:
                for i, v in enumerate(val_f1_list):
                    if abs(v - best_val) < 1e-6:
                        best_epoch_list.append(i + 1)
                        break

            if sample is None:
                sample = exp_dir
        except Exception:
            pass

    if not f1s:
        return None

    def mean(lst): return float(np.mean(lst)) if lst else None

    metrics = {
        'test_acc':         mean(accs),
        'test_f1':          mean(f1s),
        'test_f1_std':      float(np.std(f1s)) if len(f1s) > 1 else 0.0,
        'test_f1_neg':      mean(f1_negs),
        'test_f1_neu':      mean(f1_neus),
        'test_f1_pos':      mean(f1_poss),
        'test_auc_macro':   mean(aucs),
        'test_auc_weighted':None,
        'val_f1':           None,
        'best_epoch':       int(round(np.mean(best_epoch_list))) if best_epoch_list else None,
        'n_seeds':          len(f1s),
        'confusion_matrix': confusion_sum.tolist() if confusion_sum is not None else None,
        'exp_name':         sample.name if sample else '',
        'timestamp':        None,
    }
    return metrics


def read_metrics(exp_dir):
    """從實驗目錄讀取指標"""
    if exp_dir is None:
        return None

    metrics = {
        'test_acc': None,
        'test_f1': None,
        'test_f1_neg': None,
        'test_f1_neu': None,
        'test_f1_pos': None,
        'test_auc_macro': None,
        'test_auc_weighted': None,
        'val_f1': None,
        'best_epoch': None,
        'total_epochs': None,
        'exp_name': exp_dir.name,
        'timestamp': None,
        'confusion_matrix': None
    }

    # 從 experiment_config.json 讀取訓練配置
    config_file = exp_dir / "reports" / "experiment_config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                training_config = config.get('training', {})
                metrics['total_epochs'] = training_config.get('epochs')
                metrics['lr'] = training_config.get('lr')
                metrics['batch_size'] = training_config.get('batch_size')
                metrics['focal_gamma'] = training_config.get('focal_gamma')
        except Exception:
            pass

    # 從 experiment_results.json 讀取
    results_file = exp_dir / "reports" / "experiment_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                test_metrics = data.get('test_metrics', {})
                metrics['test_acc'] = test_metrics.get('accuracy')
                metrics['test_f1'] = test_metrics.get('f1_macro')

                f1_per_class = test_metrics.get('f1_per_class', [])
                if len(f1_per_class) >= 3:
                    metrics['test_f1_neg'] = f1_per_class[0]
                    metrics['test_f1_neu'] = f1_per_class[1]
                    metrics['test_f1_pos'] = f1_per_class[2]

                # 讀取 AUC
                metrics['test_auc_macro'] = test_metrics.get('auc_macro')
                metrics['test_auc_weighted'] = test_metrics.get('auc_weighted')

                # 讀取混淆矩陣
                metrics['confusion_matrix'] = test_metrics.get('confusion_matrix')

                metrics['val_f1'] = data.get('best_val_f1')

                if 'best_val_f1' in data and 'history' in data:
                    val_f1_list = data['history'].get('val_f1_macro', [])
                    if val_f1_list:
                        best_f1 = data['best_val_f1']
                        for i, f1 in enumerate(val_f1_list):
                            if abs(f1 - best_f1) < 1e-6:
                                metrics['best_epoch'] = i + 1
                                break
        except Exception as e:
            print(f"  讀取錯誤: {e}")

    # 獲取時間戳
    dir_name = exp_dir.name
    try:
        timestamp_str = dir_name.split('_')[0] + '_' + dir_name.split('_')[1]
        metrics['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except:
        metrics['timestamp'] = datetime.fromtimestamp(exp_dir.stat().st_mtime)

    return metrics


def _fmt_cm_row(cm_row):
    """將混淆矩陣一列格式化，支援整數或浮點數。"""
    def fmt(v):
        return f"{v:.1f}" if isinstance(v, float) else f"{int(v)}"
    return [fmt(v) for v in cm_row]


def _cm_stats(cm):
    """回傳 (recall_list, precision_list)，各含 3 個百分比值。"""
    totals = [sum(cm[i]) for i in range(3)]
    pred_totals = [sum(cm[i][j] for i in range(3)) for j in range(3)]
    recall    = [cm[i][i] / totals[i] * 100 if totals[i] > 0 else 0.0 for i in range(3)]
    precision = [cm[i][i] / pred_totals[i] * 100 if pred_totals[i] > 0 else 0.0 for i in range(3)]
    return recall, precision


def _append_cm_block(report, label, cm, n_seeds):
    """輸出單個模型的混淆矩陣區塊。"""
    seed_note = f" ({n_seeds}-seed 累積)" if n_seeds and n_seeds > 1 else ""
    report.append(f"{label}{seed_note}:")
    header = f"  {'':10}  {'Pred Neg':>10}  {'Pred Neu':>10}  {'Pred Pos':>10}"
    report.append(header)
    report.append(f"  {'-'*43}")
    labels = ['Neg', 'Neu', 'Pos']
    col_w = 10
    for i, row_label in enumerate(labels):
        row = _fmt_cm_row(cm[i])
        report.append(f"  {'True '+row_label:<10}  {row[0]:>{col_w}}  {row[1]:>{col_w}}  {row[2]:>{col_w}}")
    recall, precision = _cm_stats(cm)
    report.append(f"  {'-'*43}")
    report.append(f"  {'Recall':<10}  {recall[0]:>9.1f}%  {recall[1]:>9.1f}%  {recall[2]:>9.1f}%")
    report.append(f"  {'Precision':<10}  {precision[0]:>9.1f}%  {precision[1]:>9.1f}%  {precision[2]:>9.1f}%")
    report.append("")


def generate_report(dataset, baseline_metrics, hkgan_metrics):
    """生成對比報告"""
    display_name = get_display_name(dataset)
    report = []
    report.append("=" * 80)
    report.append(f"HKGAN vs Baseline 對比報告 - {display_name} Dataset")
    report.append("=" * 80)
    report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 模型說明
    report.append("-" * 80)
    report.append("模型架構")
    report.append("-" * 80)
    report.append("  Baseline:  BERT-CLS")
    report.append("             標準 BERT baseline，使用 [CLS] token 分類")
    report.append("")
    report.append("  HKGAN:     Hierarchical Knowledge-enhanced Graph Attention Network")
    report.append("             - 階層式 BERT 特徵 (Low/Mid/High layers)")
    report.append("             - SenticNet 情感知識注入")
    report.append("             - 階層式 GAT (窗口 3/5/全連接)")
    report.append("             - Inter-Aspect Attention")
    report.append("")

    # 主要結果
    report.append("-" * 80)
    report.append("實驗結果 (Main Results)")
    report.append("-" * 80)
    report.append(f"{'Model':<20} {'Acc (%)':>12} {'Macro-F1 (%)':>18} {'AUC (%)':>12} {'Best Epoch':>12}")
    report.append("-" * 80)

    def fmt_f1(m):
        if m is None or m.get('test_f1') is None:
            return "N/A"
        v = m['test_f1'] * 100
        std = m.get('test_f1_std', 0)
        n = m.get('n_seeds', 1)
        return f"{v:.2f}±{std*100:.2f}" if n > 1 else f"{v:.2f}"

    def fmt_row(label, m):
        if m and m.get('test_acc') is not None:
            n = m.get('n_seeds', 1)
            n_tag = f" ({n}-seed mean)" if n > 1 else ""
            acc   = f"{m['test_acc']*100:.2f}"
            f1    = fmt_f1(m)
            auc   = f"{m['test_auc_macro']*100:.2f}" if m.get('test_auc_macro') else "N/A"
            epoch = str(m.get('best_epoch', 'N/A'))
            report.append(f"{label:<20} {acc:>12} {f1:>18} {auc:>12} {epoch:>12}{n_tag}")
        else:
            report.append(f"{label:<20} {'N/A':>12} {'N/A':>18} {'N/A':>12} {'N/A':>12}")

    fmt_row("Baseline (BERT-CLS)", baseline_metrics)
    fmt_row("HKGAN (Ours)",        hkgan_metrics)
    report.append("-" * 80)
    report.append("")

    # Per-class F1 分析
    report.append("-" * 80)
    report.append("Per-class F1 Analysis")
    report.append("-" * 80)
    report.append(f"{'Model':<20} {'Neg F1 (%)':>12} {'Neu F1 (%)':>12} {'Pos F1 (%)':>12}")
    report.append("-" * 80)

    def fmt_per_class(label, m):
        if m and m.get('test_f1_neg') is not None:
            neg = f"{m['test_f1_neg']*100:.2f}"
            neu = f"{m['test_f1_neu']*100:.2f}" if m.get('test_f1_neu') is not None else "N/A"
            pos = f"{m['test_f1_pos']*100:.2f}" if m.get('test_f1_pos') is not None else "N/A"
        else:
            neg = neu = pos = "N/A"
        report.append(f"{label:<20} {neg:>12} {neu:>12} {pos:>12}")

    fmt_per_class("Baseline", baseline_metrics)
    fmt_per_class("HKGAN",    hkgan_metrics)
    report.append("-" * 80)
    report.append("")

    # 混淆矩陣
    report.append("-" * 80)
    report.append("混淆矩陣 (Confusion Matrix)")
    report.append("-" * 80)
    report.append("格式: 行=實際標籤 (True), 列=預測標籤 (Pred)")
    report.append("")

    has_base_cm  = baseline_metrics and baseline_metrics.get('confusion_matrix')
    has_hkgan_cm = hkgan_metrics    and hkgan_metrics.get('confusion_matrix')

    if has_base_cm:
        _append_cm_block(report, "Baseline", baseline_metrics['confusion_matrix'],
                         baseline_metrics.get('n_seeds', 1))
    if has_hkgan_cm:
        _append_cm_block(report, "HKGAN", hkgan_metrics['confusion_matrix'],
                         hkgan_metrics.get('n_seeds', 1))

    # 差異矩陣（僅當兩者都有且 seed 數相同時才有意義）
    if has_base_cm and has_hkgan_cm:
        cm_b = baseline_metrics['confusion_matrix']
        cm_h = hkgan_metrics['confusion_matrix']
        report.append("差異矩陣 (HKGAN - Baseline):")
        report.append(f"  {'':10}  {'Pred Neg':>10}  {'Pred Neu':>10}  {'Pred Pos':>10}")
        report.append(f"  {'-'*43}")
        for i, row_label in enumerate(['Neg', 'Neu', 'Pos']):
            diff = [cm_h[i][j] - cm_b[i][j] for j in range(3)]
            diff_s = [f"{d:+.0f}" for d in diff]
            report.append(f"  {'True '+row_label:<10}  {diff_s[0]:>10}  {diff_s[1]:>10}  {diff_s[2]:>10}")
        report.append("")

    report.append("-" * 80)
    report.append("")

    # 改進分析
    if baseline_metrics and hkgan_metrics and baseline_metrics.get('test_f1') and hkgan_metrics.get('test_f1'):
        report.append("-" * 80)
        report.append("改進分析 (HKGAN vs Baseline)")
        report.append("-" * 80)

        def diff_line(label, diff_val):
            sym = "↑" if diff_val > 0 else ("↓" if diff_val < 0 else "=")
            return f"  {label:<12} {sym} {diff_val:+.2f}%"

        acc_diff = (hkgan_metrics['test_acc'] - baseline_metrics['test_acc']) * 100
        f1_diff  = (hkgan_metrics['test_f1']  - baseline_metrics['test_f1'])  * 100
        report.append(diff_line("Accuracy:", acc_diff))
        report.append(diff_line("Macro-F1:", f1_diff))

        if baseline_metrics.get('test_auc_macro') and hkgan_metrics.get('test_auc_macro'):
            auc_diff = (hkgan_metrics['test_auc_macro'] - baseline_metrics['test_auc_macro']) * 100
            report.append(diff_line("AUC:", auc_diff))

        if baseline_metrics.get('test_f1_neg') is not None and hkgan_metrics.get('test_f1_neg') is not None:
            report.append(diff_line("Neg F1:", (hkgan_metrics['test_f1_neg'] - baseline_metrics['test_f1_neg']) * 100))
            report.append(diff_line("Neu F1:", (hkgan_metrics['test_f1_neu'] - baseline_metrics['test_f1_neu']) * 100))
            report.append(diff_line("Pos F1:", (hkgan_metrics['test_f1_pos'] - baseline_metrics['test_f1_pos']) * 100))

        report.append("")
        if f1_diff > 0:
            report.append(f"  ✓ HKGAN 在 {display_name} 數據集上超越 Baseline {f1_diff:+.2f}% (Macro-F1)")
        elif f1_diff < 0:
            report.append(f"  ✗ HKGAN 在 {display_name} 數據集上低於 Baseline {f1_diff:+.2f}% (Macro-F1)")
        else:
            report.append(f"  = HKGAN 與 Baseline 在 {display_name} 數據集上持平")

    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='生成 HKGAN vs Baseline 對比報告')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['restaurants', 'mams', 'laptops', 'rest16', 'lap16'],
                        help='數據集選擇')
    parser.add_argument('--all', action='store_true',
                        help='對所有數據集生成報告')

    args = parser.parse_args()

    results_dir = Path("results")

    if args.all:
        datasets = ['restaurants', 'laptops', 'rest16', 'lap16', 'mams']
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("請指定 --dataset 或 --all")

    for dataset in datasets:
        display_name = get_display_name(dataset)
        print(f"\n處理 {display_name} 數據集...")

        experiments = find_experiments(results_dir, dataset)

        # 優先用多 seed 聚合；若不足則 fallback 單筆
        if experiments['baseline_dirs']:
            baseline_metrics = read_metrics_multiseed(experiments['baseline_dirs'])
        else:
            baseline_metrics = read_metrics(experiments['baseline'])

        if experiments['hkgan_dirs']:
            hkgan_metrics = read_metrics_multiseed(experiments['hkgan_dirs'])
        else:
            hkgan_metrics = read_metrics(experiments['hkgan'])

        if baseline_metrics is None and hkgan_metrics is None:
            print(f"  ✗ 未找到任何實驗結果")
            continue

        report = generate_report(dataset, baseline_metrics, hkgan_metrics)

        # 輸出到控制台
        print(report)

        # 保存報告
        output_file = results_dir / f"HKGAN報告_{display_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n報告已保存至: {output_file}")


if __name__ == "__main__":
    main()
