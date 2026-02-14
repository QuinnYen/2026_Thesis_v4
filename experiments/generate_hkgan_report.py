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
    """查找 Baseline 和 HKGAN 實驗"""
    experiments = {
        'baseline': None,
        'hkgan': None
    }

    # 查找 baseline 實驗
    baseline_dir = results_dir / "baseline" / dataset
    if baseline_dir.exists():
        for exp_dir in sorted(baseline_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not exp_dir.is_dir():
                continue
            dir_name = exp_dir.name
            if '_baseline_bert_cls_' in dir_name or '_baseline_bert_only_' in dir_name:
                experiments['baseline'] = exp_dir
                break

    # 查找 HKGAN 實驗
    improved_dir = results_dir / "improved" / dataset
    if improved_dir.exists():
        for exp_dir in sorted(improved_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if not exp_dir.is_dir():
                continue
            dir_name = exp_dir.name
            if '_improved_hkgan_' in dir_name:
                experiments['hkgan'] = exp_dir
                break

    return experiments


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
        'test_auc_macro': None,      # 新增：AUC (Macro)
        'test_auc_weighted': None,   # 新增：AUC (Weighted)
        'val_f1': None,
        'best_epoch': None,
        'total_epochs': None,
        'exp_name': exp_dir.name,
        'timestamp': None,
        'confusion_matrix': None  # 新增：混淆矩陣
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
        except Exception as e:
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
    report.append(f"{'Model':<20} {'Acc (%)':>12} {'Macro-F1 (%)':>14} {'AUC (%)':>12} {'Best Epoch':>12}")
    report.append("-" * 80)

    # Baseline
    if baseline_metrics and baseline_metrics.get('test_acc'):
        acc = f"{baseline_metrics['test_acc']*100:.2f}"
        f1 = f"{baseline_metrics['test_f1']*100:.2f}" if baseline_metrics.get('test_f1') else "N/A"
        auc = f"{baseline_metrics['test_auc_macro']*100:.2f}" if baseline_metrics.get('test_auc_macro') else "N/A"
        epoch = f"{baseline_metrics.get('best_epoch', 'N/A')}"
        report.append(f"{'Baseline (BERT-CLS)':<20} {acc:>12} {f1:>14} {auc:>12} {epoch:>12}")
    else:
        report.append(f"{'Baseline (BERT-CLS)':<20} {'N/A':>12} {'N/A':>14} {'N/A':>12} {'N/A':>12}")

    # HKGAN
    if hkgan_metrics and hkgan_metrics.get('test_acc'):
        acc = f"{hkgan_metrics['test_acc']*100:.2f}"
        f1 = f"{hkgan_metrics['test_f1']*100:.2f}" if hkgan_metrics.get('test_f1') else "N/A"
        auc = f"{hkgan_metrics['test_auc_macro']*100:.2f}" if hkgan_metrics.get('test_auc_macro') else "N/A"
        epoch = f"{hkgan_metrics.get('best_epoch', 'N/A')}"
        report.append(f"{'HKGAN (Ours)':<20} {acc:>12} {f1:>14} {auc:>12} {epoch:>12}")
    else:
        report.append(f"{'HKGAN (Ours)':<20} {'N/A':>12} {'N/A':>14} {'N/A':>12} {'N/A':>12}")

    report.append("-" * 80)
    report.append("")

    # Per-class F1 分析
    report.append("-" * 80)
    report.append("Per-class F1 Analysis")
    report.append("-" * 80)
    report.append(f"{'Model':<20} {'Neg F1 (%)':>12} {'Neu F1 (%)':>12} {'Pos F1 (%)':>12}")
    report.append("-" * 80)

    if baseline_metrics and baseline_metrics.get('test_f1_neg'):
        neg = f"{baseline_metrics['test_f1_neg']*100:.2f}"
        neu = f"{baseline_metrics['test_f1_neu']*100:.2f}" if baseline_metrics.get('test_f1_neu') else "N/A"
        pos = f"{baseline_metrics['test_f1_pos']*100:.2f}" if baseline_metrics.get('test_f1_pos') else "N/A"
        report.append(f"{'Baseline':<20} {neg:>12} {neu:>12} {pos:>12}")
    else:
        report.append(f"{'Baseline':<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    if hkgan_metrics and hkgan_metrics.get('test_f1_neg'):
        neg = f"{hkgan_metrics['test_f1_neg']*100:.2f}"
        neu = f"{hkgan_metrics['test_f1_neu']*100:.2f}" if hkgan_metrics.get('test_f1_neu') else "N/A"
        pos = f"{hkgan_metrics['test_f1_pos']*100:.2f}" if hkgan_metrics.get('test_f1_pos') else "N/A"
        report.append(f"{'HKGAN':<20} {neg:>12} {neu:>12} {pos:>12}")
    else:
        report.append(f"{'HKGAN':<20} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    report.append("-" * 80)
    report.append("")

    # 混淆矩陣
    report.append("-" * 80)
    report.append("混淆矩陣 (Confusion Matrix)")
    report.append("-" * 80)
    report.append("格式: 行=實際標籤, 列=預測標籤")
    report.append("      [Negative]  [Neutral]  [Positive]")
    report.append("")

    if baseline_metrics and baseline_metrics.get('confusion_matrix'):
        cm = baseline_metrics['confusion_matrix']
        report.append("Baseline:")
        report.append(f"  Neg    {cm[0][0]:>6}     {cm[0][1]:>6}     {cm[0][2]:>6}")
        report.append(f"  Neu    {cm[1][0]:>6}     {cm[1][1]:>6}     {cm[1][2]:>6}")
        report.append(f"  Pos    {cm[2][0]:>6}     {cm[2][1]:>6}     {cm[2][2]:>6}")
        # 計算並顯示各類別的精確率和召回率
        total_neg = sum(cm[0])
        total_neu = sum(cm[1])
        total_pos = sum(cm[2])
        pred_neg = cm[0][0] + cm[1][0] + cm[2][0]
        pred_neu = cm[0][1] + cm[1][1] + cm[2][1]
        pred_pos = cm[0][2] + cm[1][2] + cm[2][2]
        report.append(f"  Recall:  Neg={cm[0][0]/total_neg*100:.1f}%  Neu={cm[1][1]/total_neu*100:.1f}%  Pos={cm[2][2]/total_pos*100:.1f}%")
        if pred_neg > 0 and pred_neu > 0 and pred_pos > 0:
            report.append(f"  Precision: Neg={cm[0][0]/pred_neg*100:.1f}%  Neu={cm[1][1]/pred_neu*100:.1f}%  Pos={cm[2][2]/pred_pos*100:.1f}%")
        report.append("")

    if hkgan_metrics and hkgan_metrics.get('confusion_matrix'):
        cm = hkgan_metrics['confusion_matrix']
        report.append("HKGAN:")
        report.append(f"  Neg    {cm[0][0]:>6}     {cm[0][1]:>6}     {cm[0][2]:>6}")
        report.append(f"  Neu    {cm[1][0]:>6}     {cm[1][1]:>6}     {cm[1][2]:>6}")
        report.append(f"  Pos    {cm[2][0]:>6}     {cm[2][1]:>6}     {cm[2][2]:>6}")
        # 計算並顯示各類別的精確率和召回率
        total_neg = sum(cm[0])
        total_neu = sum(cm[1])
        total_pos = sum(cm[2])
        pred_neg = cm[0][0] + cm[1][0] + cm[2][0]
        pred_neu = cm[0][1] + cm[1][1] + cm[2][1]
        pred_pos = cm[0][2] + cm[1][2] + cm[2][2]
        report.append(f"  Recall:  Neg={cm[0][0]/total_neg*100:.1f}%  Neu={cm[1][1]/total_neu*100:.1f}%  Pos={cm[2][2]/total_pos*100:.1f}%")
        if pred_neg > 0 and pred_neu > 0 and pred_pos > 0:
            report.append(f"  Precision: Neg={cm[0][0]/pred_neg*100:.1f}%  Neu={cm[1][1]/pred_neu*100:.1f}%  Pos={cm[2][2]/pred_pos*100:.1f}%")
        report.append("")

    # 混淆矩陣對比分析
    if baseline_metrics and hkgan_metrics and baseline_metrics.get('confusion_matrix') and hkgan_metrics.get('confusion_matrix'):
        cm_base = baseline_metrics['confusion_matrix']
        cm_hkgan = hkgan_metrics['confusion_matrix']
        report.append("混淆矩陣變化 (HKGAN - Baseline):")
        for i, label in enumerate(['Neg', 'Neu', 'Pos']):
            diff = [cm_hkgan[i][j] - cm_base[i][j] for j in range(3)]
            diff_str = [f"{d:+d}" for d in diff]
            report.append(f"  {label}    {diff_str[0]:>6}     {diff_str[1]:>6}     {diff_str[2]:>6}")
        report.append("")

    report.append("-" * 80)
    report.append("")

    # 改進分析
    if baseline_metrics and hkgan_metrics and baseline_metrics.get('test_f1') and hkgan_metrics.get('test_f1'):
        report.append("-" * 80)
        report.append("改進分析 (HKGAN vs Baseline)")
        report.append("-" * 80)

        acc_diff = (hkgan_metrics['test_acc'] - baseline_metrics['test_acc']) * 100
        f1_diff = (hkgan_metrics['test_f1'] - baseline_metrics['test_f1']) * 100

        acc_symbol = "↑" if acc_diff > 0 else "↓" if acc_diff < 0 else "="
        f1_symbol = "↑" if f1_diff > 0 else "↓" if f1_diff < 0 else "="

        report.append(f"  Accuracy:  {acc_symbol} {abs(acc_diff):+.2f}%")
        report.append(f"  Macro-F1:  {f1_symbol} {abs(f1_diff):+.2f}%")

        # AUC improvement
        if baseline_metrics.get('test_auc_macro') and hkgan_metrics.get('test_auc_macro'):
            auc_diff = (hkgan_metrics['test_auc_macro'] - baseline_metrics['test_auc_macro']) * 100
            auc_symbol = "↑" if auc_diff > 0 else "↓" if auc_diff < 0 else "="
            report.append(f"  AUC:       {auc_symbol} {abs(auc_diff):+.2f}%")

        # Per-class improvement
        if baseline_metrics.get('test_f1_neg') and hkgan_metrics.get('test_f1_neg'):
            neg_diff = (hkgan_metrics['test_f1_neg'] - baseline_metrics['test_f1_neg']) * 100
            neu_diff = (hkgan_metrics['test_f1_neu'] - baseline_metrics['test_f1_neu']) * 100
            pos_diff = (hkgan_metrics['test_f1_pos'] - baseline_metrics['test_f1_pos']) * 100

            report.append(f"  Neg F1:    {'+' if neg_diff >= 0 else ''}{neg_diff:.2f}%")
            report.append(f"  Neu F1:    {'+' if neu_diff >= 0 else ''}{neu_diff:.2f}%")
            report.append(f"  Pos F1:    {'+' if pos_diff >= 0 else ''}{pos_diff:.2f}%")

        report.append("")

        # 結論
        if f1_diff > 0:
            report.append(f"  ✓ HKGAN 在 {display_name} 數據集上超越 Baseline {f1_diff:.2f}% (Macro-F1)")
        elif f1_diff < 0:
            report.append(f"  ✗ HKGAN 在 {display_name} 數據集上低於 Baseline {abs(f1_diff):.2f}% (Macro-F1)")
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
    reports_dir = Path("results")

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
        baseline_metrics = read_metrics(experiments['baseline'])
        hkgan_metrics = read_metrics(experiments['hkgan'])

        if baseline_metrics is None and hkgan_metrics is None:
            print(f"  ✗ 未找到任何實驗結果")
            continue

        report = generate_report(dataset, baseline_metrics, hkgan_metrics)

        # 輸出到控制台
        print(report)

        # 保存報告
        output_file = reports_dir / f"HKGAN報告_{display_name}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n報告已保存至: {output_file}")


if __name__ == "__main__":
    main()
