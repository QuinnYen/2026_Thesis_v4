"""
生成 Baseline 比較報告

從 results/baseline/{dataset}/ 目錄讀取 baseline 實驗結果，生成綜合報告 txt

使用方法:
    python experiments/generate_baseline_report.py --dataset restaurants
    python experiments/generate_baseline_report.py --dataset laptops
    python experiments/generate_baseline_report.py --dataset mams
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def find_all_baselines(baseline_dataset_dir):
    """查找該數據集下所有 baseline 實驗"""
    baselines = {}

    # 查找所有實驗資料夾
    for exp_dir in baseline_dataset_dir.iterdir():
        if not exp_dir.is_dir():
            continue

        # 從資料夾名稱判斷 baseline 類型
        dir_name = exp_dir.name
        if '_baseline_bert_only_' in dir_name:
            baseline_type = 'bert_only'
        elif '_baseline_bert_aaha_' in dir_name:
            baseline_type = 'bert_aaha'
        elif '_baseline_bert_mean_' in dir_name:
            baseline_type = 'bert_mean'
        else:
            continue

        # 保存最新的實驗（如果有多個同類型）
        if baseline_type not in baselines:
            baselines[baseline_type] = exp_dir
        else:
            # 比較時間戳，保留最新的
            if exp_dir.stat().st_mtime > baselines[baseline_type].stat().st_mtime:
                baselines[baseline_type] = exp_dir

    return baselines


def read_metrics(exp_dir):
    """從實驗目錄讀取指標"""
    metrics = {
        'test_acc': None,
        'test_f1': None,
        'test_f1_neg': None,
        'test_f1_neu': None,
        'test_f1_pos': None,
        'val_f1': None,
        'best_epoch': None,
        'timestamp': None,
        'exp_name': exp_dir.name
    }

    # 從 experiment_results.json 讀取
    results_file = exp_dir / "reports" / "experiment_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

                # 讀取 test_metrics
                test_metrics = data.get('test_metrics', {})
                metrics['test_acc'] = test_metrics.get('accuracy')
                metrics['test_f1'] = test_metrics.get('f1_macro')

                # 讀取各類別的 F1 值
                f1_per_class = test_metrics.get('f1_per_class', [])
                if len(f1_per_class) >= 3:
                    metrics['test_f1_neg'] = f1_per_class[0]
                    metrics['test_f1_neu'] = f1_per_class[1]
                    metrics['test_f1_pos'] = f1_per_class[2]

                # 讀取 validation F1
                metrics['val_f1'] = data.get('best_val_f1')

                # 讀取 best epoch
                if 'best_val_f1' in data and 'history' in data:
                    val_f1_list = data['history'].get('val_f1_macro', [])
                    if val_f1_list:
                        best_f1 = data['best_val_f1']
                        for i, f1 in enumerate(val_f1_list):
                            if abs(f1 - best_f1) < 1e-6:
                                metrics['best_epoch'] = i + 1
                                break
        except Exception as e:
            print(f"    讀取錯誤: {e}")

    # 獲取時間戳
    dir_name = exp_dir.name
    try:
        timestamp_str = dir_name.split('_')[0] + '_' + dir_name.split('_')[1]
        metrics['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except:
        metrics['timestamp'] = datetime.fromtimestamp(exp_dir.stat().st_mtime)

    return metrics


def generate_text_report(dataset, results):
    """生成純文字報告"""
    report = []
    report.append("=" * 80)
    report.append(f"BASELINE 實驗綜合報告 - {dataset.upper()} 數據集")
    report.append("=" * 80)
    report.append(f"\n生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 實驗配置
    report.append("-" * 80)
    report.append("實驗配置")
    report.append("-" * 80)
    report.append("  Epochs:        30")
    report.append("  Learning Rate: 2e-5")
    report.append("  Dropout:       0.3")
    report.append("  Loss Type:     Focal Loss (gamma=2.5)")
    report.append("  Class Weights: [1.0, 5.0, 1.0] (Restaurants) / [1.0, 8.0, 1.0] (Laptops)")
    report.append("")

    # 實驗結果表格
    report.append("-" * 80)
    report.append("實驗結果對比")
    report.append("-" * 80)
    report.append(f"{'Baseline':<20} {'Batch':>6} {'Test Acc':>10} {'Test F1':>10} {'Val F1':>10} {'Neg F1':>10} {'Neu F1':>10} {'Pos F1':>10} {'Epoch':>6}")
    report.append("-" * 80)

    for r in results:
        baseline = r['description']
        batch = r['batch_size']
        acc = f"{r['test_acc']:.4f}" if r['test_acc'] else "N/A"
        f1 = f"{r['test_f1']:.4f}" if r['test_f1'] else "N/A"
        val_f1 = f"{r['val_f1']:.4f}" if r['val_f1'] else "N/A"
        f1_neg = f"{r['test_f1_neg']:.4f}" if r['test_f1_neg'] else "N/A"
        f1_neu = f"{r['test_f1_neu']:.4f}" if r['test_f1_neu'] else "N/A"
        f1_pos = f"{r['test_f1_pos']:.4f}" if r['test_f1_pos'] else "N/A"
        epoch = f"{r['best_epoch']}" if r['best_epoch'] else "N/A"

        report.append(f"{baseline:<20} {batch:>6} {acc:>10} {f1:>10} {val_f1:>10} {f1_neg:>10} {f1_neu:>10} {f1_pos:>10} {epoch:>6}")

    report.append("")

    # 詳細分析
    valid_results = [r for r in results if r['test_acc']]
    if valid_results:
        best_acc = max(valid_results, key=lambda x: x['test_acc'])
        best_f1 = max(valid_results, key=lambda x: x['test_f1'] or 0)

        report.append("-" * 80)
        report.append("詳細分析")
        report.append("-" * 80)
        report.append(f"\n最佳準確率:")
        report.append(f"  模型:          {best_acc['description']}")
        report.append(f"  Test Accuracy: {best_acc['test_acc']:.4f}")
        report.append(f"  Test F1:       {best_acc['test_f1']:.4f}")
        report.append(f"  Best Epoch:    {best_acc['best_epoch']}")

        report.append(f"\n最佳 F1 分數:")
        report.append(f"  模型:          {best_f1['description']}")
        report.append(f"  Test F1:       {best_f1['test_f1']:.4f}")
        report.append(f"  Test Accuracy: {best_f1['test_acc']:.4f}")
        report.append(f"  Best Epoch:    {best_f1['best_epoch']}")
        report.append("")

    # 實驗目錄
    report.append("-" * 80)
    report.append("實驗目錄")
    report.append("-" * 80)
    for r in results:
        if r['exp_name']:
            report.append(f"  {r['description']:<20} {r['exp_name']}")
    report.append("")

    # 結論
    report.append("-" * 80)
    report.append("結論")
    report.append("-" * 80)
    report.append("根據以上實驗結果，可以觀察到:")
    report.append("")
    report.append("1. BERT Only - 最簡單的基線，僅使用 BERT 的 [CLS] token")
    report.append("2. BERT + AAHA - 加入層次化注意力機制，但不包含 PMAC/IARM")
    report.append("3. BERT + Mean Pooling - 簡單的平均池化方法")
    report.append("")
    report.append("這些基線將作為評估 PMAC 和 IARM 創新模組效果的對照組。")
    report.append("")
    report.append("=" * 80)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='生成 Baseline 比較報告')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['restaurants', 'laptops', 'mams'],
                        help='數據集選擇 (restaurants, laptops, 或 mams)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    baseline_dataset_dir = project_root / "results" / "baseline" / args.dataset

    if not baseline_dataset_dir.exists():
        print(f"錯誤: results/baseline/{args.dataset}/ 目錄不存在")
        print(f"請先運行 {args.dataset} 數據集的 baseline 實驗")
        return

    # 定義 baseline（MAMS 只有 2 個，SemEval 有 3 個）
    if args.dataset == 'mams':
        baseline_info = {
            'bert_only': ("BERT Only", 32),
            'bert_aaha': ("BERT + AAHA", 24),
        }
    else:
        baseline_info = {
            'bert_only': ("BERT Only", 32),
            'bert_aaha': ("BERT + AAHA", 24),
            'bert_mean': ("BERT + Mean Pooling", 32)
        }

    # 查找所有 baseline
    print(f"\n收集 {args.dataset.upper()} Baseline 實驗結果\n")
    found_baselines = find_all_baselines(baseline_dataset_dir)

    results = []
    for baseline_type, (description, batch_size) in baseline_info.items():
        print(f"查找 {description}...")

        if baseline_type in found_baselines:
            exp_dir = found_baselines[baseline_type]
            metrics = read_metrics(exp_dir)
            results.append({
                'type': baseline_type,
                'description': description,
                'batch_size': batch_size,
                'exp_name': metrics['exp_name'],
                **metrics
            })
            print(f"  [OK] 找到: {exp_dir.name}")
            if metrics['test_acc']:
                print(f"       Test Acc: {metrics['test_acc']:.4f}, F1: {metrics['test_f1']:.4f}, Best Epoch: {metrics['best_epoch']}")
        else:
            print(f"  [X] 未找到 {baseline_type} 的實驗結果")
            results.append({
                'type': baseline_type,
                'description': description,
                'batch_size': batch_size,
                'exp_name': None,
                'test_acc': None,
                'test_f1': None,
                'val_f1': None,
                'best_epoch': None,
                'timestamp': None
            })

    if not any(r['test_acc'] for r in results):
        print("\n錯誤: 沒有找到任何有效的實驗結果")
        return

    # 生成純文字報告
    text_report = generate_text_report(args.dataset, results)

    # 保存在數據集資料夾內
    report_path = baseline_dataset_dir / "綜合報告.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text_report)

    print("\n" + "="*80)
    print(f"報告生成完成: {report_path}")
    print("="*80 + "\n")

    # 打印報告內容
    print(text_report)


if __name__ == "__main__":
    main()
