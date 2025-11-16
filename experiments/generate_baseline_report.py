"""
生成 Baseline 比較報告

從 results/baseline/ 目錄讀取所有 baseline 實驗結果，生成統整報告

使用方法:
    python experiments/generate_baseline_report.py
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd


def find_latest_baseline(baseline_dir, baseline_type):
    """查找最新的 baseline 實驗結果"""
    pattern = f"*_baseline_{baseline_type}_*"
    dirs = [d for d in baseline_dir.glob(pattern) if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.stat().st_mtime)


def read_metrics(exp_dir):
    """從實驗目錄讀取指標"""
    metrics = {
        'test_acc': None,
        'test_f1': None,
        'best_epoch': None,
        'timestamp': None
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

                # 讀取 best epoch (從 validation F1 最高的 epoch)
                if 'best_val_f1' in data and 'history' in data:
                    val_f1_list = data['history'].get('val_f1_macro', [])
                    if val_f1_list:
                        best_f1 = data['best_val_f1']
                        # 找到對應的 epoch
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


def generate_markdown(results):
    """生成 Markdown 報告"""
    report = "# Baseline 實驗比較報告\n\n"
    report += f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += "## 實驗配置\n\n"
    report += "| 參數 | 值 |\n"
    report += "|------|----|\n"
    report += "| Epochs | 30 |\n"
    report += "| Learning Rate | 2e-5 |\n"
    report += "| Dropout | 0.3 |\n"
    report += "| Loss Type | Focal Loss (gamma=2.0) |\n"
    report += "| Class Weights | [1.0, 3.0, 1.0] |\n"
    report += "| GPU | RTX 3090 (24GB) |\n\n"

    report += "## 實驗結果對比\n\n"
    report += "| Baseline | Batch Size | Test Acc | Test F1 | Best Epoch | 實驗時間 |\n"
    report += "|----------|------------|----------|---------|------------|----------|\n"

    for r in results:
        acc = f"{r['test_acc']:.4f}" if r['test_acc'] else "N/A"
        f1 = f"{r['test_f1']:.4f}" if r['test_f1'] else "N/A"
        epoch = r['best_epoch'] if r['best_epoch'] else "N/A"
        ts = r['timestamp'].strftime('%Y-%m-%d %H:%M') if r['timestamp'] else "N/A"

        report += f"| {r['description']} | {r['batch_size']} | {acc} | {f1} | {epoch} | {ts} |\n"

    # 找出最佳模型
    valid_results = [r for r in results if r['test_acc']]
    if valid_results:
        best_acc = max(valid_results, key=lambda x: x['test_acc'])
        best_f1 = max(valid_results, key=lambda x: x['test_f1'] or 0)

        report += "\n## 詳細分析\n\n"
        report += f"### 最佳準確率\n"
        report += f"- 模型: **{best_acc['description']}**\n"
        report += f"- Test Accuracy: **{best_acc['test_acc']:.4f}**\n"
        report += f"- Test F1: **{best_acc['test_f1']:.4f}**\n"
        report += f"- Best Epoch: {best_acc['best_epoch']}\n\n"

        report += f"### 最佳 F1 分數\n"
        report += f"- 模型: **{best_f1['description']}**\n"
        report += f"- Test F1: **{best_f1['test_f1']:.4f}**\n"
        report += f"- Test Accuracy: **{best_f1['test_acc']:.4f}**\n"
        report += f"- Best Epoch: {best_f1['best_epoch']}\n\n"

    report += "## 結論\n\n"
    report += "根據以上實驗結果，可以觀察到:\n\n"
    report += "1. **BERT Only**: 最簡單的基線，僅使用 BERT 的 [CLS] token\n"
    report += "2. **BERT + AAHA**: 加入層次化注意力機制，但不包含 PMAC/IARM\n"
    report += "3. **BERT + Mean Pooling**: 簡單的平均池化方法\n\n"
    report += "這些基線將作為評估 PMAC 和 IARM 創新模組效果的對照組。\n"

    return report


def print_summary(results):
    """打印摘要"""
    print("\n" + "="*80)
    print("BASELINE 實驗結果摘要")
    print("="*80 + "\n")

    for r in results:
        print(f"【{r['description']}】")
        if r['test_acc']:
            print(f"  Test Accuracy: {r['test_acc']:.4f}")
            print(f"  Test F1: {r['test_f1']:.4f}")
            print(f"  Best Epoch: {r['best_epoch']}")
        else:
            print(f"  未找到實驗結果")
        print()


def main():
    """生成 baseline 比較報告"""
    project_root = Path(__file__).parent.parent
    baseline_dir = project_root / "results" / "baseline"
    report_dir = project_root / "results" / "baseline_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_dir.exists():
        print("錯誤: results/baseline/ 目錄不存在")
        print("請先運行 baseline 實驗")
        return

    # 定義三個 baseline
    baselines = [
        ("bert_only", "BERT Only", 32),
        ("bert_aaha", "BERT + AAHA", 24),
        ("bert_mean", "BERT + Mean Pooling", 32)
    ]

    results = []

    print("\n收集 Baseline 實驗結果\n")

    for baseline_type, description, batch_size in baselines:
        print(f"查找 {description}...")
        exp_dir = find_latest_baseline(baseline_dir, baseline_type)

        if exp_dir:
            metrics = read_metrics(exp_dir)
            results.append({
                'type': baseline_type,
                'description': description,
                'batch_size': batch_size,
                'exp_dir': exp_dir,
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
                'exp_dir': None,
                'test_acc': None,
                'test_f1': None,
                'best_epoch': None,
                'timestamp': None
            })

    if not any(r['test_acc'] for r in results):
        print("\n錯誤: 沒有找到任何有效的實驗結果")
        return

    # 生成報告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Markdown 報告
    md_report = generate_markdown(results)
    md_path = report_dir / f"baseline_comparison_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)

    # JSON 報告
    json_data = [{k: str(v) if isinstance(v, Path) else v for k, v in r.items()} for r in results]
    json_path = report_dir / f"baseline_comparison_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    # CSV 報告
    csv_data = []
    for r in results:
        csv_data.append({
            'Baseline': r['description'],
            'Type': r['type'],
            'Batch_Size': r['batch_size'],
            'Test_Accuracy': r['test_acc'],
            'Test_F1': r['test_f1'],
            'Best_Epoch': r['best_epoch'],
            'Timestamp': r['timestamp']
        })
    df = pd.DataFrame(csv_data)
    csv_path = report_dir / f"baseline_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*80)
    print("報告生成完成:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    print("="*80 + "\n")

    # 打印摘要
    print_summary(results)


if __name__ == "__main__":
    main()
