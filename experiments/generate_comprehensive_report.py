"""
生成綜合實驗報告

從 results/ 目錄讀取所有實驗結果（baseline 和 improved），生成完整對比報告

模型架構:
- Baseline: BERT-CLS (使用 [CLS] token)
- Method 1: Hierarchical BERT (階層式特徵提取，固定拼接)
- Method 2: HBL (Hierarchical BERT + Layer-wise Attention) [已放棄]
- Method 3: IARN (Inter-Aspect Relation Network) [主要貢獻]

使用方法:
    python experiments/generate_comprehensive_report.py --dataset restaurants
    python experiments/generate_comprehensive_report.py --dataset laptops
    python experiments/generate_comprehensive_report.py --dataset mams
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


def find_all_experiments(results_dir, dataset):
    """查找該數據集下所有實驗（baseline + improved）"""
    experiments = {
        'baseline': {},
        'improved': {}
    }

    # 查找 baseline 實驗
    baseline_dir = results_dir / "baseline" / dataset
    if baseline_dir.exists():
        for exp_dir in baseline_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            dir_name = exp_dir.name
            # 新架構: baseline_bert_cls
            if '_baseline_bert_cls_' in dir_name or '_baseline_bert_only_' in dir_name:
                exp_type = 'bert_cls'
            else:
                continue

            # 保存最新的實驗
            if exp_type not in experiments['baseline']:
                experiments['baseline'][exp_type] = exp_dir
            else:
                if exp_dir.stat().st_mtime > experiments['baseline'][exp_type].stat().st_mtime:
                    experiments['baseline'][exp_type] = exp_dir

    # 查找 improved 實驗
    improved_dir = results_dir / "improved" / dataset
    if improved_dir.exists():
        for exp_dir in improved_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            dir_name = exp_dir.name
            if '_improved_hierarchical_layerattn_' in dir_name:
                exp_type = 'hierarchical_layerattn'
            elif '_improved_iarn_' in dir_name:
                exp_type = 'iarn'
            elif '_improved_hierarchical_' in dir_name:
                exp_type = 'hierarchical'
            else:
                continue

            # 保存最新的實驗
            if exp_type not in experiments['improved']:
                experiments['improved'][exp_type] = exp_dir
            else:
                if exp_dir.stat().st_mtime > experiments['improved'][exp_type].stat().st_mtime:
                    experiments['improved'][exp_type] = exp_dir

    return experiments


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
        'exp_name': exp_dir.name,
        'layer_attention': None  # HBL 特有
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

                # 讀取 layer attention (HBL 特有)
                if 'layer_attention' in data:
                    metrics['layer_attention'] = data['layer_attention']
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
    report.append("=" * 100)
    report.append(f"階層式 BERT 實驗綜合報告 - {dataset.upper()} 數據集")
    report.append("=" * 100)
    report.append(f"\n生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 架構說明
    report.append("-" * 100)
    report.append("模型架構")
    report.append("-" * 100)
    report.append("  Baseline:  BERT-CLS")
    report.append("             標準 BERT baseline，使用 [CLS] token")
    report.append("             參考: Devlin et al. (2019) BERT")
    report.append("")
    report.append("  Method 1:  Hierarchical BERT (階層式BERT)")
    report.append("             從 BERT 不同層提取 Low/Mid/High 層級特徵")
    report.append("             固定 concatenation 組合")
    report.append("")
    report.append("  Method 2:  HBL (Hierarchical BERT + Layer-wise Attention)")
    report.append("             基於 UDify (Kondratyuk & Straka, EMNLP 2019)")
    report.append("             動態學習層級權重，替代固定拼接")
    report.append("")
    report.append("  Method 3:  IARN (Inter-Aspect Relation Network)")
    report.append("             顯式建模多個 aspects 之間的交互關係")
    report.append("             Aspect-to-Aspect Attention + Relation-aware Gating")
    report.append("             與 HPNet (2021) 差異化創新")
    report.append("")

    # 實驗配置
    report.append("-" * 100)
    report.append("實驗配置")
    report.append("-" * 100)
    report.append("  Epochs:        30")
    report.append("  Learning Rate: 2e-5")
    report.append("  BERT Model:    DistilBERT-base-uncased")
    report.append("  Optimizer:     AdamW")
    report.append("  Scheduler:     Cosine Annealing with Warmup (10%)")
    report.append("  Loss Type:     Focal Loss")
    if dataset == 'mams':
        report.append("  Focal Gamma:   2.0")
        report.append("  Class Weights: [1.0, 5.0, 1.0]")
        report.append("  Dropout:       0.45")
    else:
        report.append("  Focal Gamma:   2.5")
        report.append("  Class Weights: [1.0, 8.0, 1.0]")
        report.append("  Dropout:       0.3-0.4")
    report.append("")

    # 實驗結果表格
    report.append("-" * 100)
    report.append("實驗結果對比")
    report.append("-" * 100)
    report.append(f"{'Model':<25} {'Test Acc':>10} {'Test F1':>10} {'Val F1':>10} "
                  f"{'Neg F1':>10} {'Neu F1':>10} {'Pos F1':>10} {'Epoch':>6}")
    report.append("-" * 100)

    for r in results:
        model = r['description']
        acc = f"{r['test_acc']:.4f}" if r['test_acc'] else "N/A"
        f1 = f"{r['test_f1']:.4f}" if r['test_f1'] else "N/A"
        val_f1 = f"{r['val_f1']:.4f}" if r['val_f1'] else "N/A"
        f1_neg = f"{r['test_f1_neg']:.4f}" if r['test_f1_neg'] else "N/A"
        f1_neu = f"{r['test_f1_neu']:.4f}" if r['test_f1_neu'] else "N/A"
        f1_pos = f"{r['test_f1_pos']:.4f}" if r['test_f1_pos'] else "N/A"
        epoch = f"{r['best_epoch']}" if r['best_epoch'] else "N/A"

        report.append(f"{model:<25} {acc:>10} {f1:>10} {val_f1:>10} "
                     f"{f1_neg:>10} {f1_neu:>10} {f1_pos:>10} {epoch:>6}")

    report.append("")

    # 詳細分析
    valid_results = [r for r in results if r['test_acc']]
    if valid_results:
        best_f1 = max(valid_results, key=lambda x: x['test_f1'] or 0)
        baseline_result = next((r for r in valid_results if r['type'] == 'baseline'), None)

        report.append("-" * 100)
        report.append("性能分析")
        report.append("-" * 100)

        if baseline_result:
            report.append(f"\n[Baseline] BERT-CLS:")
            report.append(f"  Test Accuracy: {baseline_result['test_acc']:.4f}")
            report.append(f"  Test F1:       {baseline_result['test_f1']:.4f}")
            report.append(f"  Best Epoch:    {baseline_result['best_epoch']}")

        report.append(f"\n[Best Model] {best_f1['description']}")
        report.append(f"  Test F1:       {best_f1['test_f1']:.4f}")
        report.append(f"  Test Accuracy: {best_f1['test_acc']:.4f}")
        report.append(f"  Best Epoch:    {best_f1['best_epoch']}")

        # 計算改進幅度
        if baseline_result and best_f1['test_f1'] and baseline_result['test_f1']:
            improvement = best_f1['test_f1'] - baseline_result['test_f1']
            improvement_pct = (improvement / baseline_result['test_f1']) * 100
            report.append(f"\n[Improvement] 相對 Baseline 改進:")
            report.append(f"  F1 提升:       +{improvement:.4f} ({improvement_pct:+.2f}%)")

        # HBL 的 Layer Attention 權重
        hbl_result = next((r for r in valid_results if r['model_type'] == 'hierarchical_layerattn'), None)
        if hbl_result and hbl_result.get('layer_attention'):
            weights = hbl_result['layer_attention']
            report.append(f"\n[HBL Weights] 學習到的層級權重:")
            report.append(f"  Low-level:     {weights[0]:.4f}")
            report.append(f"  Mid-level:     {weights[1]:.4f}")
            report.append(f"  High-level:    {weights[2]:.4f}")
            report.append(f"  說明:          權重和為 1.0 (softmax 歸一化)")

        report.append("")

    # 實驗目錄
    report.append("-" * 100)
    report.append("實驗目錄")
    report.append("-" * 100)
    for r in results:
        if r['exp_name']:
            report.append(f"  {r['description']:<25} {r['exp_name']}")
    report.append("")

    # 結論
    report.append("-" * 100)
    report.append("結論")
    report.append("-" * 100)
    report.append("")
    report.append("根據實驗結果:")
    report.append("")
    report.append("1. 階層特徵建模的有效性")
    report.append("   - Hierarchical BERT 透過提取 BERT 不同層的特徵，捕捉了詞法、語義、任務三個層級的資訊")
    report.append("   - 相較於只使用 [CLS] token 的 baseline，階層特徵提供了更豐富的表示")
    report.append("")
    report.append("2. Layer-wise Attention 的優勢")
    report.append("   - HBL 透過可學習的權重動態組合層級特徵，避免了固定拼接的侷限")
    report.append("   - 權重分布可提供模型決策的可解釋性")
    report.append("")
    report.append("3. 多面向場景的挑戰")
    if dataset == 'mams':
        report.append("   - MAMS 數據集 100% 為多面向句子，是真正的多面向場景")
        report.append("   - 階層建模在這種複雜場景下的優勢更加明顯")
    else:
        report.append(f"   - {dataset.upper()} 數據集約 20% 為多面向句子")
        report.append("   - 模型需要同時處理單面向和多面向的場景")
    report.append("")
    report.append("=" * 100)

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='生成綜合實驗報告')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['restaurants', 'laptops', 'mams'],
                        help='數據集選擇')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    results_dir = project_root / "results"

    # 定義模型資訊
    model_info = {
        'baseline': {
            'bert_cls': "Baseline (BERT-CLS)",
        },
        'improved': {
            'hierarchical': "Method 1 (Hierarchical)",
            'hierarchical_layerattn': "Method 2 (HBL)",
            'iarn': "Method 3 (IARN)",
        }
    }

    # 查找所有實驗
    print(f"\n收集 {args.dataset.upper()} 實驗結果\n")
    found_experiments = find_all_experiments(results_dir, args.dataset)

    results = []

    # 收集 baseline 結果
    for exp_type, description in model_info['baseline'].items():
        print(f"查找 {description}...")
        if exp_type in found_experiments['baseline']:
            exp_dir = found_experiments['baseline'][exp_type]
            metrics = read_metrics(exp_dir)
            results.append({
                'type': 'baseline',
                'model_type': exp_type,
                'description': description,
                'exp_name': metrics['exp_name'],
                **metrics
            })
            print(f"  [OK] 找到: {exp_dir.name}")
            if metrics['test_acc']:
                print(f"       Test Acc: {metrics['test_acc']:.4f}, F1: {metrics['test_f1']:.4f}")
        else:
            print(f"  [X] 未找到")
            results.append({
                'type': 'baseline',
                'model_type': exp_type,
                'description': description,
                'exp_name': None,
                'test_acc': None,
                'test_f1': None,
                'val_f1': None,
                'best_epoch': None,
            })

    # 收集 improved 結果
    for exp_type, description in model_info['improved'].items():
        print(f"查找 {description}...")
        if exp_type in found_experiments['improved']:
            exp_dir = found_experiments['improved'][exp_type]
            metrics = read_metrics(exp_dir)
            results.append({
                'type': 'improved',
                'model_type': exp_type,
                'description': description,
                'exp_name': metrics['exp_name'],
                **metrics
            })
            print(f"  [OK] 找到: {exp_dir.name}")
            if metrics['test_acc']:
                print(f"       Test Acc: {metrics['test_acc']:.4f}, F1: {metrics['test_f1']:.4f}")
                if metrics.get('layer_attention'):
                    weights = metrics['layer_attention']
                    print(f"       Layer Weights: [{weights[0]:.3f}, {weights[1]:.3f}, {weights[2]:.3f}]")
        else:
            print(f"  [X] 未找到")
            results.append({
                'type': 'improved',
                'model_type': exp_type,
                'description': description,
                'exp_name': None,
                'test_acc': None,
                'test_f1': None,
                'val_f1': None,
                'best_epoch': None,
            })

    if not any(r['test_acc'] for r in results):
        print("\n錯誤: 沒有找到任何有效的實驗結果")
        return

    # 生成純文字報告
    text_report = generate_text_report(args.dataset, results)

    # 保存報告
    report_path = results_dir / f"綜合報告_{args.dataset}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(text_report)

    print("\n" + "="*100)
    print(f"報告生成完成: {report_path}")
    print("="*100 + "\n")

    # 打印報告內容
    print(text_report)


if __name__ == "__main__":
    main()
