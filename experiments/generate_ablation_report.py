"""
消融實驗報告生成腳本

功能：
1. 從 results/ablation/ 讀取所有消融實驗結果
2. 生成綜合分析報告（文字 + 圖表）
3. 計算各組件的貢獻度
4. 支援多資料集對比分析

使用方法:
    # 生成單一資料集的消融報告
    python experiments/generate_ablation_report.py --dataset laptops

    # 生成所有資料集的綜合報告
    python experiments/generate_ablation_report.py --all

    # 只生成圖表
    python experiments/generate_ablation_report.py --all --figures-only

輸出:
    results/ablation/
    ├── ablation_report_{dataset}.txt      # 單一資料集報告
    ├── ablation_report_summary.txt        # 綜合報告
    └── figures/
        ├── ablation_comparison_{dataset}.png   # 消融對比圖
        ├── ablation_heatmap.png                # 組件貢獻熱力圖
        └── ablation_neutral_impact.png         # Neutral 影響分析
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np

# 嘗試導入繪圖庫
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式後端
    # 設定中文字體
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[警告] matplotlib 未安裝，將跳過圖表生成")

# 消融變體定義
ABLATION_VARIANTS = {
    'full': '完整模型 (Full)',
    'no_senticnet': 'w/o SenticNet',
    'no_dynamic_gate': 'w/o Dynamic Gate',
    'no_confidence_gate': 'w/o Confidence Gate',
    'no_contrastive': 'w/o Contrastive',
    'no_logit_adjust': 'w/o Logit Adjust',
    'no_focal_loss': 'w/o Focal Loss',
    'no_llrd': 'w/o LLRD',
}

# 組件分類
COMPONENT_CATEGORIES = {
    '核心創新': ['no_senticnet', 'no_dynamic_gate', 'no_confidence_gate'],
    '訓練策略': ['no_contrastive', 'no_focal_loss', 'no_llrd'],
    '推理優化': ['no_logit_adjust'],
}

# 資料集列表
ALL_DATASETS = ['restaurants', 'laptops', 'mams', 'rest16', 'lap16']


def load_ablation_results(dataset):
    """載入指定資料集的所有消融實驗結果

    Args:
        dataset: 資料集名稱

    Returns:
        dict: {ablation_type: {metrics...}}
    """
    results_dir = Path("results/ablation") / dataset
    results = {}

    if not results_dir.exists():
        print(f"[警告] 找不到消融結果目錄: {results_dir}")
        return results

    # 尋找所有消融實驗目錄
    for ablation_type in ABLATION_VARIANTS.keys():
        # 尋找對應的實驗目錄
        pattern = f"*{ablation_type}*"
        exp_dirs = sorted(results_dir.glob(pattern),
                         key=lambda x: x.stat().st_mtime, reverse=True)

        if not exp_dirs:
            continue

        # 讀取最新的實驗結果
        latest_dir = exp_dirs[0]
        result_file = latest_dir / "reports" / "experiment_results.json"

        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    test_metrics = data.get('test_metrics', {})
                    results[ablation_type] = {
                        'accuracy': test_metrics.get('accuracy', 0),
                        'f1_macro': test_metrics.get('f1_macro', 0),
                        'f1_per_class': test_metrics.get('f1_per_class', [0, 0, 0]),
                        'exp_dir': str(latest_dir.name),
                    }
            except Exception as e:
                print(f"[警告] 無法讀取 {result_file}: {e}")

    return results


def load_multi_seed_results(dataset):
    """載入多種子消融實驗結果（從 JSON 報告）

    Args:
        dataset: 資料集名稱

    Returns:
        dict: {ablation_type: {mean, std, ...}}
    """
    reports_dir = Path("results/ablation")
    results = {}

    for ablation_type in ABLATION_VARIANTS.keys():
        json_file = reports_dir / f"ablation_{ablation_type}_{dataset}.json"

        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    agg = data.get('aggregated', {})
                    results[ablation_type] = {
                        'acc_mean': agg.get('accuracy', {}).get('mean', 0),
                        'acc_std': agg.get('accuracy', {}).get('std', 0),
                        'f1_mean': agg.get('f1_macro', {}).get('mean', 0),
                        'f1_std': agg.get('f1_macro', {}).get('std', 0),
                        'neg_mean': agg.get('f1_neg', {}).get('mean', 0),
                        'neg_std': agg.get('f1_neg', {}).get('std', 0),
                        'neu_mean': agg.get('f1_neu', {}).get('mean', 0),
                        'neu_std': agg.get('f1_neu', {}).get('std', 0),
                        'pos_mean': agg.get('f1_pos', {}).get('mean', 0),
                        'pos_std': agg.get('f1_pos', {}).get('std', 0),
                        'seeds': data.get('seeds', []),
                    }
            except Exception as e:
                print(f"[警告] 無法讀取 {json_file}: {e}")

    return results


def calculate_contribution(results):
    """計算各組件的貢獻度

    Args:
        results: 消融實驗結果

    Returns:
        dict: {ablation_type: contribution}
    """
    if 'full' not in results:
        return {}

    baseline_f1 = results['full'].get('f1_mean', results['full'].get('f1_macro', 0))
    contributions = {}

    for ablation_type, metrics in results.items():
        if ablation_type == 'full':
            continue

        ablation_f1 = metrics.get('f1_mean', metrics.get('f1_macro', 0))
        # 貢獻度 = 移除後的下降幅度（正值表示該組件有正面貢獻）
        contribution = baseline_f1 - ablation_f1
        contributions[ablation_type] = contribution

    return contributions


def generate_dataset_report(dataset, results, output_dir):
    """生成單一資料集的消融報告

    Args:
        dataset: 資料集名稱
        results: 消融實驗結果
        output_dir: 輸出目錄
    """
    if not results:
        print(f"[跳過] {dataset}: 無消融實驗結果")
        return

    # 判斷是多種子結果還是單次結果
    is_multi_seed = 'f1_mean' in results.get('full', {})

    # 計算貢獻度
    contributions = calculate_contribution(results)

    # 生成報告
    report = []
    report.append("=" * 80)
    report.append(f"消融實驗分析報告 - {dataset.upper()}")
    report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # === 1. 整體性能對比 ===
    report.append("-" * 80)
    report.append("1. 整體性能對比")
    report.append("-" * 80)

    if is_multi_seed:
        report.append(f"{'變體':<25} {'Accuracy':<18} {'Macro-F1':<18} {'Delta F1':<10}")
        report.append("-" * 80)

        # 按 F1 排序（從高到低）
        sorted_variants = sorted(results.items(),
                                key=lambda x: x[1].get('f1_mean', 0),
                                reverse=True)

        baseline_f1 = results.get('full', {}).get('f1_mean', 0)

        for ablation_type, metrics in sorted_variants:
            name = ABLATION_VARIANTS.get(ablation_type, ablation_type)
            acc = f"{metrics['acc_mean']:.2f} +/- {metrics['acc_std']:.2f}"
            f1 = f"{metrics['f1_mean']:.2f} +/- {metrics['f1_std']:.2f}"

            if ablation_type == 'full':
                delta = "-"
            else:
                delta_val = metrics['f1_mean'] - baseline_f1
                delta = f"{delta_val:+.2f}"

            report.append(f"{name:<25} {acc:<18} {f1:<18} {delta:<10}")
    else:
        report.append(f"{'變體':<25} {'Accuracy':<12} {'Macro-F1':<12} {'Delta F1':<10}")
        report.append("-" * 80)

        baseline_f1 = results.get('full', {}).get('f1_macro', 0) * 100

        for ablation_type in ['full'] + list(ABLATION_VARIANTS.keys())[1:]:
            if ablation_type not in results:
                continue

            metrics = results[ablation_type]
            name = ABLATION_VARIANTS.get(ablation_type, ablation_type)
            acc = f"{metrics['accuracy']*100:.2f}%"
            f1 = f"{metrics['f1_macro']*100:.2f}%"

            if ablation_type == 'full':
                delta = "-"
            else:
                delta_val = metrics['f1_macro'] * 100 - baseline_f1
                delta = f"{delta_val:+.2f}%"

            report.append(f"{name:<25} {acc:<12} {f1:<12} {delta:<10}")

    report.append("")

    # === 2. 類別層面分析 ===
    if is_multi_seed:
        report.append("-" * 80)
        report.append("2. 類別層面分析 (F1 Score)")
        report.append("-" * 80)
        report.append(f"{'變體':<25} {'Negative':<18} {'Neutral':<18} {'Positive':<18}")
        report.append("-" * 80)

        for ablation_type in ['full'] + list(ABLATION_VARIANTS.keys())[1:]:
            if ablation_type not in results:
                continue

            metrics = results[ablation_type]
            name = ABLATION_VARIANTS.get(ablation_type, ablation_type)
            neg = f"{metrics['neg_mean']:.2f} +/- {metrics['neg_std']:.2f}"
            neu = f"{metrics['neu_mean']:.2f} +/- {metrics['neu_std']:.2f}"
            pos = f"{metrics['pos_mean']:.2f} +/- {metrics['pos_std']:.2f}"
            report.append(f"{name:<25} {neg:<18} {neu:<18} {pos:<18}")

        report.append("")

    # === 3. 組件貢獻度排序 ===
    if contributions:
        report.append("-" * 80)
        report.append("3. 組件貢獻度排序 (移除後 F1 下降幅度)")
        report.append("-" * 80)

        # 按貢獻度排序（從高到低）
        sorted_contrib = sorted(contributions.items(), key=lambda x: x[1], reverse=True)

        for i, (ablation_type, contrib) in enumerate(sorted_contrib, 1):
            name = ABLATION_VARIANTS.get(ablation_type, ablation_type)
            bar = "+" * int(abs(contrib) * 5) if contrib > 0 else "-" * int(abs(contrib) * 5)
            sign = "+" if contrib > 0 else ""
            report.append(f"  {i}. {name:<25} {sign}{contrib:.2f}% {bar}")

        report.append("")

        # 貢獻度統計
        positive_contrib = [c for c in contributions.values() if c > 0]
        if positive_contrib:
            report.append(f"  * 有正面貢獻的組件: {len(positive_contrib)}/{len(contributions)}")
            report.append(f"  * 總貢獻度: {sum(positive_contrib):.2f}%")
            report.append(f"  * 平均貢獻度: {np.mean(positive_contrib):.2f}%")

    report.append("")

    # === 4. 分類別貢獻分析 ===
    report.append("-" * 80)
    report.append("4. 組件分類貢獻分析")
    report.append("-" * 80)

    for category, variants in COMPONENT_CATEGORIES.items():
        category_contribs = [contributions.get(v, 0) for v in variants if v in contributions]
        if category_contribs:
            avg_contrib = np.mean(category_contribs)
            report.append(f"  {category}:")
            for v in variants:
                if v in contributions:
                    name = ABLATION_VARIANTS.get(v, v)
                    report.append(f"    - {name}: {contributions[v]:+.2f}%")
            report.append(f"    [平均貢獻: {avg_contrib:+.2f}%]")
            report.append("")

    # === 5. 關鍵發現 ===
    report.append("-" * 80)
    report.append("5. 關鍵發現")
    report.append("-" * 80)

    if contributions:
        # 最重要的組件
        most_important = max(contributions.items(), key=lambda x: x[1])
        if most_important[1] > 0:
            report.append(f"  * 最重要的組件: {ABLATION_VARIANTS.get(most_important[0], most_important[0])}")
            report.append(f"    移除後 F1 下降 {most_important[1]:.2f}%")
            report.append("")

        # Neutral 識別相關
        if is_multi_seed and 'full' in results:
            baseline_neu = results['full'].get('neu_mean', 0)
            neu_impacts = []
            for abl, metrics in results.items():
                if abl != 'full':
                    neu_drop = baseline_neu - metrics.get('neu_mean', 0)
                    neu_impacts.append((abl, neu_drop))

            if neu_impacts:
                most_neu_impact = max(neu_impacts, key=lambda x: x[1])
                if most_neu_impact[1] > 1:
                    report.append(f"  * 對 Neutral 識別最重要: {ABLATION_VARIANTS.get(most_neu_impact[0], most_neu_impact[0])}")
                    report.append(f"    移除後 Neutral F1 下降 {most_neu_impact[1]:.2f}%")

    report.append("")
    report.append("=" * 80)

    # 保存報告（存到 results/ablation/{dataset}/ 下）
    report_text = "\n".join(report)
    dataset_dir = output_dir / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_file = dataset_dir / f"ablation_report_{dataset}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"[完成] 報告已保存: {output_file}")

    return report_text


def generate_figures(dataset, results, output_dir):
    """生成消融實驗圖表

    Args:
        dataset: 資料集名稱
        results: 消融實驗結果
        output_dir: 輸出目錄
    """
    if not HAS_MATPLOTLIB:
        return

    if not results:
        return

    # 圖表存到 results/ablation/{dataset}/figures/ 下
    figures_dir = output_dir / dataset / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    is_multi_seed = 'f1_mean' in results.get('full', {})

    # === 1. 消融對比柱狀圖 ===
    fig, ax = plt.subplots(figsize=(12, 6))

    variants = []
    f1_scores = []
    f1_stds = []
    colors = []

    for ablation_type in ['full'] + list(ABLATION_VARIANTS.keys())[1:]:
        if ablation_type not in results:
            continue

        variants.append(ABLATION_VARIANTS.get(ablation_type, ablation_type))

        if is_multi_seed:
            f1_scores.append(results[ablation_type]['f1_mean'])
            f1_stds.append(results[ablation_type]['f1_std'])
        else:
            f1_scores.append(results[ablation_type]['f1_macro'] * 100)
            f1_stds.append(0)

        # 顏色：full 用綠色，其他用藍色（下降多的用紅色）
        if ablation_type == 'full':
            colors.append('#2ecc71')
        elif is_multi_seed:
            drop = results['full']['f1_mean'] - results[ablation_type]['f1_mean']
            colors.append('#e74c3c' if drop > 2 else '#3498db')
        else:
            baseline = results['full']['f1_macro'] * 100
            drop = baseline - results[ablation_type]['f1_macro'] * 100
            colors.append('#e74c3c' if drop > 2 else '#3498db')

    x = np.arange(len(variants))
    bars = ax.bar(x, f1_scores, yerr=f1_stds if is_multi_seed else None,
                  color=colors, capsize=5, edgecolor='black', linewidth=0.5)

    # 添加數值標籤
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.annotate(f'{score:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Ablation Variant', fontsize=12)
    ax.set_ylabel('Macro-F1 (%)', fontsize=12)
    ax.set_title(f'Ablation Study Results - {dataset.upper()}', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha='right', fontsize=10)

    # 添加基準線
    if 'full' in results:
        baseline = results['full'].get('f1_mean', results['full'].get('f1_macro', 0) * 100)
        ax.axhline(y=baseline, color='#2ecc71', linestyle='--', linewidth=2, alpha=0.7, label='Full Model')
        ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig(figures_dir / f"ablation_comparison_{dataset}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[完成] 圖表已保存: {figures_dir / f'ablation_comparison_{dataset}.png'}")

    # === 2. 類別 F1 對比圖（僅多種子）===
    if is_multi_seed:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        class_names = ['Negative', 'Neutral', 'Positive']
        class_keys = ['neg_mean', 'neu_mean', 'pos_mean']
        std_keys = ['neg_std', 'neu_std', 'pos_std']

        for ax, cls_name, key, std_key in zip(axes, class_names, class_keys, std_keys):
            variants_plot = []
            scores = []
            stds = []

            for ablation_type in ['full'] + list(ABLATION_VARIANTS.keys())[1:]:
                if ablation_type not in results:
                    continue
                variants_plot.append(ABLATION_VARIANTS.get(ablation_type, ablation_type)[:12])
                scores.append(results[ablation_type][key])
                stds.append(results[ablation_type][std_key])

            x = np.arange(len(variants_plot))
            bars = ax.bar(x, scores, yerr=stds, capsize=3, color='#3498db', edgecolor='black', linewidth=0.5)

            ax.set_xlabel('Variant', fontsize=10)
            ax.set_ylabel(f'{cls_name} F1 (%)', fontsize=10)
            ax.set_title(f'{cls_name}', fontsize=12, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(variants_plot, rotation=45, ha='right', fontsize=8)

            # 基準線
            if 'full' in results:
                baseline = results['full'][key]
                ax.axhline(y=baseline, color='#2ecc71', linestyle='--', linewidth=1.5, alpha=0.7)

        plt.suptitle(f'Per-Class F1 Analysis - {dataset.upper()}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(figures_dir / f"ablation_per_class_{dataset}.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"[完成] 圖表已保存: {figures_dir / f'ablation_per_class_{dataset}.png'}")


def generate_cross_dataset_report(all_results, output_dir):
    """生成跨資料集綜合報告

    Args:
        all_results: {dataset: results}
        output_dir: 輸出目錄
    """
    report = []
    report.append("=" * 100)
    report.append("消融實驗綜合分析報告 (Cross-Dataset)")
    report.append(f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)
    report.append("")

    # === 1. 各資料集 Full Model 性能 ===
    report.append("-" * 100)
    report.append("1. 各資料集 Full Model 基準性能")
    report.append("-" * 100)
    report.append(f"{'資料集':<15} {'Accuracy':<18} {'Macro-F1':<18} {'Neu F1':<18}")
    report.append("-" * 100)

    for dataset in ALL_DATASETS:
        if dataset not in all_results or 'full' not in all_results[dataset]:
            continue

        metrics = all_results[dataset]['full']
        is_multi = 'f1_mean' in metrics

        if is_multi:
            acc = f"{metrics['acc_mean']:.2f} +/- {metrics['acc_std']:.2f}"
            f1 = f"{metrics['f1_mean']:.2f} +/- {metrics['f1_std']:.2f}"
            neu = f"{metrics['neu_mean']:.2f} +/- {metrics['neu_std']:.2f}"
        else:
            acc = f"{metrics['accuracy']*100:.2f}%"
            f1 = f"{metrics['f1_macro']*100:.2f}%"
            neu = f"{metrics['f1_per_class'][1]*100:.2f}%" if len(metrics.get('f1_per_class', [])) >= 3 else "N/A"

        report.append(f"{dataset.upper():<15} {acc:<18} {f1:<18} {neu:<18}")

    report.append("")

    # === 2. 組件貢獻度矩陣 ===
    report.append("-" * 100)
    report.append("2. 組件貢獻度矩陣 (Delta F1 %)")
    report.append("-" * 100)

    # 表頭
    header = f"{'組件':<25}"
    for dataset in ALL_DATASETS:
        if dataset in all_results:
            header += f"{dataset.upper():<12}"
    header += f"{'平均':<12}"
    report.append(header)
    report.append("-" * 100)

    # 計算各資料集的貢獻度
    all_contributions = {}
    for dataset, results in all_results.items():
        all_contributions[dataset] = calculate_contribution(results)

    # 輸出每個組件
    for ablation_type in list(ABLATION_VARIANTS.keys())[1:]:
        name = ABLATION_VARIANTS.get(ablation_type, ablation_type)
        row = f"{name:<25}"
        contribs = []

        for dataset in ALL_DATASETS:
            if dataset in all_contributions:
                contrib = all_contributions[dataset].get(ablation_type, None)
                if contrib is not None:
                    row += f"{contrib:+.2f}%      "
                    contribs.append(contrib)
                else:
                    row += f"{'N/A':<12}"
            else:
                row += f"{'N/A':<12}"

        # 平均貢獻
        if contribs:
            avg = np.mean(contribs)
            row += f"{avg:+.2f}%"
        else:
            row += "N/A"

        report.append(row)

    report.append("")

    # === 3. 關鍵發現總結 ===
    report.append("-" * 100)
    report.append("3. 關鍵發現總結")
    report.append("-" * 100)

    # 計算每個組件的平均貢獻
    avg_contributions = {}
    for ablation_type in list(ABLATION_VARIANTS.keys())[1:]:
        contribs = []
        for dataset, contrib_dict in all_contributions.items():
            if ablation_type in contrib_dict:
                contribs.append(contrib_dict[ablation_type])
        if contribs:
            avg_contributions[ablation_type] = np.mean(contribs)

    if avg_contributions:
        # 排序
        sorted_avg = sorted(avg_contributions.items(), key=lambda x: x[1], reverse=True)

        report.append("  [組件重要性排序 (跨資料集平均)]")
        for i, (abl, avg) in enumerate(sorted_avg, 1):
            name = ABLATION_VARIANTS.get(abl, abl)
            indicator = "[重要]" if avg > 1 else "[輔助]" if avg > 0 else "[待驗證]"
            report.append(f"    {i}. {name:<25} {avg:+.2f}% {indicator}")

        report.append("")

        # 分類總結
        report.append("  [分類貢獻總結]")
        for category, variants in COMPONENT_CATEGORIES.items():
            category_avg = [avg_contributions.get(v, 0) for v in variants if v in avg_contributions]
            if category_avg:
                report.append(f"    {category}: {np.mean(category_avg):+.2f}% (平均)")

    report.append("")
    report.append("=" * 100)

    # 保存報告（綜合報告存到 results/ablation/ 根目錄）
    # 文件名包含所有資料集名稱
    report_text = "\n".join(report)
    dataset_names = "_".join(sorted(all_results.keys()))
    output_file = output_dir / f"ablation_report_summary_{dataset_names}.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"[完成] 綜合報告已保存: {output_file}")

    return report_text


def generate_heatmap(all_results, output_dir):
    """生成組件貢獻度熱力圖

    Args:
        all_results: {dataset: results}
        output_dir: 輸出目錄
    """
    if not HAS_MATPLOTLIB:
        return

    # 熱力圖存到 results/ablation/figures/ 根目錄
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 收集數據
    datasets = [d for d in ALL_DATASETS if d in all_results]
    ablation_types = list(ABLATION_VARIANTS.keys())[1:]  # 排除 full

    if not datasets:
        return

    # 建立貢獻度矩陣
    matrix = []
    valid_ablations = []

    for ablation_type in ablation_types:
        row = []
        has_data = False
        for dataset in datasets:
            contrib = calculate_contribution(all_results[dataset]).get(ablation_type, np.nan)
            row.append(contrib)
            if not np.isnan(contrib):
                has_data = True

        if has_data:
            matrix.append(row)
            valid_ablations.append(ablation_type)

    if not matrix:
        return

    matrix = np.array(matrix)

    # 繪製熱力圖
    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用發散色圖（紅色=貢獻大，藍色=貢獻小/負）
    cmap = plt.cm.RdYlGn
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)))
    vmin = -vmax

    im = ax.imshow(matrix, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)

    # 添加數值標籤
    for i in range(len(valid_ablations)):
        for j in range(len(datasets)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if abs(val) > vmax * 0.6 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', color=text_color, fontsize=10)

    # 設定軸標籤
    ax.set_xticks(np.arange(len(datasets)))
    ax.set_yticks(np.arange(len(valid_ablations)))
    ax.set_xticklabels([d.upper() for d in datasets], fontsize=11)
    ax.set_yticklabels([ABLATION_VARIANTS.get(a, a) for a in valid_ablations], fontsize=10)

    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Ablation Variant', fontsize=12)
    ax.set_title('Component Contribution Heatmap (Delta F1 %)', fontsize=14, fontweight='bold')

    # 添加色標
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Delta F1 (%)', fontsize=11)

    plt.tight_layout()
    plt.savefig(figures_dir / "ablation_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[完成] 熱力圖已保存: {figures_dir / 'ablation_heatmap.png'}")


def main():
    parser = argparse.ArgumentParser(description='消融實驗報告生成')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=ALL_DATASETS,
                        help='生成單一資料集的報告')
    parser.add_argument('--all', action='store_true',
                        help='生成所有資料集的綜合報告')
    parser.add_argument('--figures-only', action='store_true',
                        help='只生成圖表')
    parser.add_argument('--use-multi-seed', action='store_true',
                        help='使用多種子結果（從 ablation_*.json 讀取）')

    args = parser.parse_args()

    output_dir = Path("results/ablation")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        # 生成所有資料集的報告
        all_results = {}

        for dataset in ALL_DATASETS:
            print(f"\n[處理] {dataset.upper()}")

            if args.use_multi_seed:
                results = load_multi_seed_results(dataset)
            else:
                results = load_ablation_results(dataset)

            if results:
                all_results[dataset] = results

                if not args.figures_only:
                    generate_dataset_report(dataset, results, output_dir)

                generate_figures(dataset, results, output_dir)

        # 生成綜合報告
        if all_results and not args.figures_only:
            print(f"\n[處理] 綜合報告")
            generate_cross_dataset_report(all_results, output_dir)
            generate_heatmap(all_results, output_dir)

    elif args.dataset:
        # 生成單一資料集的報告
        print(f"\n[處理] {args.dataset.upper()}")

        if args.use_multi_seed:
            results = load_multi_seed_results(args.dataset)
        else:
            results = load_ablation_results(args.dataset)

        if results:
            if not args.figures_only:
                generate_dataset_report(args.dataset, results, output_dir)
            generate_figures(args.dataset, results, output_dir)
        else:
            print(f"[錯誤] 找不到 {args.dataset} 的消融實驗結果")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
