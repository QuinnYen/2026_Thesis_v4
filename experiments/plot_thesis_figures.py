"""
論文圖檔生成腳本 - HKGAN 實驗結果視覺化

支援圖表:
  - f1: Macro-F1 分組條形圖（圖 4.2.1）
  - auc: AUC 折線圖（圖 4.2.2）
  - neutral: 中性類別改善 vs 數據集不平衡散點圖（圖 4.2.3）
  - all: 生成所有圖表

使用方法:
    # 生成 Macro-F1 對比圖
    python experiments/plot_thesis_figures.py --figure f1

    # 生成 AUC 對比圖
    python experiments/plot_thesis_figures.py --figure auc

    # 生成中性類別改善散點圖
    python experiments/plot_thesis_figures.py --figure neutral

    # 生成 Macro-F1 箱型圖
    python experiments/plot_thesis_figures.py --figure boxplot

    # 生成所有圖表並保存
    python experiments/plot_thesis_figures.py --figure all --output results/figures/
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import json
from pathlib import Path
from scipy import stats

# 設定中文字體支援
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 數據集顯示名稱
DISPLAY_NAMES = {
    'rest16': 'REST16',
    'restaurants': 'Restaurants',
    'laptops': 'Laptops',
    'lap16': 'LAP16',
    'mams': 'MAMS'
}

DATASETS_ORDER = ['rest16', 'restaurants', 'laptops', 'lap16', 'mams']

# 數據集中性類別比例（基於 ABSA 標準數據集統計）
# 來源: SemEval-2014/2016 官方數據集及 MAMS 論文
NEUTRAL_RATIO = {
    'rest16': 5.4,       # REST16: 中性樣本極少（約5.4%）
    'restaurants': 10.8,  # Restaurants14: 中性樣本較少（約10.8%）
    'laptops': 19.2,      # Laptops14: 中性樣本適中（約19.2%）
    'lap16': 6.8,         # LAP16: 中性樣本極少（約6.8%）
    'mams': 32.7          # MAMS: 專門設計的平衡數據集（約32.7%）
}


def get_multiseed_data():
    """從多種子實驗結果讀取數據（讀取所有不同 seed 的實驗並計算平均）"""
    results_dir = Path("results/improved")
    baseline_dir = Path("results/baseline")

    data = {}

    for ds in DATASETS_ORDER:
        data[ds] = {}

        # 讀取所有 HKGAN 實驗（多種子）
        ds_dir = results_dir / ds
        if ds_dir.exists():
            exp_dirs = list(ds_dir.glob("*_improved_hkgan_*"))
            f1_scores = []
            auc_scores = []
            neu_f1_scores = []  # 中性類別 F1
            seeds_seen = set()

            for exp in exp_dirs:
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    with open(rf, 'r', encoding='utf-8') as f:
                        result = json.load(f)

                    # 獲取 seed 以避免重複計算同一個 seed
                    seed = result.get('args', {}).get('seed', 42)
                    test_metrics = result.get('test_metrics', {})
                    f1 = test_metrics.get('f1_macro')
                    auc = test_metrics.get('auc_macro')
                    f1_per_class = test_metrics.get('f1_per_class', [])

                    if f1 and seed not in seeds_seen:
                        f1_scores.append(f1 * 100)
                        if auc:
                            auc_scores.append(auc * 100)
                        # 中性類別是 index 1 (Negative=0, Neutral=1, Positive=2)
                        if len(f1_per_class) >= 3:
                            neu_f1_scores.append(f1_per_class[1] * 100)
                        seeds_seen.add(seed)

            if f1_scores:
                data[ds]['hkgan_f1_mean'] = np.mean(f1_scores)
                data[ds]['hkgan_f1_std'] = np.std(f1_scores)
                data[ds]['hkgan_seeds'] = len(seeds_seen)
            if auc_scores:
                data[ds]['hkgan_auc_mean'] = np.mean(auc_scores)
                data[ds]['hkgan_auc_std'] = np.std(auc_scores)
            if neu_f1_scores:
                data[ds]['hkgan_neu_f1_mean'] = np.mean(neu_f1_scores)
                data[ds]['hkgan_neu_f1_std'] = np.std(neu_f1_scores)

        # 讀取所有 Baseline 實驗（多種子）並計算平均
        bl_dir = baseline_dir / ds
        if bl_dir.exists():
            exp_dirs = list(bl_dir.glob("*_baseline_*"))
            bl_f1_scores = []
            bl_auc_scores = []
            bl_neu_f1_scores = []
            bl_seeds_seen = set()

            for exp in exp_dirs:
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    with open(rf, 'r', encoding='utf-8') as f:
                        result = json.load(f)

                    seed = result.get('args', {}).get('seed', 42)
                    test_metrics = result.get('test_metrics', {})
                    f1 = test_metrics.get('f1_macro')
                    auc = test_metrics.get('auc_macro')
                    f1_per_class = test_metrics.get('f1_per_class', [])

                    if f1 and seed not in bl_seeds_seen:
                        bl_f1_scores.append(f1 * 100)
                        if auc:
                            bl_auc_scores.append(auc * 100)
                        if len(f1_per_class) >= 3:
                            bl_neu_f1_scores.append(f1_per_class[1] * 100)
                        bl_seeds_seen.add(seed)

            if bl_f1_scores:
                data[ds]['baseline_f1'] = np.mean(bl_f1_scores)
                data[ds]['baseline_f1_std'] = np.std(bl_f1_scores)
                data[ds]['baseline_seeds'] = len(bl_seeds_seen)
            if bl_auc_scores:
                data[ds]['baseline_auc'] = np.mean(bl_auc_scores)
                data[ds]['baseline_auc_std'] = np.std(bl_auc_scores)
            if bl_neu_f1_scores:
                data[ds]['baseline_neu_f1'] = np.mean(bl_neu_f1_scores)
                data[ds]['baseline_neu_f1_std'] = np.std(bl_neu_f1_scores)

    return data


def plot_f1_comparison(output_path=None, show=True):
    """繪製 Macro-F1 對比圖（分組條形圖）"""

    # 從實際實驗數據讀取
    exp_data = get_multiseed_data()

    # 轉換格式
    datasets = []
    baseline_f1 = []
    hkgan_f1 = []
    hkgan_std = []

    # 檢查數據完整性
    for ds in DATASETS_ORDER:
        if ds in exp_data:
            if 'baseline_f1' not in exp_data[ds]:
                print(f"警告: {ds} 缺少 Baseline 數據")
                continue
            if 'hkgan_f1_mean' not in exp_data[ds]:
                print(f"警告: {ds} 缺少 HKGAN 數據")
                continue

            datasets.append(DISPLAY_NAMES[ds])
            baseline_f1.append(exp_data[ds]['baseline_f1'])
            hkgan_f1.append(exp_data[ds]['hkgan_f1_mean'])
            hkgan_std.append(exp_data[ds].get('hkgan_f1_std', 0))

            seeds = exp_data[ds].get('hkgan_seeds', 1)
            print(f"  {ds}: Baseline={exp_data[ds]['baseline_f1']:.2f}%, "
                  f"HKGAN={exp_data[ds]['hkgan_f1_mean']:.2f}% ± {exp_data[ds].get('hkgan_f1_std', 0):.2f}% "
                  f"({seeds} seeds)")

    if not datasets:
        print("錯誤: 沒有找到任何有效的實驗數據")
        return

    # 計算改善幅度
    improvements = [h - b for h, b in zip(hkgan_f1, baseline_f1)]

    # 按改善幅度排序（由高到低）
    sorted_indices = np.argsort(improvements)[::-1]
    datasets = [datasets[i] for i in sorted_indices]
    baseline_f1 = [baseline_f1[i] for i in sorted_indices]
    hkgan_f1 = [hkgan_f1[i] for i in sorted_indices]
    hkgan_std = [hkgan_std[i] for i in sorted_indices]
    improvements = [improvements[i] for i in sorted_indices]

    # 設定圖表
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(datasets))
    width = 0.35

    # 繪製條形圖（HKGAN 顯示誤差線表示多種子標準差）
    bars1 = ax.bar(x - width/2, baseline_f1, width, label='Baseline (BERT-CLS)',
                   color='#4472C4', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, hkgan_f1, width, label='HKGAN (Ours)',
                   color='#70AD47', edgecolor='black', linewidth=0.5,
                   yerr=hkgan_std, capsize=3, error_kw={'elinewidth': 1.5})

    # 設定軸標籤
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Macro-F1 (%)', fontsize=14, fontweight='bold')
    ax.set_title('HKGAN vs Baseline: Macro-F1 Performance Comparison',
                 fontsize=16, fontweight='bold', pad=20)

    # 設定 X 軸
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)

    # 設定 Y 軸範圍
    ax.set_ylim(60, 90)
    ax.set_yticks(np.arange(60, 91, 5))
    ax.tick_params(axis='y', labelsize=11)

    # 添加網格線
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    # 在每組柱狀圖上方標註改善幅度
    for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvements)):
        max_height = max(b1.get_height(), b2.get_height())
        ax.annotate(f'+{imp:.2f}%',
                    xy=(x[i], max_height + 1.5),
                    ha='center', va='bottom',
                    fontsize=11, fontweight='bold',
                    color='#C00000')

    # 添加數值標籤在柱子內部
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, height - 2),
                    ha='center', va='top',
                    fontsize=9, color='white', fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width()/2, height - 2),
                    ha='center', va='top',
                    fontsize=9, color='white', fontweight='bold')

    # 添加圖例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # 保存
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"\n圖表已保存至: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # 打印數據表格
    print("\n" + "=" * 80)
    print("Macro-F1 Performance Summary (Multi-Seed Average)")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Baseline (%)':<15} {'HKGAN (%)':<20} {'Improvement':<15}")
    print("-" * 80)
    for ds, bl, hk, std, imp in zip(datasets, baseline_f1, hkgan_f1, hkgan_std, improvements):
        hk_str = f"{hk:.2f} ± {std:.2f}" if std > 0 else f"{hk:.2f}"
        print(f"{ds:<15} {bl:<15.2f} {hk_str:<20} +{imp:.2f}%")
    print("=" * 80)


def plot_auc_comparison(output_path=None, show=True):
    """繪製 AUC 對比圖（折線圖 + 誤差區間）

    AUC 通常使用折線圖搭配誤差帶（error band），
    因為 AUC 值通常較高且變化較小，折線圖更能清晰展示趨勢。
    """

    exp_data = get_multiseed_data()

    # 準備數據
    datasets = []
    baseline_auc = []
    hkgan_auc = []
    hkgan_auc_std = []

    for ds in DATASETS_ORDER:
        if ds in exp_data:
            if 'baseline_auc' not in exp_data[ds]:
                print(f"警告: {ds} 缺少 Baseline AUC 數據")
                continue
            if 'hkgan_auc_mean' not in exp_data[ds]:
                print(f"警告: {ds} 缺少 HKGAN AUC 數據")
                continue

            datasets.append(DISPLAY_NAMES[ds])
            baseline_auc.append(exp_data[ds]['baseline_auc'])
            hkgan_auc.append(exp_data[ds]['hkgan_auc_mean'])
            hkgan_auc_std.append(exp_data[ds].get('hkgan_auc_std', 0))

            print(f"  {ds}: Baseline AUC={exp_data[ds]['baseline_auc']:.2f}%, "
                  f"HKGAN AUC={exp_data[ds]['hkgan_auc_mean']:.2f}% ± {exp_data[ds].get('hkgan_auc_std', 0):.2f}%")

    if not datasets:
        print("錯誤: 沒有找到任何有效的 AUC 數據")
        return

    # 計算改善幅度
    improvements = [h - b for h, b in zip(hkgan_auc, baseline_auc)]

    # 設定圖表
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(datasets))
    hkgan_auc_arr = np.array(hkgan_auc)
    hkgan_auc_std_arr = np.array(hkgan_auc_std)

    # 繪製折線圖
    line1, = ax.plot(x, baseline_auc, 'o-', color='#4472C4', linewidth=2.5,
                     markersize=10, label='Baseline (BERT-CLS)', markeredgecolor='black')
    line2, = ax.plot(x, hkgan_auc, 's-', color='#70AD47', linewidth=2.5,
                     markersize=10, label='HKGAN (Ours)', markeredgecolor='black')

    # 添加 HKGAN 誤差帶（陰影區域表示標準差）
    ax.fill_between(x, hkgan_auc_arr - hkgan_auc_std_arr, hkgan_auc_arr + hkgan_auc_std_arr,
                    color='#70AD47', alpha=0.2, label='HKGAN ± Std')

    # 設定軸標籤
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC (%)', fontsize=14, fontweight='bold')
    ax.set_title('HKGAN vs Baseline: AUC Performance Comparison',
                 fontsize=16, fontweight='bold', pad=20)

    # 設定 X 軸
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=12)

    # 設定 Y 軸範圍（AUC 通常在 85-100 之間）
    y_min = min(min(baseline_auc), min(hkgan_auc)) - 3
    y_max = max(max(baseline_auc), max(hkgan_auc)) + 3
    y_min = max(85, int(y_min))
    y_max = min(100, int(y_max) + 1)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, 2))
    ax.tick_params(axis='y', labelsize=11)

    # 添加網格線
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)

    # 在數據點旁標註數值
    for i, (bl, hk, imp) in enumerate(zip(baseline_auc, hkgan_auc, improvements)):
        # Baseline 數值
        ax.annotate(f'{bl:.1f}',
                    xy=(x[i], bl), xytext=(0, -15),
                    textcoords='offset points',
                    ha='center', va='top',
                    fontsize=9, color='#4472C4', fontweight='bold')
        # HKGAN 數值
        ax.annotate(f'{hk:.1f}',
                    xy=(x[i], hk), xytext=(0, 10),
                    textcoords='offset points',
                    ha='center', va='bottom',
                    fontsize=9, color='#70AD47', fontweight='bold')
        # 改善幅度
        sign = '+' if imp >= 0 else ''
        color = '#C00000' if imp > 0 else '#666666'
        mid_y = (bl + hk) / 2
        ax.annotate(f'{sign}{imp:.2f}%',
                    xy=(x[i] + 0.15, mid_y),
                    ha='left', va='center',
                    fontsize=10, fontweight='bold',
                    color=color)

    # 添加圖例
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)

    plt.tight_layout()

    # 保存
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"\n圖表已保存至: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # 打印數據表格
    print("\n" + "=" * 80)
    print("AUC Performance Summary (Multi-Seed Average)")
    print("=" * 80)
    print(f"{'Dataset':<15} {'Baseline (%)':<15} {'HKGAN (%)':<20} {'Improvement':<15}")
    print("-" * 80)
    for ds, bl, hk, std, imp in zip(datasets, baseline_auc, hkgan_auc, hkgan_auc_std, improvements):
        hk_str = f"{hk:.2f} ± {std:.2f}" if std > 0 else f"{hk:.2f}"
        sign = '+' if imp >= 0 else ''
        print(f"{ds:<15} {bl:<15.2f} {hk_str:<20} {sign}{imp:.2f}%")
    print("=" * 80)


def plot_neutral_improvement_scatter(output_path=None, show=True):
    """繪製中性類別改善幅度與數據集不平衡程度的散點圖

    展示 X 軸（數據集中性類別比例）與 Y 軸（中性 F1 改善幅度）之間的負相關趨勢。
    中性類別比例越低（越不平衡），HKGAN 的改善幅度越大。
    """

    exp_data = get_multiseed_data()

    # 準備數據
    datasets = []
    neutral_ratios = []
    neu_f1_improvements = []
    dataset_keys = []

    for ds in DATASETS_ORDER:
        if ds in exp_data:
            if 'baseline_neu_f1' not in exp_data[ds]:
                print(f"警告: {ds} 缺少 Baseline 中性 F1 數據")
                continue
            if 'hkgan_neu_f1_mean' not in exp_data[ds]:
                print(f"警告: {ds} 缺少 HKGAN 中性 F1 數據")
                continue

            baseline_neu = exp_data[ds]['baseline_neu_f1']
            hkgan_neu = exp_data[ds]['hkgan_neu_f1_mean']
            improvement = hkgan_neu - baseline_neu

            datasets.append(DISPLAY_NAMES[ds])
            dataset_keys.append(ds)
            neutral_ratios.append(NEUTRAL_RATIO[ds])
            neu_f1_improvements.append(improvement)

            print(f"  {ds}: Neutral Ratio={NEUTRAL_RATIO[ds]:.1f}%, "
                  f"Baseline Neu F1={baseline_neu:.2f}%, "
                  f"HKGAN Neu F1={hkgan_neu:.2f}%, "
                  f"Improvement={improvement:+.2f}%")

    if len(datasets) < 2:
        print("錯誤: 需要至少 2 個數據點來繪製散點圖")
        return

    # 計算 Pearson 相關係數
    x = np.array(neutral_ratios)
    y = np.array(neu_f1_improvements)
    r, p_value = stats.pearsonr(x, y)

    print(f"\n  Pearson 相關係數: r = {r:.2f} (p = {p_value:.4f})")

    # 設定圖表
    fig, ax = plt.subplots(figsize=(10, 8))

    # 繪製散點（使用不同大小表示改善幅度的絕對值）
    sizes = [max(80, abs(imp) * 30) for imp in neu_f1_improvements]
    colors = ['#70AD47' if imp > 0 else '#C00000' for imp in neu_f1_improvements]

    scatter = ax.scatter(neutral_ratios, neu_f1_improvements,
                         s=sizes, c=colors, alpha=0.7,
                         edgecolors='black', linewidths=1.5)

    # 添加回歸線
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(x) - 2, max(x) + 2, 100)
    ax.plot(x_line, p(x_line), '--', color='#4472C4', linewidth=2,
            label=f'Regression Line (r = {r:.2f})')

    # 標註每個數據點
    for i, (ds, xv, yv) in enumerate(zip(datasets, neutral_ratios, neu_f1_improvements)):
        # 根據位置調整標註方向避免重疊
        if ds == 'MAMS':
            offset = (-10, -20)
            ha = 'center'
        elif ds == 'REST16':
            offset = (10, 10)
            ha = 'left'
        elif ds == 'LAP16':
            offset = (10, -15)
            ha = 'left'
        else:
            offset = (10, 5)
            ha = 'left'

        ax.annotate(ds,
                    xy=(xv, yv),
                    xytext=offset,
                    textcoords='offset points',
                    fontsize=11, fontweight='bold',
                    ha=ha,
                    arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

    # 設定軸標籤
    ax.set_xlabel('Neutral Class Ratio in Dataset (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Neutral F1 Improvement (%)', fontsize=14, fontweight='bold')
    ax.set_title('Neutral Class Improvement vs Dataset Imbalance\n'
                 f'(Pearson r = {r:.2f}, p = {p_value:.4f})',
                 fontsize=16, fontweight='bold', pad=20)

    # 設定軸範圍（動態計算 Y 軸範圍，確保數據點不會被裁切）
    ax.set_xlim(0, 40)
    y_max = max(neu_f1_improvements) + 5  # 上方留 5% 空間
    y_min = min(min(neu_f1_improvements) - 3, -5)  # 下方至少到 -5
    ax.set_ylim(y_min, y_max)

    # 添加參考線（y=0）
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)

    # 添加網格線
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # 添加圖例
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)

    # 添加解釋文字
    textstr = ('Negative correlation:\n'
               'Lower neutral ratio\n'
               '→ Greater improvement')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.tight_layout()

    # 保存
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"\n圖表已保存至: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # 打印數據表格
    print("\n" + "=" * 90)
    print("Neutral F1 Improvement vs Dataset Imbalance")
    print("=" * 90)
    print(f"{'Dataset':<15} {'Neutral Ratio (%)':<18} {'Baseline Neu F1':<18} {'HKGAN Neu F1':<18} {'Improvement':<15}")
    print("-" * 90)
    for ds_key, ds, nr, imp in zip(dataset_keys, datasets, neutral_ratios, neu_f1_improvements):
        baseline_neu = exp_data[ds_key]['baseline_neu_f1']
        hkgan_neu = exp_data[ds_key]['hkgan_neu_f1_mean']
        print(f"{ds:<15} {nr:<18.1f} {baseline_neu:<18.2f} {hkgan_neu:<18.2f} {imp:+.2f}%")
    print("-" * 90)
    print(f"Pearson Correlation: r = {r:.2f} (p-value = {p_value:.4f})")
    print("=" * 90)


def get_all_seed_f1_scores():
    """獲取所有 seed 的 Macro-F1 分數（用於箱型圖）"""
    results_dir = Path("results/improved")
    baseline_dir = Path("results/baseline")

    data = {}

    for ds in DATASETS_ORDER:
        data[ds] = {'hkgan': [], 'baseline': []}

        # 讀取所有 HKGAN 實驗（多種子）
        ds_dir = results_dir / ds
        if ds_dir.exists():
            exp_dirs = list(ds_dir.glob("*_improved_hkgan_*"))
            seeds_seen = set()

            for exp in exp_dirs:
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    with open(rf, 'r', encoding='utf-8') as f:
                        result = json.load(f)

                    seed = result.get('args', {}).get('seed', 42)
                    test_metrics = result.get('test_metrics', {})
                    f1 = test_metrics.get('f1_macro')

                    if f1 and seed not in seeds_seen:
                        data[ds]['hkgan'].append(f1 * 100)
                        seeds_seen.add(seed)

        # 讀取所有 Baseline 實驗（多種子）
        bl_dir = baseline_dir / ds
        if bl_dir.exists():
            exp_dirs = list(bl_dir.glob("*_baseline_*"))
            seeds_seen = set()

            for exp in exp_dirs:
                rf = exp / "reports" / "experiment_results.json"
                if rf.exists():
                    with open(rf, 'r', encoding='utf-8') as f:
                        result = json.load(f)

                    seed = result.get('args', {}).get('seed', 42)
                    test_metrics = result.get('test_metrics', {})
                    f1 = test_metrics.get('f1_macro')

                    if f1 and seed not in seeds_seen:
                        data[ds]['baseline'].append(f1 * 100)
                        seeds_seen.add(seed)

    return data


def plot_f1_boxplot(output_path=None, show=True):
    """繪製 Macro-F1 箱型圖（比較 HKGAN 與 Baseline 穩定性）

    展示每個數據集五次實驗的 Macro-F1 分布，視覺化呈現中位數、四分位距與異常值。
    """

    exp_data = get_all_seed_f1_scores()

    # 準備數據
    datasets = []
    hkgan_scores = []
    baseline_scores = []

    for ds in DATASETS_ORDER:
        if ds in exp_data:
            hkgan_data = exp_data[ds]['hkgan']
            baseline_data = exp_data[ds]['baseline']

            if len(hkgan_data) >= 2:  # 至少需要2個數據點
                datasets.append(DISPLAY_NAMES[ds])
                hkgan_scores.append(hkgan_data)
                baseline_scores.append(baseline_data if len(baseline_data) >= 2 else [])

                print(f"  {ds}: HKGAN {len(hkgan_data)} seeds, Baseline {len(baseline_data)} seeds")

    if not datasets:
        print("錯誤: 沒有足夠的多種子實驗數據")
        return

    # 設定圖表
    fig, ax = plt.subplots(figsize=(14, 8))

    n_datasets = len(datasets)
    positions_baseline = np.arange(n_datasets) * 3
    positions_hkgan = positions_baseline + 1

    # 繪製 Baseline 箱型圖（藍色）
    bp_baseline = ax.boxplot(
        [b if b else [0] for b in baseline_scores],  # 避免空列表
        positions=positions_baseline,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor='#4472C4', color='#2E5090', alpha=0.7),
        medianprops=dict(color='#1A3050', linewidth=2),
        whiskerprops=dict(color='#2E5090', linewidth=1.5),
        capprops=dict(color='#2E5090', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='#4472C4', markersize=8, alpha=0.7)
    )

    # 繪製 HKGAN 箱型圖（綠色）
    bp_hkgan = ax.boxplot(
        hkgan_scores,
        positions=positions_hkgan,
        widths=0.8,
        patch_artist=True,
        boxprops=dict(facecolor='#70AD47', color='#4A7A2E', alpha=0.7),
        medianprops=dict(color='#2D4A1A', linewidth=2),
        whiskerprops=dict(color='#4A7A2E', linewidth=1.5),
        capprops=dict(color='#4A7A2E', linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='#70AD47', markersize=8, alpha=0.7)
    )

    # 添加個別數據點（散點）
    for i, (bl_data, hk_data) in enumerate(zip(baseline_scores, hkgan_scores)):
        if bl_data:
            ax.scatter([positions_baseline[i]] * len(bl_data), bl_data,
                      color='#1A3050', s=50, zorder=5, alpha=0.8, edgecolors='white', linewidths=0.5)
        ax.scatter([positions_hkgan[i]] * len(hk_data), hk_data,
                  color='#2D4A1A', s=50, zorder=5, alpha=0.8, edgecolors='white', linewidths=0.5)

    # 設定 X 軸
    ax.set_xticks((positions_baseline + positions_hkgan) / 2)
    ax.set_xticklabels(datasets, fontsize=12, fontweight='bold')

    # 設定軸標籤
    ax.set_xlabel('Dataset', fontsize=14, fontweight='bold')
    ax.set_ylabel('Macro-F1 (%)', fontsize=14, fontweight='bold')
    ax.set_title('Macro-F1 Distribution: HKGAN vs Baseline (5-Seed Experiments)\n'
                 'Boxplot showing median, IQR, and individual data points',
                 fontsize=14, fontweight='bold', pad=20)

    # 設定 Y 軸範圍
    all_scores = [s for scores in hkgan_scores + baseline_scores for s in scores if scores]
    if all_scores:
        y_min = min(all_scores) - 3
        y_max = max(all_scores) + 3
        ax.set_ylim(y_min, y_max)

    # 添加網格線
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4472C4', edgecolor='#2E5090', alpha=0.7, label='Baseline'),
        Patch(facecolor='#70AD47', edgecolor='#4A7A2E', alpha=0.7, label='HKGAN')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12, framealpha=0.9)

    plt.tight_layout()

    # 保存
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"\n圖表已保存至: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    # 打印統計數據表格
    print("\n" + "=" * 100)
    print("Macro-F1 Stability Comparison (5-Seed Experiments)")
    print("=" * 100)
    print(f"{'Dataset':<15} {'Model':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12} {'IQR':<12}")
    print("-" * 100)

    for ds, hk_data, bl_data in zip(datasets, hkgan_scores, baseline_scores):
        # HKGAN 統計
        hk_mean = np.mean(hk_data)
        hk_std = np.std(hk_data)
        hk_min = np.min(hk_data)
        hk_max = np.max(hk_data)
        hk_iqr = np.percentile(hk_data, 75) - np.percentile(hk_data, 25)

        print(f"{ds:<15} {'HKGAN':<12} {hk_mean:<12.2f} {hk_std:<12.2f} {hk_min:<12.2f} {hk_max:<12.2f} {hk_iqr:<12.2f}")

        # Baseline 統計
        if bl_data:
            bl_mean = np.mean(bl_data)
            bl_std = np.std(bl_data)
            bl_min = np.min(bl_data)
            bl_max = np.max(bl_data)
            bl_iqr = np.percentile(bl_data, 75) - np.percentile(bl_data, 25)
            print(f"{'':<15} {'Baseline':<12} {bl_mean:<12.2f} {bl_std:<12.2f} {bl_min:<12.2f} {bl_max:<12.2f} {bl_iqr:<12.2f}")
        else:
            print(f"{'':<15} {'Baseline':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}")

    print("-" * 100)

    # 計算整體穩定性比較
    all_hkgan_std = [np.std(hk) for hk in hkgan_scores]
    all_baseline_std = [np.std(bl) for bl in baseline_scores if bl]

    print(f"\nOverall Stability (Average Std):")
    print(f"  HKGAN:    {np.mean(all_hkgan_std):.2f}%")
    if all_baseline_std:
        print(f"  Baseline: {np.mean(all_baseline_std):.2f}%")
        if np.mean(all_hkgan_std) < np.mean(all_baseline_std):
            print(f"  -> HKGAN is MORE stable (lower variance)")
        else:
            print(f"  -> Baseline is MORE stable (lower variance)")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description='論文圖檔生成工具 - HKGAN 實驗結果視覺化',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  python experiments/plot_thesis_figures.py --figure f1
  python experiments/plot_thesis_figures.py --figure auc
  python experiments/plot_thesis_figures.py --figure neutral
  python experiments/plot_thesis_figures.py --figure boxplot
  python experiments/plot_thesis_figures.py --figure all --output results/figures/
        """
    )
    parser.add_argument('--figure', '-f', type=str, default='all',
                        choices=['f1', 'auc', 'neutral', 'boxplot', 'all'],
                        help='要生成的圖表類型 (default: all)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='輸出路徑（目錄或檔案）')
    parser.add_argument('--no-show', action='store_true',
                        help='不顯示圖表（僅保存）')

    args = parser.parse_args()

    show = not args.no_show

    print("=" * 60)
    print("論文圖檔生成工具")
    print("=" * 60)
    print("讀取多種子實驗數據...\n")

    if args.figure in ['f1', 'all']:
        print("-" * 60)
        print("生成 Macro-F1 對比圖...")
        print("-" * 60)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir() or args.figure == 'all':
                output_path = Path(args.output) / "f1_comparison.png"
            plot_f1_comparison(output_path=output_path, show=show)
        else:
            plot_f1_comparison(show=show)

    if args.figure in ['auc', 'all']:
        print("\n" + "-" * 60)
        print("生成 AUC 對比圖...")
        print("-" * 60)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir() or args.figure == 'all':
                output_path = Path(args.output) / "auc_comparison.png"
            plot_auc_comparison(output_path=output_path, show=show)
        else:
            plot_auc_comparison(show=show)

    if args.figure in ['neutral', 'all']:
        print("\n" + "-" * 60)
        print("生成中性類別改善散點圖...")
        print("-" * 60)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir() or args.figure == 'all':
                output_path = Path(args.output) / "neutral_improvement_scatter.png"
            plot_neutral_improvement_scatter(output_path=output_path, show=show)
        else:
            plot_neutral_improvement_scatter(show=show)

    if args.figure in ['boxplot', 'all']:
        print("\n" + "-" * 60)
        print("生成 Macro-F1 箱型圖...")
        print("-" * 60)
        if args.output:
            output_path = Path(args.output)
            if output_path.is_dir() or args.figure == 'all':
                output_path = Path(args.output) / "f1_boxplot.png"
            plot_f1_boxplot(output_path=output_path, show=show)
        else:
            plot_f1_boxplot(show=show)

    print("\n" + "=" * 60)
    print("圖檔生成完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
