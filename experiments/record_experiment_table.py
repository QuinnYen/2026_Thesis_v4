#!/usr/bin/env python3
"""
實驗結果表格記錄腳本

功能:
1. 從所有實驗資料夾中提取結果（包括 baseline）
2. 生成 Markdown 表格格式
3. 方便直接比較所有實驗結果

使用方式:
    # 記錄所有實驗（包括 baseline）
    python experiments/record_experiment_table.py

    # 只記錄完整模型實驗
    python experiments/record_experiment_table.py --only-full

    # 只記錄 baseline
    python experiments/record_experiment_table.py --only-baseline

    # 指定輸出檔案
    python experiments/record_experiment_table.py --output results/實驗結果報告.md
"""

import json
import os
import argparse
import glob
from datetime import datetime
from pathlib import Path


def find_all_experiments(base_dir='results'):
    """找到所有實驗資料夾"""
    experiments = []

    # 查找完整模型實驗 (results/experiments/)
    exp_dirs = glob.glob(f"{base_dir}/experiments/*/")
    for exp_dir in exp_dirs:
        exp_dir = exp_dir.rstrip('/')
        results_path = os.path.join(exp_dir, 'reports', 'experiment_results.json')
        if os.path.exists(results_path):
            experiments.append({
                'path': exp_dir,
                'type': 'full',
                'name': os.path.basename(exp_dir)
            })

    # 查找 baseline 實驗 (results/baseline/{dataset}/)
    baseline_dirs = glob.glob(f"{base_dir}/baseline/*/*/")
    for baseline_dir in baseline_dirs:
        baseline_dir = baseline_dir.rstrip('/')
        results_path = os.path.join(baseline_dir, 'reports', 'experiment_results.json')
        if os.path.exists(results_path):
            experiments.append({
                'path': baseline_dir,
                'type': 'baseline',
                'name': os.path.basename(baseline_dir)
            })

    return experiments


def load_experiment_results(exp_path):
    """載入實驗結果 JSON"""
    results_path = os.path.join(exp_path, 'reports', 'experiment_results.json')
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def infer_dataset_from_path(exp_path):
    """從實驗路徑或名稱推斷數據集（僅用於舊實驗的備用方案）"""
    path_lower = exp_path.lower()

    # 檢查路徑中是否包含數據集名稱
    if 'restaurant' in path_lower:
        return 'restaurants'
    elif 'laptop' in path_lower:
        return 'laptops'

    # 如果無法推斷，返回 None
    return None


def extract_metrics(results, exp_path=None):
    """提取關鍵指標"""
    args = results.get('args', {})
    test_metrics = results.get('test_metrics', {})
    best_val_f1 = results.get('best_val_f1', None)

    # 提取各類別 F1
    f1_per_class = test_metrics.get('f1_per_class', [])
    f1_neg = f1_per_class[0] if len(f1_per_class) > 0 else None
    f1_neu = f1_per_class[1] if len(f1_per_class) > 1 else None
    f1_pos = f1_per_class[2] if len(f1_per_class) > 2 else None

    # 提取 Gate 統計
    gate_stats = test_metrics.get('gate_stats', None)

    # 提取數據集，優先順序: args['dataset'] -> 路徑推斷
    # 注意：新實驗訓練時會自動記錄 dataset 參數，只有舊實驗才需要推斷
    dataset = args.get('dataset', None)
    if not dataset and exp_path:
        # 嘗試從路徑推斷（舊實驗的備用方案）
        dataset = infer_dataset_from_path(exp_path)
    if not dataset:
        dataset = 'Unknown'

    return {
        # 配置
        'dataset': dataset,
        'baseline': args.get('baseline', None),
        'use_pmac': args.get('use_pmac', False),
        'use_iarm': args.get('use_iarm', False),
        'batch_size': args.get('batch_size', None),
        'epochs': args.get('epochs', None),
        'lr': args.get('lr', None),
        'dropout': args.get('dropout', None),
        'loss_type': args.get('loss_type', 'CE'),
        'focal_gamma': args.get('focal_gamma', None),
        'class_weights': args.get('class_weights', None),
        'gate_bias_init': args.get('gate_bias_init', None),

        # 結果
        'best_val_f1': best_val_f1,
        'test_acc': test_metrics.get('accuracy', None),
        'test_f1': test_metrics.get('f1_macro', None),
        'test_precision': test_metrics.get('precision', None),
        'test_recall': test_metrics.get('recall', None),
        'f1_neg': f1_neg,
        'f1_neu': f1_neu,
        'f1_pos': f1_pos,
        'auc_macro': test_metrics.get('auc_macro', None),
        'auc_weighted': test_metrics.get('auc_weighted', None),

        # Gate 統計
        'gate_mean': gate_stats.get('mean', None) if gate_stats else None,
        'gate_sparsity': gate_stats.get('sparsity', None) if gate_stats else None,
        'gate_activation': gate_stats.get('activation_rate', None) if gate_stats else None,
    }


def get_model_name(exp_info, metrics):
    """生成易讀的模型名稱"""
    baseline = metrics['baseline']
    use_pmac = metrics['use_pmac']
    use_iarm = metrics['use_iarm']

    if baseline == 'bert_only':
        return 'BERT Only'
    elif baseline == 'bert_aaha':
        return 'BERT + AAHA'
    elif baseline == 'bert_mean':
        return 'BERT + Mean'
    elif use_pmac and use_iarm:
        return 'Full (PMAC + IARM)'
    elif use_pmac:
        return 'PMAC Only'
    elif use_iarm:
        return 'IARM Only'
    else:
        return 'BERT + AAHA (No PMAC/IARM)'


def format_value(value, format_type='float'):
    """格式化數值"""
    if value is None or value == 'N/A':
        return '-'

    if format_type == 'float':
        if isinstance(value, (int, float)):
            return f'{value:.4f}'
        return str(value)
    elif format_type == 'lr':  # 學習率使用科學記號
        if isinstance(value, (int, float)):
            return f'{value:.0e}'
        return str(value)
    elif format_type == 'percent':
        if isinstance(value, (int, float)):
            return f'{value:.2%}'
        return str(value)
    elif format_type == 'int':
        if isinstance(value, (int, float)):
            return str(int(value))
        return str(value)
    else:
        return str(value)


def generate_comparison_table(experiments_data):
    """生成比較表格（Markdown格式）"""
    lines = []

    # 標題
    lines.append("# 實驗結果比較表")
    lines.append(f"\n生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 按 dataset 分組
    datasets = {}
    for exp in experiments_data:
        dataset = exp['metrics']['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(exp)

    # 為每個 dataset 生成表格
    for dataset, exps in sorted(datasets.items()):
        lines.append(f"\n## {dataset.upper()} 數據集\n")

        # 主要結果表格
        lines.append("### 主要結果\n")
        lines.append("| 模型 | Test Acc | Test F1 | Val F1 (Best) | AUC (macro) | AUC (weighted) | Neg F1 | Neu F1 | Pos F1 | Precision | Recall |")
        lines.append("|------|----------|---------|---------------|-------------|----------------|--------|--------|--------|-----------|--------|")

        # 排序：baseline 在前，完整模型在後
        exps_sorted = sorted(exps, key=lambda x: (
            0 if x['type'] == 'baseline' else 1,  # baseline 優先
            -x['metrics']['test_f1'] if x['metrics']['test_f1'] else 0  # F1 降序
        ))

        for exp in exps_sorted:
            m = exp['metrics']
            model_name = get_model_name(exp, m)

            row = [
                model_name,
                format_value(m['test_acc'], 'float'),
                format_value(m['test_f1'], 'float'),
                format_value(m['best_val_f1'], 'float'),
                format_value(m['auc_macro'], 'float'),
                format_value(m['auc_weighted'], 'float'),
                format_value(m['f1_neg'], 'float'),
                format_value(m['f1_neu'], 'float'),
                format_value(m['f1_pos'], 'float'),
                format_value(m['test_precision'], 'float'),
                format_value(m['test_recall'], 'float'),
            ]
            lines.append("| " + " | ".join(row) + " |")

        # Gate 統計表格（只顯示有 PMAC 的模型）
        pmac_exps = [exp for exp in exps if exp['metrics']['use_pmac']]
        if pmac_exps:
            lines.append("\n### Gate 統計 (PMAC 模型)\n")
            lines.append("| 模型 | Gate Mean | Sparsity (<0.1) | Activation (>0.5) | Gate Bias Init |")
            lines.append("|------|-----------|-----------------|-------------------|----------------|")

            for exp in pmac_exps:
                m = exp['metrics']
                model_name = get_model_name(exp, m)

                row = [
                    model_name,
                    format_value(m['gate_mean'], 'float'),
                    format_value(m['gate_sparsity'], 'percent'),
                    format_value(m['gate_activation'], 'percent'),
                    format_value(m['gate_bias_init'], 'float'),
                ]
                lines.append("| " + " | ".join(row) + " |")

        # 配置詳情表格
        lines.append("\n### 訓練配置\n")
        lines.append("| 模型 | Batch Size | Epochs | LR | Dropout | Loss Type | Focal γ | Class Weights |")
        lines.append("|------|------------|--------|----|---------|-----------|---------| --------------|")

        for exp in exps_sorted:
            m = exp['metrics']
            model_name = get_model_name(exp, m)

            class_weights_str = str(m['class_weights']) if m['class_weights'] else '-'

            row = [
                model_name,
                format_value(m['batch_size'], 'int'),
                format_value(m['epochs'], 'int'),
                format_value(m['lr'], 'lr'),  # 使用科學記號格式
                format_value(m['dropout'], 'float'),
                m['loss_type'] if m['loss_type'] else '-',
                format_value(m['focal_gamma'], 'float'),
                class_weights_str,
            ]
            lines.append("| " + " | ".join(row) + " |")

        lines.append("\n---\n")
        lines.append('<div style="page-break-after: always;"></div>\n')

    # 最佳模型總結
    lines.append("\n## 最佳模型總結\n")

    for dataset in sorted(datasets.keys()):
        exps = datasets[dataset]

        # 找最佳 Test F1
        best_exp = max(exps, key=lambda x: x['metrics']['test_f1'] if x['metrics']['test_f1'] else 0)
        best_model_name = get_model_name(best_exp, best_exp['metrics'])
        best_f1 = best_exp['metrics']['test_f1']

        lines.append(f"**{dataset.upper()}**: {best_model_name} (Test F1: {format_value(best_f1, 'float')})")

    lines.append("")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='生成實驗結果比較表格')
    parser.add_argument('--output', type=str, default='results/實驗結果報告.md',
                        help='輸出檔案名稱 (預設: results/實驗結果報告.md)')
    parser.add_argument('--base-dir', type=str, default='results',
                        help='實驗結果基礎路徑 (預設: results)')
    parser.add_argument('--only-full', action='store_true',
                        help='只記錄完整模型實驗')
    parser.add_argument('--only-baseline', action='store_true',
                        help='只記錄 baseline 實驗')

    args = parser.parse_args()

    try:
        print("正在搜尋實驗資料夾...")
        experiments = find_all_experiments(args.base_dir)

        # 過濾
        if args.only_full:
            experiments = [exp for exp in experiments if exp['type'] == 'full']
            print(f"找到 {len(experiments)} 個完整模型實驗")
        elif args.only_baseline:
            experiments = [exp for exp in experiments if exp['type'] == 'baseline']
            print(f"找到 {len(experiments)} 個 baseline 實驗")
        else:
            num_full = len([exp for exp in experiments if exp['type'] == 'full'])
            num_baseline = len([exp for exp in experiments if exp['type'] == 'baseline'])
            print(f"找到 {num_full} 個完整模型實驗, {num_baseline} 個 baseline 實驗")

        if not experiments:
            print("沒有找到任何實驗結果")
            return 1

        # 載入所有結果
        print("載入實驗結果...")
        experiments_data = []
        for exp in experiments:
            try:
                results = load_experiment_results(exp['path'])
                metrics = extract_metrics(results, exp_path=exp['path'])  # 傳入路徑用於推斷數據集
                experiments_data.append({
                    'path': exp['path'],
                    'name': exp['name'],
                    'type': exp['type'],
                    'metrics': metrics
                })
            except Exception as e:
                print(f"警告: 無法載入 {exp['path']}: {str(e)}")
                continue

        print(f"成功載入 {len(experiments_data)} 個實驗結果")

        # 生成表格
        print("生成比較表格...")
        table_text = generate_comparison_table(experiments_data)

        # 儲存
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(table_text)

        print(f"\n[SAVED] 表格已儲存到: {args.output}")
        print(f"\n預覽:\n")
        print(table_text[:1000] + "..." if len(table_text) > 1000 else table_text)

    except Exception as e:
        print(f"錯誤: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
