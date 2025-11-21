"""
從 YAML 配置文件訓練模型

使用方法:
    python experiments/train_from_config.py --config configs/full_model_optimized.yaml
    python experiments/train_from_config.py --config configs/baseline_bert_only.yaml
    python experiments/train_from_config.py --config configs/pmac_only.yaml

優勢:
    - 配置集中管理，易於追蹤
    - 避免命令行參數過長
    - 便於實驗重現
"""

import argparse
import yaml
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_multiaspect import main as train_main


def load_config(config_path):
    """載入 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """將 YAML 配置轉換為命令行參數"""
    args = []

    # 模型配置
    if 'model' in config:
        model_cfg = config['model']

        if model_cfg.get('baseline'):
            args.extend(['--baseline', model_cfg['baseline']])

        if model_cfg.get('improved'):
            args.extend(['--improved', model_cfg['improved']])

        if model_cfg.get('bert_model'):
            args.extend(['--bert_model', model_cfg['bert_model']])

        if model_cfg.get('freeze_bert'):
            args.append('--freeze_bert')

        if 'hidden_dim' in model_cfg:
            args.extend(['--hidden_dim', str(model_cfg['hidden_dim'])])

        if 'dropout' in model_cfg:
            args.extend(['--dropout', str(model_cfg['dropout'])])

        # PMAC 參數
        if model_cfg.get('use_pmac'):
            args.append('--use_pmac')

            if 'gate_bias_init' in model_cfg:
                args.extend(['--gate_bias_init', str(model_cfg['gate_bias_init'])])

            if 'gate_weight_gain' in model_cfg:
                args.extend(['--gate_weight_gain', str(model_cfg['gate_weight_gain'])])

            if 'gate_sparsity_weight' in model_cfg:
                args.extend(['--gate_sparsity_weight', str(model_cfg['gate_sparsity_weight'])])

            if 'gate_sparsity_type' in model_cfg:
                args.extend(['--gate_sparsity_type', model_cfg['gate_sparsity_type']])

        # IARM 參數
        if model_cfg.get('use_iarm'):
            args.append('--use_iarm')

            if 'iarm_heads' in model_cfg:
                args.extend(['--iarm_heads', str(model_cfg['iarm_heads'])])

            if 'iarm_layers' in model_cfg:
                args.extend(['--iarm_layers', str(model_cfg['iarm_layers'])])

    # 數據配置
    if 'data' in config:
        data_cfg = config['data']

        # 數據增強配置
        if data_cfg.get('use_augmented'):
            args.append('--use_augmented')

        if 'augmented_dir' in data_cfg:
            args.extend(['--augmented_dir', data_cfg['augmented_dir']])

        if 'min_aspects' in data_cfg:
            args.extend(['--min_aspects', str(data_cfg['min_aspects'])])

        if 'max_aspects' in data_cfg:
            args.extend(['--max_aspects', str(data_cfg['max_aspects'])])

        # include_single_aspect 需要明確處理 True/False
        if 'include_single_aspect' in data_cfg:
            if data_cfg['include_single_aspect']:
                args.append('--include_single_aspect')
            else:
                args.append('--no_include_single_aspect')  # 明確禁用

        if 'virtual_aspect_mode' in data_cfg:
            args.extend(['--virtual_aspect_mode', data_cfg['virtual_aspect_mode']])

        if 'max_text_len' in data_cfg:
            args.extend(['--max_text_len', str(data_cfg['max_text_len'])])

        if 'max_aspect_len' in data_cfg:
            args.extend(['--max_aspect_len', str(data_cfg['max_aspect_len'])])

    # 訓練配置
    if 'training' in config:
        train_cfg = config['training']

        if 'batch_size' in train_cfg:
            args.extend(['--batch_size', str(train_cfg['batch_size'])])

        if 'accumulation_steps' in train_cfg:
            args.extend(['--accumulation_steps', str(train_cfg['accumulation_steps'])])

        if 'epochs' in train_cfg:
            args.extend(['--epochs', str(train_cfg['epochs'])])

        if 'lr' in train_cfg:
            args.extend(['--lr', str(train_cfg['lr'])])

        if 'weight_decay' in train_cfg:
            args.extend(['--weight_decay', str(train_cfg['weight_decay'])])

        if 'grad_clip' in train_cfg:
            args.extend(['--grad_clip', str(train_cfg['grad_clip'])])

        if 'patience' in train_cfg:
            args.extend(['--patience', str(train_cfg['patience'])])

        if train_cfg.get('use_scheduler'):
            args.append('--use_scheduler')

        if 'warmup_ratio' in train_cfg:
            args.extend(['--warmup_ratio', str(train_cfg['warmup_ratio'])])

        # 損失函數
        if 'loss_type' in train_cfg:
            args.extend(['--loss_type', train_cfg['loss_type']])

        if 'focal_gamma' in train_cfg:
            args.extend(['--focal_gamma', str(train_cfg['focal_gamma'])])

        if 'label_smoothing' in train_cfg:
            args.extend(['--label_smoothing', str(train_cfg['label_smoothing'])])

        if 'class_weights' in train_cfg:
            weights = [str(w) for w in train_cfg['class_weights']]
            args.extend(['--class_weights'] + weights)

        # Virtual aspect
        if 'virtual_weight' in train_cfg:
            args.extend(['--virtual_weight', str(train_cfg['virtual_weight'])])

        # 隨機種子
        if 'seed' in train_cfg:
            args.extend(['--seed', str(train_cfg['seed'])])

    return args


def main():
    parser = argparse.ArgumentParser(description='從 YAML 配置文件訓練模型')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML 配置文件路徑')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['restaurants', 'laptops', 'mams'],
                        help='數據集選擇 (restaurants, laptops, 或 mams)')
    parser.add_argument('--override', nargs='*', default=[],
                        help='覆蓋配置的額外參數，例如: --override --epochs 50 --lr 3e-5')

    args, unknown = parser.parse_known_args()

    # 載入配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"錯誤: 配置文件不存在: {config_path}")
        return

    print(f"\n{'='*80}")
    print(f"載入配置文件: {config_path}")
    print(f"數據集: {args.dataset.upper()}")
    print(f"{'='*80}\n")

    config = load_config(config_path)

    # 顯示配置
    print("實驗配置:")
    print(yaml.dump(config, default_flow_style=False, allow_unicode=True))
    print(f"{'='*80}\n")

    # 轉換為命令行參數
    train_args = config_to_args(config)

    # 添加 dataset 參數（必須）
    train_args.extend(['--dataset', args.dataset])

    # 添加覆蓋參數
    if args.override:
        print(f"覆蓋參數: {' '.join(args.override)}\n")
        train_args.extend(args.override)

    # 調用訓練主函數
    sys.argv = ['train_multiaspect.py'] + train_args

    print(f"執行訓練命令:")
    print(f"  python experiments/train_multiaspect.py {' '.join(train_args)}")
    print(f"\n{'='*80}\n")

    train_main()


if __name__ == "__main__":
    main()
