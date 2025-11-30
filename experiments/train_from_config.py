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


def config_to_args(config, dataset=None):
    """
    將 YAML 配置轉換為命令行參數

    Args:
        config: YAML 配置字典
        dataset: 資料集名稱，用於選擇對應的 bert_model
    """
    args = []

    # 自動選擇模式
    if config.get('auto_select'):
        args.append('--auto_select')

    # 模型配置
    if 'model' in config:
        model_cfg = config['model']

        if model_cfg.get('baseline'):
            args.extend(['--baseline', model_cfg['baseline']])

        if model_cfg.get('improved'):
            args.extend(['--improved', model_cfg['improved']])

        if model_cfg.get('bert_model'):
            bert_model = model_cfg['bert_model']
            # 支援對應表格式：根據 dataset 選擇對應的模型
            if isinstance(bert_model, dict):
                if dataset and dataset in bert_model:
                    bert_model = bert_model[dataset]
                else:
                    bert_model = bert_model.get('default', 'bert-base-uncased')
                print(f"[Config] 根據資料集 '{dataset}' 選擇 bert_model: {bert_model}")
            args.extend(['--bert_model', bert_model])

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

        # Attention 參數
        if 'num_attention_heads' in model_cfg:
            args.extend(['--num_attention_heads', str(model_cfg['num_attention_heads'])])

        # HKGAN 參數
        if 'gat_heads' in model_cfg:
            args.extend(['--gat_heads', str(model_cfg['gat_heads'])])

        if 'gat_layers' in model_cfg:
            args.extend(['--gat_layers', str(model_cfg['gat_layers'])])

        if 'knowledge_weight' in model_cfg:
            args.extend(['--knowledge_weight', str(model_cfg['knowledge_weight'])])

        if model_cfg.get('use_senticnet'):
            args.append('--use_senticnet')

        # HKGAN v2.0 新增：Neutral 識別改進
        if 'use_confidence_gate' in model_cfg:
            if model_cfg['use_confidence_gate']:
                args.append('--use_confidence_gate')
            else:
                args.append('--no_confidence_gate')

        if 'domain' in model_cfg and model_cfg['domain']:
            args.extend(['--domain', model_cfg['domain']])

    # 數據配置
    if 'data' in config:
        data_cfg = config['data']

        # 數據增強配置
        if data_cfg.get('use_augmented'):
            args.append('--use_augmented')

        if data_cfg.get('use_self_training'):
            args.append('--use_self_training')

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
            cw = train_cfg['class_weights']
            if cw == 'auto':
                # 動態計算 class weights
                args.append('--auto_class_weights')
            elif isinstance(cw, list):
                weights = [str(w) for w in cw]
                args.extend(['--class_weights'] + weights)

        # Virtual aspect
        if 'virtual_weight' in train_cfg:
            args.extend(['--virtual_weight', str(train_cfg['virtual_weight'])])

        # 隨機種子
        if 'seed' in train_cfg:
            args.extend(['--seed', str(train_cfg['seed'])])

        # 對比學習參數
        if 'contrastive_weight' in train_cfg:
            args.extend(['--contrastive_weight', str(train_cfg['contrastive_weight'])])

        if 'contrastive_temperature' in train_cfg:
            args.extend(['--contrastive_temperature', str(train_cfg['contrastive_temperature'])])

        # Layer-wise Learning Rate Decay (LLRD)
        if train_cfg.get('use_llrd'):
            args.append('--use_llrd')

        if 'llrd_decay' in train_cfg:
            args.extend(['--llrd_decay', str(train_cfg['llrd_decay'])])

        # 非對稱 Logit 調整 (推理時)
        # 支援對應表格式：根據 dataset 選擇對應的值
        if 'neutral_boost' in train_cfg:
            neutral_boost = train_cfg['neutral_boost']
            if isinstance(neutral_boost, dict):
                if dataset and dataset in neutral_boost:
                    neutral_boost = neutral_boost[dataset]
                else:
                    neutral_boost = neutral_boost.get('default', 0.0)
                print(f"[Config] 根據資料集 '{dataset}' 選擇 neutral_boost: {neutral_boost}")
            args.extend(['--neutral_boost', str(neutral_boost)])

        if 'neg_suppress' in train_cfg:
            neg_suppress = train_cfg['neg_suppress']
            if isinstance(neg_suppress, dict):
                if dataset and dataset in neg_suppress:
                    neg_suppress = neg_suppress[dataset]
                else:
                    neg_suppress = neg_suppress.get('default', 0.0)
                print(f"[Config] 根據資料集 '{dataset}' 選擇 neg_suppress: {neg_suppress}")
            args.extend(['--neg_suppress', str(neg_suppress)])

        if 'pos_suppress' in train_cfg:
            pos_suppress = train_cfg['pos_suppress']
            if isinstance(pos_suppress, dict):
                if dataset and dataset in pos_suppress:
                    pos_suppress = pos_suppress[dataset]
                else:
                    pos_suppress = pos_suppress.get('default', 0.0)
                print(f"[Config] 根據資料集 '{dataset}' 選擇 pos_suppress: {pos_suppress}")
            args.extend(['--pos_suppress', str(pos_suppress)])

    # 知識蒸餾配置
    if 'distillation' in config:
        distill_cfg = config['distillation']

        if distill_cfg.get('enabled'):
            # 啟用蒸餾時，loss_type 設為 distill
            if '--loss_type' not in args:
                args.extend(['--loss_type', 'distill'])

        if 'alpha' in distill_cfg:
            args.extend(['--distill_alpha', str(distill_cfg['alpha'])])

        if 'temperature' in distill_cfg:
            args.extend(['--distill_temperature', str(distill_cfg['temperature'])])

        if 'soft_labels_file' in distill_cfg:
            args.extend(['--soft_labels_file', distill_cfg['soft_labels_file']])

        # 蒸餾配置中的 focal 設定
        if 'use_focal' in distill_cfg and distill_cfg['use_focal']:
            if 'focal_gamma' in distill_cfg:
                args.extend(['--focal_gamma', str(distill_cfg['focal_gamma'])])

    return args


def main():
    parser = argparse.ArgumentParser(description='從 YAML 配置文件訓練模型')
    parser.add_argument('--config', type=str, required=True,
                        help='YAML 配置文件路徑')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['restaurants', 'laptops', 'mams',
                                 'rest16', 'lap16',
                                 'memd_books', 'memd_clothing', 'memd_hotel',
                                 'memd_laptop', 'memd_restaurant'],
                        help='數據集選擇 (restaurants, laptops, mams, rest16, lap16, 或 memd_* 系列)')
    parser.add_argument('--override', nargs='*', default=[],
                        help='覆蓋配置的額外參數，例如: --override --epochs 50 --lr 3e-5')

    args, unknown = parser.parse_known_args()

    # 載入配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"錯誤: 配置文件不存在: {config_path}")
        return

    config = load_config(config_path)

    # 轉換為命令行參數（傳入 dataset 以選擇對應的 bert_model）
    train_args = config_to_args(config, dataset=args.dataset)

    # 添加 dataset 參數（必須）
    train_args.extend(['--dataset', args.dataset])

    # 添加覆蓋參數
    if args.override:
        train_args.extend(args.override)

    # 調用訓練主函數
    sys.argv = ['train_multiaspect.py'] + train_args
    train_main()


if __name__ == "__main__":
    main()
