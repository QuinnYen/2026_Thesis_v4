"""
模型自動選擇器

根據數據集特徵自動選擇最佳模型架構:
- Hierarchical BERT: 適合單面向為主的數據集
- IARN: 適合多面向為主的數據集
"""

from typing import Dict, Any, Tuple


# 選擇閾值
MULTI_ASPECT_THRESHOLD = 0.5  # 多面向比例閾值


def select_model(stats: Dict[str, Any]) -> Tuple[str, str]:
    """
    根據數據集統計選擇最佳模型

    參數:
        stats: 數據集統計資訊 (來自 dataset_analyzer.analyze_dataset)

    返回:
        (model_type, reason): 模型類型和選擇原因
    """
    multi_ratio = stats.get('multi_aspect_ratio', 0.0)
    avg_aspects = stats.get('avg_aspects', 1.0)

    # 決策邏輯
    if multi_ratio > MULTI_ASPECT_THRESHOLD:
        model_type = 'iarn'
        reason = (
            f"Multi-aspect ratio ({multi_ratio*100:.1f}%) > {MULTI_ASPECT_THRESHOLD*100:.0f}% threshold. "
            f"IARN's Aspect-to-Aspect Attention can model inter-aspect dependencies."
        )
    else:
        model_type = 'hierarchical'
        reason = (
            f"Multi-aspect ratio ({multi_ratio*100:.1f}%) <= {MULTI_ASPECT_THRESHOLD*100:.0f}% threshold. "
            f"Hierarchical BERT's multi-level features are more effective for single-aspect dominated data."
        )

    return model_type, reason


def get_model_config(model_type: str, dataset: str) -> Dict[str, Any]:
    """
    獲取模型的推薦配置

    參數:
        model_type: 'hierarchical' 或 'iarn'
        dataset: 數據集名稱

    返回:
        配置字典
    """
    # 基礎配置
    base_config = {
        'bert_model': 'bert-base-uncased',
        'hidden_dim': 768,
        'freeze_bert': False,
        'lr': 2e-5,
        'weight_decay': 0.05,
        'grad_clip': 1.0,
        'use_scheduler': True,
        'warmup_ratio': 0.1,
        'loss_type': 'focal',
    }

    # 根據數據集調整
    if dataset == 'mams':
        # MAMS 相對平衡，可充分訓練
        base_config.update({
            'epochs': 40,
            'patience': 12,
            'focal_gamma': 2.0,
            'class_weights': [1.0, 3.0, 1.0],
        })
    else:
        # SemEval (restaurants/laptops) 不平衡，提早停止
        base_config.update({
            'epochs': 25,
            'patience': 8,
            'focal_gamma': 2.5,
            'class_weights': [1.0, 5.0, 1.0],
        })

    # 根據模型類型調整
    if model_type == 'iarn':
        base_config.update({
            'dropout': 0.3,
            'num_attention_heads': 4,
        })
    else:  # hierarchical
        base_config.update({
            'dropout': 0.4,
        })

    return base_config


def print_selection_result(model_type: str, reason: str) -> None:
    """
    打印模型選擇結果
    """
    model_names = {
        'hierarchical': 'Hierarchical BERT',
        'iarn': 'IARN (Inter-Aspect Relation Network)'
    }

    print(f"\n{'='*60}")
    print(f"Auto Model Selection")
    print(f"{'='*60}")
    print(f"  Selected:  {model_names.get(model_type, model_type)}")
    print(f"  Reason:    {reason}")
    print(f"{'='*60}\n")
