"""
數據集特徵分析器

分析數據集的 multi-aspect 特性，用於自動選擇最佳模型架構
"""

from typing import List, Dict, Any
from collections import Counter
import numpy as np


def analyze_dataset(samples: List[Any]) -> Dict[str, Any]:
    """
    分析數據集的 multi-aspect 特性

    參數:
        samples: MultiAspectSample 列表

    返回:
        統計資訊字典
    """
    if not samples:
        return {
            'total_samples': 0,
            'multi_aspect_ratio': 0.0,
            'avg_aspects': 0.0,
            'recommended_model': 'hierarchical'
        }

    total = len(samples)

    # 計算每個樣本的 aspect 數量
    aspect_counts = []
    for s in samples:
        # 計算非虛擬的 aspect 數量
        if hasattr(s, 'is_virtual'):
            real_aspects = sum(1 for v in s.is_virtual if not v)
        else:
            real_aspects = s.num_aspects
        aspect_counts.append(real_aspects)

    # 統計多面向樣本
    multi_aspect_count = sum(1 for c in aspect_counts if c > 1)
    single_aspect_count = sum(1 for c in aspect_counts if c == 1)

    # 情感分布
    all_labels = []
    for s in samples:
        all_labels.extend(s.labels)

    label_dist = Counter(all_labels)
    total_labels = sum(label_dist.values())

    stats = {
        'total_samples': total,
        'multi_aspect_count': multi_aspect_count,
        'single_aspect_count': single_aspect_count,
        'multi_aspect_ratio': multi_aspect_count / total if total > 0 else 0.0,
        'avg_aspects': np.mean(aspect_counts) if aspect_counts else 0.0,
        'max_aspects': max(aspect_counts) if aspect_counts else 0,
        'min_aspects': min(aspect_counts) if aspect_counts else 0,
        'sentiment_distribution': {
            'negative': label_dist.get(0, 0) / total_labels if total_labels > 0 else 0,
            'neutral': label_dist.get(1, 0) / total_labels if total_labels > 0 else 0,
            'positive': label_dist.get(2, 0) / total_labels if total_labels > 0 else 0,
        }
    }

    return stats


def print_dataset_stats(stats: Dict[str, Any], dataset_name: str = "Dataset") -> None:
    """
    打印數據集統計資訊

    參數:
        stats: analyze_dataset 返回的統計字典
        dataset_name: 數據集名稱
    """
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {dataset_name}")
    print(f"{'='*60}")
    print(f"  Total samples:      {stats['total_samples']}")
    print(f"  Multi-aspect:       {stats['multi_aspect_count']} ({stats['multi_aspect_ratio']*100:.1f}%)")
    print(f"  Single-aspect:      {stats['single_aspect_count']} ({(1-stats['multi_aspect_ratio'])*100:.1f}%)")
    print(f"  Avg aspects/sample: {stats['avg_aspects']:.2f}")
    print(f"  Max aspects:        {stats['max_aspects']}")
    print(f"  Sentiment dist:     Neg {stats['sentiment_distribution']['negative']*100:.1f}% | "
          f"Neu {stats['sentiment_distribution']['neutral']*100:.1f}% | "
          f"Pos {stats['sentiment_distribution']['positive']*100:.1f}%")
    print(f"{'='*60}\n")
