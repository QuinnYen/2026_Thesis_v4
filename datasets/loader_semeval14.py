"""
SemEval-2014 Multi-Aspect 數據加載器

支持句子級別的多面向情感分析：
- 將同一句子的多個 aspects 組合在一起
- 支持混合模式（multi-aspect + single-aspect with virtual aspect）
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from collections import Counter
import numpy as np


class MultiAspectSample:
    """多面向樣本"""

    def __init__(
        self,
        text: str,
        aspects: List[str],
        labels: List[int],
        aspect_positions: Optional[List[Tuple[int, int]]] = None
    ):
        """
        初始化多面向樣本

        參數:
            text: 句子文本
            aspects: aspect 列表 (可能包含虛擬 aspect)
            labels: 情感標籤列表 (0=negative, 1=neutral, 2=positive)
            aspect_positions: aspect 在文本中的位置 [(start, end), ...]
        """
        self.text = text
        self.aspects = aspects
        self.labels = labels
        self.num_aspects = len(aspects)
        self.aspect_positions = aspect_positions

        # 標記是否為虛擬 aspect
        self.is_virtual = ['<VIRTUAL>' in asp for asp in aspects]

    def __repr__(self):
        return f"MultiAspectSample(text='{self.text[:50]}...', num_aspects={self.num_aspects})"


class SemEvalMultiAspectLoader:
    """SemEval-2014 Multi-Aspect 數據加載器"""

    # 情感標籤映射
    SENTIMENT_MAP = {
        'positive': 2,
        'neutral': 1,
        'negative': 0,
        'conflict': 1  # 衝突標記為中性
    }

    def __init__(
        self,
        xml_path: str,
        min_aspects: int = 2,
        max_aspects: int = 8,
        include_single_aspect: bool = True,
        virtual_aspect_mode: str = 'overall'  # 'overall', 'context', or 'none'
    ):
        """
        初始化數據加載器

        參數:
            xml_path: XML 文件路徑
            min_aspects: 最小 aspect 數量（用於過濾）
            max_aspects: 最大 aspect 數量（超過則截斷）
            include_single_aspect: 是否包含單 aspect 句子（帶虛擬 aspect）
            virtual_aspect_mode: 虛擬 aspect 模式
                - 'overall': 添加整體評價 aspect
                - 'context': 添加上下文 aspect
                - 'none': 不添加虛擬 aspect
        """
        self.xml_path = xml_path
        self.min_aspects = min_aspects
        self.max_aspects = max_aspects
        self.include_single_aspect = include_single_aspect
        self.virtual_aspect_mode = virtual_aspect_mode

        self.samples = []
        self.statistics = {}

    def load(self) -> List[MultiAspectSample]:
        """
        加載數據

        返回:
            MultiAspectSample 列表
        """

        # 解析 XML
        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        # 統計信息
        sentence_counts = Counter()
        aspect_counts = Counter()

        # 處理每個句子
        for sentence_elem in root.findall('.//sentence'):
            text = sentence_elem.find('text').text
            if not text:
                continue

            # 提取所有 aspects
            aspect_terms = sentence_elem.findall('.//aspectTerm')

            if len(aspect_terms) == 0:
                # 跳過沒有 aspect 的句子
                sentence_counts[0] += 1
                continue

            # 收集 aspects 和 labels
            aspects = []
            labels = []
            positions = []

            for term_elem in aspect_terms:
                aspect_text = term_elem.get('term')
                polarity = term_elem.get('polarity')

                # 轉換情感標籤
                label = self.SENTIMENT_MAP.get(polarity, 1)

                aspects.append(aspect_text)
                labels.append(label)

                # 提取位置信息（如果有）
                try:
                    start = int(term_elem.get('from'))
                    end = int(term_elem.get('to'))
                    positions.append((start, end))
                except:
                    positions.append(None)

            num_aspects = len(aspects)
            sentence_counts[num_aspects] += 1

            # 根據配置處理樣本
            if num_aspects >= self.min_aspects:
                # Multi-aspect 樣本
                if num_aspects > self.max_aspects:
                    # 截斷
                    aspects = aspects[:self.max_aspects]
                    labels = labels[:self.max_aspects]
                    positions = positions[:self.max_aspects]

                sample = MultiAspectSample(
                    text=text,
                    aspects=aspects,
                    labels=labels,
                    aspect_positions=positions if all(p for p in positions) else None
                )
                self.samples.append(sample)
                aspect_counts[len(aspects)] += 1

            elif num_aspects == 1 and self.include_single_aspect:
                # 單 aspect 樣本，添加虛擬 aspect
                virtual_aspects, virtual_labels = self._create_virtual_aspects(
                    text, aspects[0], labels[0]
                )

                if virtual_aspects:
                    sample = MultiAspectSample(
                        text=text,
                        aspects=virtual_aspects,
                        labels=virtual_labels,
                        aspect_positions=None  # 虛擬 aspect 沒有位置
                    )
                    self.samples.append(sample)
                    aspect_counts[len(virtual_aspects)] += 1

        # 保存統計信息
        self.statistics = {
            'total_samples': len(self.samples),
            'sentence_aspect_distribution': dict(sentence_counts),
            'sample_aspect_distribution': dict(aspect_counts),
            'avg_aspects_per_sample': np.mean([s.num_aspects for s in self.samples])
        }

        self._print_statistics()

        return self.samples

    def _create_virtual_aspects(
        self,
        text: str,
        real_aspect: str,
        real_label: int
    ) -> Tuple[List[str], List[int]]:
        """
        為單 aspect 樣本創建虛擬 aspects

        參數:
            text: 句子文本
            real_aspect: 真實 aspect
            real_label: 真實標籤

        返回:
            (aspects 列表, labels 列表)
        """
        if self.virtual_aspect_mode == 'none':
            return [real_aspect], [real_label]

        elif self.virtual_aspect_mode == 'overall':
            # 添加整體評價虛擬 aspect
            virtual_aspects = [
                real_aspect,
                '<VIRTUAL>overall experience'
            ]
            # 虛擬 aspect 使用相同標籤（假設整體情感與該 aspect 一致）
            virtual_labels = [real_label, real_label]

            return virtual_aspects, virtual_labels

        elif self.virtual_aspect_mode == 'context':
            # 添加上下文虛擬 aspect
            virtual_aspects = [
                real_aspect,
                '<VIRTUAL>context'
            ]
            # 虛擬 aspect 標籤設為中性（不確定）
            virtual_labels = [real_label, 1]

            return virtual_aspects, virtual_labels

        else:
            return [real_aspect], [real_label]

    def _print_statistics(self):
        """打印統計信息（簡化版）"""
        # 簡化輸出：只顯示關鍵統計
        pass  # 統計信息已在主訓練腳本中顯示

    def get_samples(self) -> List[MultiAspectSample]:
        """獲取樣本"""
        return self.samples

    def get_statistics(self) -> Dict:
        """獲取統計信息"""
        return self.statistics

    def print_examples(self, num_examples: int = 5):
        """打印範例（簡化版）"""
        # 簡化輸出：不打印範例
        pass


def load_augmented_neutral(augmented_path: str, max_samples: Optional[int] = None) -> List[MultiAspectSample]:
    """
    加載 Claude 增強的 Neutral 數據

    參數:
        augmented_path: 增強數據路徑 (支持 JSON 或 XML 格式)
        max_samples: 最大樣本數 (用於控制增強比例，避免過度增強)

    返回:
        MultiAspectSample 列表 (每個樣本只有一個 aspect)
    """
    import random
    import json

    samples = []

    # 根據文件擴展名選擇解析方式
    if augmented_path.endswith('.json'):
        # JSON 格式 (新版 Claude 增強)
        with open(augmented_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            text = item.get('text', '')
            aspect = item.get('aspect', '')

            if not text or not aspect:
                continue

            # Neutral = 1
            label = 1

            # 創建單 aspect 樣本 (帶 virtual aspect)
            sample = MultiAspectSample(
                text=text,
                aspects=[aspect, '<VIRTUAL>overall experience'],
                labels=[label, label],  # 虛擬 aspect 也用相同標籤
                aspect_positions=None
            )
            samples.append(sample)

    else:
        # XML 格式 (舊版)
        tree = ET.parse(augmented_path)
        root = tree.getroot()

        for sentence_elem in root.findall('.//sentence'):
            text_elem = sentence_elem.find('text')
            if text_elem is None or not text_elem.text:
                continue

            text = text_elem.text

            # 提取 aspect
            for term_elem in sentence_elem.findall('.//aspectTerm'):
                aspect = term_elem.get('term', '')

                # Neutral = 1
                label = 1

                # 創建單 aspect 樣本 (帶 virtual aspect)
                sample = MultiAspectSample(
                    text=text,
                    aspects=[aspect, '<VIRTUAL>overall experience'],
                    labels=[label, label],  # 虛擬 aspect 也用相同標籤
                    aspect_positions=None
                )
                samples.append(sample)

    # 如果指定了最大樣本數，隨機抽取
    if max_samples and len(samples) > max_samples:
        random.seed(42)  # 固定種子確保可重複性
        samples = random.sample(samples, max_samples)

    return samples


def load_pseudo_labeled(pseudo_path: str, max_samples: Optional[int] = None) -> List[MultiAspectSample]:
    """
    加載偽標籤數據 (Self-Training)

    參數:
        pseudo_path: 偽標籤數據 JSON 路徑
        max_samples: 最大樣本數

    返回:
        MultiAspectSample 列表
    """
    import json
    import random
    import os

    if not os.path.exists(pseudo_path):
        return []

    with open(pseudo_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    for item in data:
        text = item.get('text', '')
        aspect = item.get('aspect', '')
        label = item.get('label', 1)  # 預設 neutral

        if not text or not aspect:
            continue

        # 創建單 aspect 樣本 (帶 virtual aspect)
        sample = MultiAspectSample(
            text=text,
            aspects=[aspect, '<VIRTUAL>overall experience'],
            labels=[label, label],
            aspect_positions=None
        )
        samples.append(sample)

    # 如果指定了最大樣本數，隨機抽取
    if max_samples and len(samples) > max_samples:
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return samples


def load_multiaspect_data(
    train_path: str,
    test_path: str,
    min_aspects: int = 2,
    max_aspects: int = 8,
    include_single_aspect: bool = True,
    virtual_aspect_mode: str = 'overall',
    augmented_neutral_path: Optional[str] = None,
    pseudo_labeled_path: Optional[str] = None
) -> Tuple[List[MultiAspectSample], List[MultiAspectSample]]:
    """
    加載訓練集和測試集

    參數:
        train_path: 訓練集 XML 路徑
        test_path: 測試集 XML 路徑
        min_aspects: 最小 aspect 數量
        max_aspects: 最大 aspect 數量
        include_single_aspect: 是否包含單 aspect
        virtual_aspect_mode: 虛擬 aspect 模式
        augmented_neutral_path: (可選) 增強的 Neutral 數據路徑
        pseudo_labeled_path: (可選) 偽標籤數據路徑 (Self-Training)

    返回:
        (train_samples, test_samples)
    """
    import os
    from collections import Counter

    # 加載訓練集
    train_loader = SemEvalMultiAspectLoader(
        xml_path=train_path,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
        include_single_aspect=include_single_aspect,
        virtual_aspect_mode=virtual_aspect_mode
    )
    train_samples = train_loader.load()

    # 如果有增強的 Neutral 數據，合併到訓練集
    if augmented_neutral_path:
        if os.path.exists(augmented_neutral_path):
            # 先統計原始訓練集的類別分布
            original_counts = Counter()
            for sample in train_samples:
                for label in sample.labels:
                    original_counts[label] += 1

            # 計算目標：讓 Neutral 達到約 33%（三類平衡）
            # 注意：每個增強樣本有 2 個 labels（真實 + 虛擬），都是 Neutral
            total_labels = sum(original_counts.values())
            current_neutral = original_counts[1]
            current_neutral_pct = current_neutral / total_labels * 100

            # 目標：Neutral 佔 33%（三類平衡）
            target_pct = 0.33

            # 加載所有增強數據（不限制數量，讓模型充分學習 Neutral）
            augmented_samples = load_augmented_neutral(augmented_neutral_path, max_samples=None)
            original_count = len(train_samples)
            train_samples.extend(augmented_samples)

            print(f"  [Augmentation] 原始 Neutral: {current_neutral} ({current_neutral_pct:.1f}%)")
            print(f"  [Augmentation] 加載 {len(augmented_samples)} 個增強樣本 (目標: {target_pct*100:.0f}%)")
            print(f"  [Augmentation] 訓練集: {original_count} -> {len(train_samples)}")

    # 加載偽標籤數據 (Self-Training)
    if pseudo_labeled_path:
        if os.path.exists(pseudo_labeled_path):
            pseudo_samples = load_pseudo_labeled(pseudo_labeled_path)
            original_count = len(train_samples)
            train_samples.extend(pseudo_samples)

            # 統計偽標籤的類別分布
            pseudo_labels = Counter()
            for s in pseudo_samples:
                for label in s.labels:
                    pseudo_labels[label] += 1

            print(f"  [Self-Training] 加載 {len(pseudo_samples)} 個偽標籤樣本")
            print(f"  [Self-Training] 偽標籤分布: Neg={pseudo_labels[0]}, Neu={pseudo_labels[1]}, Pos={pseudo_labels[2]}")
            print(f"  [Self-Training] 訓練集: {original_count} -> {len(train_samples)}")

    # 加載測試集
    test_loader = SemEvalMultiAspectLoader(
        xml_path=test_path,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
        include_single_aspect=include_single_aspect,
        virtual_aspect_mode=virtual_aspect_mode
    )
    test_samples = test_loader.load()

    return train_samples, test_samples


if __name__ == '__main__':
    """測試數據加載"""
    import sys
    from pathlib import Path

    # 設置路徑
    project_root = Path(__file__).parent.parent
    train_path = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Train_v2.xml'
    test_path = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Test_Gold.xml'

    # 測試加載（混合模式：2.2）
    print("測試混合模式數據加載 (min_aspects=2, include_single_aspect=True)")
    train_samples, test_samples = load_multiaspect_data(
        train_path=str(train_path),
        test_path=str(test_path),
        min_aspects=2,
        max_aspects=8,
        include_single_aspect=True,
        virtual_aspect_mode='overall'
    )

    print(f"\n最終數據集大小:")
    print(f"  訓練集: {len(train_samples)} 樣本")
    print(f"  測試集: {len(test_samples)} 樣本")
