"""
SemEval-2016 Multi-Aspect 數據加載器

支持 SemEval-2016 Task 5 格式:
- Reviews -> Review -> sentences -> sentence -> Opinions -> Opinion
- Opinion: target, category, polarity, from, to

與 SemEval-2014 格式不同:
- 2014: sentence -> aspectTerms -> aspectTerm (term, polarity)
- 2016: Review -> sentences -> sentence -> Opinions -> Opinion (target, polarity)
"""

import xml.etree.ElementTree as ET
from typing import List, Tuple
from collections import Counter
import numpy as np

from datasets.loader_semeval14 import MultiAspectSample


class SemEval2016MultiAspectLoader:
    """SemEval-2016 Multi-Aspect 數據加載器"""

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
        virtual_aspect_mode: str = 'overall'
    ):
        """
        初始化數據加載器

        參數:
            xml_path: XML 文件路徑
            min_aspects: 最小 aspect 數量（用於過濾）
            max_aspects: 最大 aspect 數量（超過則截斷）
            include_single_aspect: 是否包含單 aspect 句子（帶虛擬 aspect）
            virtual_aspect_mode: 虛擬 aspect 模式
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

        # SemEval-2016 格式: Reviews -> Review -> sentences -> sentence
        for sentence_elem in root.findall('.//sentence'):
            text_elem = sentence_elem.find('text')
            if text_elem is None or not text_elem.text:
                continue

            text = text_elem.text

            # 提取所有 Opinions
            opinions = sentence_elem.findall('.//Opinion')

            if len(opinions) == 0:
                sentence_counts[0] += 1
                continue

            # 收集 aspects 和 labels
            aspects = []
            labels = []
            positions = []

            for opinion in opinions:
                # SemEval-2016 格式:
                # - Restaurants: 有 target 屬性 (Subtask 1)
                # - Laptops: 只有 category 屬性 (Slot 1, implicit aspects)
                target = opinion.get('target')
                category = opinion.get('category')
                polarity = opinion.get('polarity')

                # 確定 aspect 文本
                if target and target != 'NULL':
                    # 有明確的 target (Restaurants)
                    aspect_text = target
                elif category:
                    # 沒有 target，使用 category (Laptops)
                    # 格式: "LAPTOP#GENERAL" -> "laptop general"
                    aspect_text = category.replace('#', ' ').replace('_', ' ').lower()
                else:
                    continue

                # 轉換情感標籤
                label = self.SENTIMENT_MAP.get(polarity, 1)

                aspects.append(aspect_text)
                labels.append(label)

                # 提取位置信息 (只有 target 類型有位置)
                try:
                    start = int(opinion.get('from', 0))
                    end = int(opinion.get('to', 0))
                    if start > 0 or end > 0:
                        positions.append((start, end))
                    else:
                        positions.append(None)
                except:
                    positions.append(None)

            num_aspects = len(aspects)
            sentence_counts[num_aspects] += 1

            if num_aspects == 0:
                # 所有 opinions 都是 NULL target
                continue

            # 根據配置處理樣本
            if num_aspects >= self.min_aspects:
                # Multi-aspect 樣本
                if num_aspects > self.max_aspects:
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
                        aspect_positions=None
                    )
                    self.samples.append(sample)
                    aspect_counts[len(virtual_aspects)] += 1

        # 保存統計信息
        self.statistics = {
            'total_samples': len(self.samples),
            'sentence_aspect_distribution': dict(sentence_counts),
            'sample_aspect_distribution': dict(aspect_counts),
            'avg_aspects_per_sample': np.mean([s.num_aspects for s in self.samples]) if self.samples else 0
        }

        self._print_statistics()

        return self.samples

    def _create_virtual_aspects(
        self,
        text: str,
        real_aspect: str,
        real_label: int
    ) -> Tuple[List[str], List[int]]:
        """為單 aspect 樣本創建虛擬 aspects"""
        if self.virtual_aspect_mode == 'none':
            return None, None

        if self.virtual_aspect_mode == 'overall':
            virtual_aspect = '<VIRTUAL>overall'
            virtual_label = real_label  # 使用相同標籤
        elif self.virtual_aspect_mode == 'context':
            virtual_aspect = '<VIRTUAL>context'
            virtual_label = 1  # 中性
        else:
            return None, None

        return [real_aspect, virtual_aspect], [real_label, virtual_label]

    def _print_statistics(self):
        pass


def load_semeval2016_data(
    train_path: str,
    test_path: str,
    min_aspects: int = 2,
    max_aspects: int = 8,
    include_single_aspect: bool = True,
    virtual_aspect_mode: str = 'overall',
    val_split: float = 0.1
) -> Tuple[List[MultiAspectSample], List[MultiAspectSample], List[MultiAspectSample]]:
    """
    加載 SemEval-2016 數據

    參數:
        train_path: 訓練數據路徑
        test_path: 測試數據路徑
        min_aspects: 最小 aspect 數量
        max_aspects: 最大 aspect 數量
        include_single_aspect: 是否包含單 aspect
        virtual_aspect_mode: 虛擬 aspect 模式
        val_split: 驗證集比例

    返回:
        (train_samples, val_samples, test_samples)
    """
    train_loader = SemEval2016MultiAspectLoader(
        xml_path=train_path,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
        include_single_aspect=include_single_aspect,
        virtual_aspect_mode=virtual_aspect_mode
    )
    train_all_samples = train_loader.load()

    test_loader = SemEval2016MultiAspectLoader(
        xml_path=test_path,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
        include_single_aspect=include_single_aspect,
        virtual_aspect_mode=virtual_aspect_mode
    )
    test_samples = test_loader.load()

    # 分割訓練集和驗證集
    np.random.seed(42)
    indices = np.random.permutation(len(train_all_samples))
    val_size = int(len(train_all_samples) * val_split)

    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_samples = [train_all_samples[i] for i in train_indices]
    val_samples = [train_all_samples[i] for i in val_indices]

    return train_samples, val_samples, test_samples
