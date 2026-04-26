"""
MAMS-ATSA Multi-Aspect 數據加載器

MAMS (Multi-Aspect Multi-Sentiment) 數據集是專門為多面向情感分析設計的：
- 每個句子包含至少 2 個 aspects
- 強制不同情感極性（正面、負面、中性必須混合）
- 真正的多面向建模場景

與 SemEval-2014 的主要區別：
- MAMS: 100% 多面向句子，適合驗證 PMAC + IARM
- SemEval: ~20% 多面向句子，大多數為單面向

數據格式與 SemEval-2014 完全相同（XML），可共用大部分代碼。
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


class MAMSMultiAspectLoader:
    """MAMS-ATSA Multi-Aspect 數據加載器"""

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
        include_single_aspect: bool = False,  # MAMS 通常不需要，因為已經都是多面向
        virtual_aspect_mode: str = 'none'  # MAMS 通常不需要虛擬 aspect
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
        sentiment_diversity = Counter()  # MAMS 特有：統計情感多樣性

        # 處理每個句子
        for sentence_elem in root.findall('.//sentence'):
            text_elem = sentence_elem.find('text')
            if text_elem is None or not text_elem.text:
                continue

            text = text_elem.text

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

            # 統計情感多樣性（MAMS 特有）
            unique_sentiments = len(set(labels))
            sentiment_diversity[unique_sentiments] += 1

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
            'sentiment_diversity': dict(sentiment_diversity),
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
        # 簡化輸出：統計信息已在主訓練腳本中顯示
        pass

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


def load_mams_data(
    train_path: str,
    val_path: str,
    test_path: str,
    min_aspects: int = 2,
    max_aspects: int = 8,
    include_single_aspect: bool = False,
    virtual_aspect_mode: str = 'none'
) -> Tuple[List[MultiAspectSample], List[MultiAspectSample], List[MultiAspectSample]]:
    """
    加載 MAMS 訓練集、驗證集和測試集

    參數:
        train_path: 訓練集 XML 路徑
        val_path: 驗證集 XML 路徑
        test_path: 測試集 XML 路徑
        min_aspects: 最小 aspect 數量
        max_aspects: 最大 aspect 數量
        include_single_aspect: 是否包含單 aspect
        virtual_aspect_mode: 虛擬 aspect 模式

    返回:
        (train_samples, val_samples, test_samples)
    """
    # 加載訓練集
    train_loader = MAMSMultiAspectLoader(
        xml_path=train_path,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
        include_single_aspect=include_single_aspect,
        virtual_aspect_mode=virtual_aspect_mode
    )
    train_samples = train_loader.load()

    # 加載驗證集
    val_loader = MAMSMultiAspectLoader(
        xml_path=val_path,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
        include_single_aspect=include_single_aspect,
        virtual_aspect_mode=virtual_aspect_mode
    )
    val_samples = val_loader.load()

    # 加載測試集
    test_loader = MAMSMultiAspectLoader(
        xml_path=test_path,
        min_aspects=min_aspects,
        max_aspects=max_aspects,
        include_single_aspect=include_single_aspect,
        virtual_aspect_mode=virtual_aspect_mode
    )
    test_samples = test_loader.load()

    return train_samples, val_samples, test_samples


if __name__ == '__main__':
    """測試數據加載"""
    import sys
    from pathlib import Path

    # 設置路徑
    project_root = Path(__file__).parent.parent
    train_path = project_root / 'data' / 'raw' / 'MAMS-ATSA' / 'train.xml'
    val_path = project_root / 'data' / 'raw' / 'MAMS-ATSA' / 'val.xml'
    test_path = project_root / 'data' / 'raw' / 'MAMS-ATSA' / 'test.xml'

    # 測試加載（純多面向模式）
    print("測試 MAMS 數據加載 (min_aspects=2, 純多面向場景)")
    train_samples, val_samples, test_samples = load_mams_data(
        train_path=str(train_path),
        val_path=str(val_path),
        test_path=str(test_path),
        min_aspects=2,
        max_aspects=8,
        include_single_aspect=False,  # MAMS 通常不需要
        virtual_aspect_mode='none'  # MAMS 已經是多面向，不需要虛擬 aspect
    )

    print(f"\n最終數據集大小:")
    print(f"  訓練集: {len(train_samples)} 樣本")
    print(f"  驗證集: {len(val_samples)} 樣本")
    print(f"  測試集: {len(test_samples)} 樣本")
