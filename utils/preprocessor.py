"""
數據預處理模組
負責載入和預處理 SemEval-2014 等面向級情感分析數據集
"""

import os
import re
import pickle
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional, Set
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm


class SemEvalPreprocessor:
    """
    SemEval 數據集預處理器

    功能:
        - 解析 XML 格式的 SemEval 數據
        - 文本分詞和清理
        - 構建詞彙表
        - 生成面向掩碼
        - 數據增強（可選）
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = False,
        max_seq_length: int = 80,
        min_word_freq: int = 2
    ):
        """
        初始化預處理器

        參數:
            lowercase: 是否轉為小寫
            remove_punctuation: 是否移除標點符號
            max_seq_length: 最大序列長度
            min_word_freq: 最小詞頻（用於構建詞彙表）
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.max_seq_length = max_seq_length
        self.min_word_freq = min_word_freq

        # 詞彙表
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

        # 標籤映射
        self.label2idx = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.idx2label = {0: 'negative', 1: 'neutral', 2: 'positive'}

    def parse_semeval_xml(self, file_path: str) -> List[Dict]:
        """
        解析 SemEval XML 格式數據

        參數:
            file_path: XML 檔案路徑

        返回:
            樣本列表，每個樣本包含 text, aspects, polarities
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到檔案: {file_path}")

        tree = ET.parse(file_path)
        root = tree.getroot()

        samples = []

        for sentence in root.iter('sentence'):
            # 獲取句子文本
            text_elem = sentence.find('text')
            if text_elem is None:
                continue

            text = text_elem.text
            if text is None:
                continue

            # 獲取面向詞和極性
            aspect_terms = sentence.find('aspectTerms')
            if aspect_terms is None:
                continue

            aspects = []
            polarities = []

            for aspect_term in aspect_terms.iter('aspectTerm'):
                term = aspect_term.get('term')
                polarity = aspect_term.get('polarity')

                # 過濾 conflict 標籤
                if polarity == 'conflict':
                    continue

                if term and polarity:
                    aspects.append(term)
                    polarities.append(polarity)

            # 只保留有面向的樣本
            if aspects:
                samples.append({
                    'text': text,
                    'aspects': aspects,
                    'polarities': polarities
                })

        return samples

    def tokenize(self, text: str) -> List[str]:
        """
        簡單分詞

        參數:
            text: 輸入文本

        返回:
            詞列表
        """
        # 轉小寫
        if self.lowercase:
            text = text.lower()

        # 移除標點符號（可選）
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', ' ', text)

        # 簡單的空白分詞（可以替換為 spaCy 或 NLTK）
        tokens = text.split()

        return tokens

    def build_vocabulary(self, samples: List[Dict]) -> None:
        """
        從樣本構建詞彙表

        參數:
            samples: 樣本列表
        """
        # 統計詞頻
        word_counter = Counter()

        for sample in samples:
            tokens = self.tokenize(sample['text'])
            word_counter.update(tokens)

            # 也將面向詞加入
            for aspect in sample['aspects']:
                aspect_tokens = self.tokenize(aspect)
                word_counter.update(aspect_tokens)

        # 構建詞彙表（過濾低頻詞）
        idx = len(self.word2idx)
        for word, count in word_counter.items():
            if count >= self.min_word_freq:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

        print(f"詞彙表大小: {len(self.word2idx)}")

    def encode_text(self, text: str) -> List[int]:
        """
        將文本編碼為索引序列

        參數:
            text: 輸入文本

        返回:
            索引列表
        """
        tokens = self.tokenize(text)
        indices = [
            self.word2idx.get(token, self.word2idx['<UNK>'])
            for token in tokens
        ]
        return indices

    def pad_sequence(
        self,
        sequence: List[int],
        max_length: Optional[int] = None
    ) -> List[int]:
        """
        填充或截斷序列

        參數:
            sequence: 輸入序列
            max_length: 最大長度

        返回:
            填充後的序列
        """
        if max_length is None:
            max_length = self.max_seq_length

        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            return sequence + [self.word2idx['<PAD>']] * (max_length - len(sequence))

    def create_aspect_mask(
        self,
        text: str,
        aspect: str,
        seq_length: int
    ) -> List[int]:
        """
        創建面向掩碼（標記面向詞的位置）

        參數:
            text: 完整文本
            aspect: 面向詞
            seq_length: 序列長度

        返回:
            掩碼列表（1 表示面向詞位置，0 表示其他）
        """
        text_tokens = self.tokenize(text)
        aspect_tokens = self.tokenize(aspect)

        mask = [0] * seq_length

        # 查找面向詞在文本中的位置
        aspect_len = len(aspect_tokens)
        for i in range(len(text_tokens) - aspect_len + 1):
            if text_tokens[i:i+aspect_len] == aspect_tokens:
                # 標記面向詞位置
                for j in range(min(aspect_len, seq_length - i)):
                    if i + j < seq_length:
                        mask[i + j] = 1
                break

        return mask

    def process_samples(
        self,
        samples: List[Dict],
        build_vocab: bool = False
    ) -> pd.DataFrame:
        """
        處理樣本列表

        參數:
            samples: 樣本列表
            build_vocab: 是否構建詞彙表

        返回:
            處理後的 DataFrame
        """
        if build_vocab:
            self.build_vocabulary(samples)

        processed_data = []

        for sample in tqdm(samples, desc="處理樣本"):
            text = sample['text']
            text_indices = self.encode_text(text)
            text_padded = self.pad_sequence(text_indices)

            # 處理每個面向
            for aspect, polarity in zip(sample['aspects'], sample['polarities']):
                # 編碼面向詞
                aspect_indices = self.encode_text(aspect)
                aspect_padded = self.pad_sequence(aspect_indices, max_length=10)

                # 創建面向掩碼
                aspect_mask = self.create_aspect_mask(
                    text, aspect, self.max_seq_length
                )

                # 編碼標籤
                label = self.label2idx.get(polarity, -1)

                if label == -1:  # 跳過未知標籤
                    continue

                processed_data.append({
                    'text': text,
                    'aspect': aspect,
                    'text_indices': text_padded,
                    'aspect_indices': aspect_padded,
                    'aspect_mask': aspect_mask,
                    'label': label,
                    'polarity': polarity
                })

        df = pd.DataFrame(processed_data)
        print(f"處理後樣本數: {len(df)}")

        return df

    def save_vocabulary(self, save_path: str):
        """
        保存詞彙表

        參數:
            save_path: 保存路徑
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': self.idx2word,
            'label2idx': self.label2idx,
            'idx2label': self.idx2label
        }

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(vocab_data, f)

        print(f"詞彙表已保存到: {save_path}")

    def load_vocabulary(self, load_path: str):
        """
        載入詞彙表

        參數:
            load_path: 載入路徑
        """
        with open(load_path, 'rb') as f:
            vocab_data = pickle.load(f)

        self.word2idx = vocab_data['word2idx']
        self.idx2word = vocab_data['idx2word']
        self.label2idx = vocab_data['label2idx']
        self.idx2label = vocab_data['idx2label']

        print(f"詞彙表已從 {load_path} 載入")

    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """
        獲取數據統計資訊

        參數:
            df: 數據 DataFrame

        返回:
            統計資訊字典
        """
        stats = {
            'num_samples': len(df),
            'num_unique_texts': df['text'].nunique(),
            'num_unique_aspects': df['aspect'].nunique(),
            'label_distribution': df['polarity'].value_counts().to_dict(),
            'avg_text_length': df['text'].apply(lambda x: len(x.split())).mean(),
            'avg_aspect_length': df['aspect'].apply(lambda x: len(x.split())).mean(),
            'vocab_size': len(self.word2idx)
        }

        return stats

    def print_statistics(self, df: pd.DataFrame):
        """
        打印數據統計資訊

        參數:
            df: 數據 DataFrame
        """
        stats = self.get_statistics(df)

        print("\n" + "=" * 60)
        print("數據統計資訊")
        print("=" * 60)
        print(f"樣本數量: {stats['num_samples']}")
        print(f"唯一句子數: {stats['num_unique_texts']}")
        print(f"唯一面向數: {stats['num_unique_aspects']}")
        print(f"詞彙表大小: {stats['vocab_size']}")
        print(f"平均句子長度: {stats['avg_text_length']:.2f} 詞")
        print(f"平均面向長度: {stats['avg_aspect_length']:.2f} 詞")
        print("\n標籤分布:")
        for label, count in stats['label_distribution'].items():
            percentage = count / stats['num_samples'] * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
        print("=" * 60 + "\n")


def load_semeval_2014(
    data_dir: str,
    domain: str = "restaurant",
    preprocessor: Optional[SemEvalPreprocessor] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    載入 SemEval-2014 數據集

    參數:
        data_dir: 數據目錄
        domain: 領域 ("restaurant" 或 "laptop")
        preprocessor: 預處理器實例（None 則創建新的）

    返回:
        (訓練數據, 測試數據) 的 DataFrame 元組
    """
    if preprocessor is None:
        preprocessor = SemEvalPreprocessor()

    # 構建檔案路徑
    train_file = os.path.join(data_dir, f"{domain}_train.xml")
    test_file = os.path.join(data_dir, f"{domain}_test.xml")

    print(f"載入 SemEval-2014 {domain.capitalize()} 數據集...")

    # 解析訓練數據
    print("解析訓練數據...")
    train_samples = preprocessor.parse_semeval_xml(train_file)

    # 構建詞彙表
    preprocessor.build_vocabulary(train_samples)

    # 處理訓練數據
    train_df = preprocessor.process_samples(train_samples, build_vocab=False)

    # 解析測試數據
    print("解析測試數據...")
    test_samples = preprocessor.parse_semeval_xml(test_file)

    # 處理測試數據（使用相同的詞彙表）
    test_df = preprocessor.process_samples(test_samples, build_vocab=False)

    # 打印統計資訊
    print("\n訓練集統計:")
    preprocessor.print_statistics(train_df)

    print("測試集統計:")
    preprocessor.print_statistics(test_df)

    return train_df, test_df


def split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.2,
    stratify: bool = True,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    將訓練數據劃分為訓練集和驗證集

    參數:
        df: 輸入 DataFrame
        val_ratio: 驗證集比例
        stratify: 是否分層抽樣
        random_state: 隨機種子

    返回:
        (訓練集, 驗證集) 的 DataFrame 元組
    """
    from sklearn.model_selection import train_test_split

    if stratify:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=random_state,
            stratify=df['label']
        )
    else:
        train_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=random_state
        )

    print(f"訓練集大小: {len(train_df)}")
    print(f"驗證集大小: {len(val_df)}")

    return train_df, val_df


if __name__ == "__main__":
    # 測試預處理器
    print("測試 SemEval 預處理器...")

    # 創建預處理器
    preprocessor = SemEvalPreprocessor(
        lowercase=True,
        max_seq_length=80,
        min_word_freq=2
    )

    # 模擬一些樣本（實際使用時替換為真實的 XML 檔案路徑）
    sample_data = [
        {
            'text': 'The food was great but the service was terrible.',
            'aspects': ['food', 'service'],
            'polarities': ['positive', 'negative']
        },
        {
            'text': 'I love the atmosphere here.',
            'aspects': ['atmosphere'],
            'polarities': ['positive']
        }
    ]

    # 處理樣本
    df = preprocessor.process_samples(sample_data, build_vocab=True)

    # 打印統計
    preprocessor.print_statistics(df)

    # 顯示前幾筆數據
    print("\n前幾筆處理後的數據:")
    print(df.head())

    print("\n預處理器測試完成！")
