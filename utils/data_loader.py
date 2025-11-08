"""
數據載入模組
提供 PyTorch DataLoader 和批次處理功能
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class ABSADataset(Dataset):
    """
    面向級情感分析數據集

    功能:
        - 封裝預處理後的數據
        - 支援動態批次處理
        - 返回 PyTorch 張量
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_seq_length: int = 80,
        max_aspect_length: int = 10
    ):
        """
        初始化數據集

        參數:
            dataframe: 預處理後的 DataFrame
            max_seq_length: 最大序列長度
            max_aspect_length: 最大面向長度
        """
        self.df = dataframe.reset_index(drop=True)
        self.max_seq_length = max_seq_length
        self.max_aspect_length = max_aspect_length

    def __len__(self) -> int:
        """返回數據集大小"""
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        獲取單個樣本

        參數:
            idx: 樣本索引

        返回:
            包含張量的字典
        """
        row = self.df.iloc[idx]

        # 文本索引
        text_indices = torch.tensor(row['text_indices'], dtype=torch.long)

        # 面向索引
        aspect_indices = torch.tensor(row['aspect_indices'], dtype=torch.long)

        # 面向掩碼
        aspect_mask = torch.tensor(row['aspect_mask'], dtype=torch.float)

        # 標籤
        label = torch.tensor(row['label'], dtype=torch.long)

        # 計算實際長度（用於動態 RNN）
        text_len = torch.sum(text_indices != 0).item()
        aspect_len = torch.sum(aspect_indices != 0).item()

        return {
            'text_indices': text_indices,          # [seq_len]
            'aspect_indices': aspect_indices,      # [aspect_len]
            'aspect_mask': aspect_mask,            # [seq_len]
            'text_len': text_len,                  # scalar
            'aspect_len': aspect_len,              # scalar
            'label': label                         # scalar
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    自定義批次整理函數

    參數:
        batch: 樣本列表

    返回:
        批次字典
    """
    # 提取各個欄位
    text_indices = torch.stack([item['text_indices'] for item in batch])
    aspect_indices = torch.stack([item['aspect_indices'] for item in batch])
    aspect_mask = torch.stack([item['aspect_mask'] for item in batch])
    text_len = torch.tensor([item['text_len'] for item in batch], dtype=torch.long)
    aspect_len = torch.tensor([item['aspect_len'] for item in batch], dtype=torch.long)
    labels = torch.stack([item['label'] for item in batch])

    return {
        'text_indices': text_indices,      # [batch, seq_len]
        'aspect_indices': aspect_indices,  # [batch, aspect_len]
        'aspect_mask': aspect_mask,        # [batch, seq_len]
        'text_len': text_len,              # [batch]
        'aspect_len': aspect_len,          # [batch]
        'labels': labels                   # [batch]
    }


def create_data_loader(
    dataframe: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    創建 DataLoader

    參數:
        dataframe: 預處理後的 DataFrame
        batch_size: 批次大小
        shuffle: 是否打亂
        num_workers: 工作執行緒數
        pin_memory: 是否使用固定記憶體
        drop_last: 是否丟棄最後不完整的批次

    返回:
        DataLoader 實例
    """
    dataset = ABSADataset(dataframe)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

    return data_loader


class MultiAspectDataset(Dataset):
    """
    多面向數據集
    用於處理一個句子包含多個面向的情況
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        max_seq_length: int = 80,
        max_aspect_length: int = 10,
        max_num_aspects: int = 5
    ):
        """
        初始化多面向數據集

        參數:
            dataframe: 預處理後的 DataFrame
            max_seq_length: 最大序列長度
            max_aspect_length: 單個面向的最大長度
            max_num_aspects: 最多的面向數
        """
        self.max_seq_length = max_seq_length
        self.max_aspect_length = max_aspect_length
        self.max_num_aspects = max_num_aspects

        # 按文本分組（一個句子可能有多個面向）
        self.grouped_data = self._group_by_text(dataframe)

    def _group_by_text(self, df: pd.DataFrame) -> List[Dict]:
        """
        按文本分組

        參數:
            df: DataFrame

        返回:
            分組後的列表
        """
        grouped = []

        for text, group in df.groupby('text'):
            # 收集所有面向和標籤
            aspects = []
            aspect_masks = []
            labels = []

            for _, row in group.iterrows():
                if len(aspects) >= self.max_num_aspects:
                    break

                aspects.append(row['aspect_indices'])
                aspect_masks.append(row['aspect_mask'])
                labels.append(row['label'])

            grouped.append({
                'text_indices': group.iloc[0]['text_indices'],
                'aspects': aspects,
                'aspect_masks': aspect_masks,
                'labels': labels,
                'num_aspects': len(aspects)
            })

        return grouped

    def __len__(self) -> int:
        return len(self.grouped_data)

    def __getitem__(self, idx: int) -> Dict:
        """
        獲取單個樣本

        參數:
            idx: 樣本索引

        返回:
            包含多個面向的字典
        """
        item = self.grouped_data[idx]

        # 文本索引
        text_indices = torch.tensor(item['text_indices'], dtype=torch.long)

        # 面向索引（填充到 max_num_aspects）
        aspects = item['aspects']
        num_aspects = len(aspects)

        # 填充面向列表
        aspect_indices_list = []
        for i in range(self.max_num_aspects):
            if i < num_aspects:
                aspect_indices_list.append(torch.tensor(aspects[i], dtype=torch.long))
            else:
                aspect_indices_list.append(torch.zeros(self.max_aspect_length, dtype=torch.long))

        aspect_indices = torch.stack(aspect_indices_list)  # [max_num_aspects, aspect_len]

        # 面向掩碼
        aspect_masks_list = []
        for i in range(self.max_num_aspects):
            if i < num_aspects:
                aspect_masks_list.append(torch.tensor(item['aspect_masks'][i], dtype=torch.float))
            else:
                aspect_masks_list.append(torch.zeros(self.max_seq_length, dtype=torch.float))

        aspect_masks = torch.stack(aspect_masks_list)  # [max_num_aspects, seq_len]

        # 標籤
        labels_list = []
        for i in range(self.max_num_aspects):
            if i < num_aspects:
                labels_list.append(item['labels'][i])
            else:
                labels_list.append(-1)  # 填充標籤

        labels = torch.tensor(labels_list, dtype=torch.long)

        # 有效面向掩碼
        valid_mask = torch.zeros(self.max_num_aspects, dtype=torch.bool)
        valid_mask[:num_aspects] = True

        return {
            'text_indices': text_indices,        # [seq_len]
            'aspect_indices': aspect_indices,    # [max_num_aspects, aspect_len]
            'aspect_masks': aspect_masks,        # [max_num_aspects, seq_len]
            'labels': labels,                    # [max_num_aspects]
            'valid_mask': valid_mask,            # [max_num_aspects]
            'num_aspects': num_aspects
        }


def multi_aspect_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    多面向批次整理函數

    參數:
        batch: 樣本列表

    返回:
        批次字典
    """
    text_indices = torch.stack([item['text_indices'] for item in batch])
    aspect_indices = torch.stack([item['aspect_indices'] for item in batch])
    aspect_masks = torch.stack([item['aspect_masks'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    valid_mask = torch.stack([item['valid_mask'] for item in batch])
    num_aspects = torch.tensor([item['num_aspects'] for item in batch], dtype=torch.long)

    return {
        'text_indices': text_indices,        # [batch, seq_len]
        'aspect_indices': aspect_indices,    # [batch, max_num_aspects, aspect_len]
        'aspect_masks': aspect_masks,        # [batch, max_num_aspects, seq_len]
        'labels': labels,                    # [batch, max_num_aspects]
        'valid_mask': valid_mask,            # [batch, max_num_aspects]
        'num_aspects': num_aspects           # [batch]
    }


def create_multi_aspect_loader(
    dataframe: pd.DataFrame,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    創建多面向 DataLoader

    參數:
        dataframe: 預處理後的 DataFrame
        batch_size: 批次大小
        shuffle: 是否打亂
        num_workers: 工作執行緒數
        pin_memory: 是否使用固定記憶體

    返回:
        DataLoader 實例
    """
    dataset = MultiAspectDataset(dataframe)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=multi_aspect_collate_fn
    )

    return data_loader


def load_glove_embeddings(
    glove_path: str,
    word2idx: Dict[str, int],
    embedding_dim: int = 300
) -> np.ndarray:
    """
    載入 GloVe 詞嵌入

    參數:
        glove_path: GloVe 檔案路徑
        word2idx: 詞彙表映射
        embedding_dim: 嵌入維度

    返回:
        嵌入矩陣 [vocab_size, embedding_dim]
    """
    import os

    if not os.path.exists(glove_path):
        print(f"警告: GloVe 檔案不存在 ({glove_path})，使用隨機初始化")
        embedding_matrix = np.random.uniform(
            -0.25, 0.25, (len(word2idx), embedding_dim)
        )
        embedding_matrix[0] = 0  # PAD 向量設為 0
        return embedding_matrix

    print(f"載入 GloVe 嵌入: {glove_path}")

    # 初始化嵌入矩陣
    embedding_matrix = np.random.uniform(
        -0.25, 0.25, (len(word2idx), embedding_dim)
    )
    embedding_matrix[0] = 0  # PAD 向量設為 0

    # 讀取 GloVe 檔案
    found_words = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]

            if word in word2idx:
                idx = word2idx[word]
                vector = np.array(parts[1:], dtype=np.float32)

                if len(vector) == embedding_dim:
                    embedding_matrix[idx] = vector
                    found_words += 1

    print(f"找到 {found_words}/{len(word2idx)} 個詞的預訓練嵌入")

    return embedding_matrix


if __name__ == "__main__":
    # 測試數據載入器
    print("測試數據載入器...")

    # 創建模擬數據
    data = {
        'text': ['great food', 'bad service', 'nice place'],
        'aspect': ['food', 'service', 'place'],
        'text_indices': [[1, 2, 0, 0], [3, 4, 0, 0], [5, 6, 0, 0]],
        'aspect_indices': [[1, 0], [4, 0], [6, 0]],
        'aspect_mask': [[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]],
        'label': [2, 0, 2]
    }

    df = pd.DataFrame(data)

    # 創建 DataLoader
    loader = create_data_loader(df, batch_size=2, shuffle=False)

    # 測試批次
    for batch in loader:
        print("\n批次樣本:")
        print(f"  text_indices shape: {batch['text_indices'].shape}")
        print(f"  aspect_indices shape: {batch['aspect_indices'].shape}")
        print(f"  aspect_mask shape: {batch['aspect_mask'].shape}")
        print(f"  labels shape: {batch['labels'].shape}")
        print(f"  labels: {batch['labels']}")
        break

    print("\n數據載入器測試完成！")
