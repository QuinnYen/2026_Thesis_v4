"""
Multi-Aspect Dataset for PyTorch

支持變長 aspects 的 batch 處理
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, AutoTokenizer
from typing import List, Dict, Tuple
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.loader_semeval14 import MultiAspectSample


class MultiAspectBERTDataset(Dataset):
    """
    Multi-Aspect BERT Dataset

    每個樣本包含:
        - text: 句子文本
        - aspects: 多個 aspects
        - labels: 每個 aspect 的情感標籤
        - soft_labels: (可選) Claude 生成的 soft labels

    重要：使用正確的 BERT sentence-pair 編碼格式：
        [CLS] text [SEP] aspect [SEP]
        token_type_ids: 0 0 0 ... 0 1 1 ... 1
    """

    def __init__(
        self,
        samples: List[MultiAspectSample],
        tokenizer: AutoTokenizer,
        max_text_len: int = 128,
        max_aspect_len: int = 10,
        max_num_aspects: int = 8,
        soft_labels_dict: dict = None  # {(text, aspect): [neg, neu, pos]}
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_seq_len = max_text_len  # 總序列長度 (text + aspect)
        self.max_num_aspects = max_num_aspects
        self.soft_labels_dict = soft_labels_dict  # {(text, aspect): [neg, neu, pos]}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        獲取單個樣本

        使用正確的 BERT sentence-pair 編碼：
            [CLS] text [SEP] aspect [SEP]
            token_type_ids: 0...0 1...1

        返回:
            {
                'pair_input_ids': [num_aspects, seq_len],
                'pair_attention_mask': [num_aspects, seq_len],
                'pair_token_type_ids': [num_aspects, seq_len],
                'labels': [num_aspects],
                'aspect_mask': [num_aspects],
                'num_aspects': scalar,
                'is_virtual': [num_aspects]
            }
        """
        sample = self.samples[idx]

        # 為每個 aspect 創建 sentence-pair 編碼
        pair_input_ids_list = []
        pair_attention_mask_list = []
        pair_token_type_ids_list = []

        for aspect in sample.aspects:
            # 移除虛擬標記
            aspect_text = aspect.replace('<VIRTUAL>', '').strip()

            # 使用 BERT sentence-pair 編碼：
            # tokenizer(text, aspect) 會自動生成:
            # [CLS] text [SEP] aspect [SEP]
            # token_type_ids: 0 0 0 ... 0 1 1 ... 1
            encoding = self.tokenizer(
                sample.text,
                aspect_text,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation='only_first',  # 只截斷第一個句子(text)，保留完整aspect
                return_tensors='pt'
            )

            pair_input_ids_list.append(encoding['input_ids'].squeeze(0))
            pair_attention_mask_list.append(encoding['attention_mask'].squeeze(0))

            # DeBERTa 不返回 token_type_ids，手動創建全 0
            if 'token_type_ids' in encoding:
                pair_token_type_ids_list.append(encoding['token_type_ids'].squeeze(0))
            else:
                pair_token_type_ids_list.append(torch.zeros_like(encoding['input_ids'].squeeze(0)))

        # Stack all aspects
        pair_input_ids = torch.stack(pair_input_ids_list)      # [num_aspects, seq_len]
        pair_attention_mask = torch.stack(pair_attention_mask_list)
        pair_token_type_ids = torch.stack(pair_token_type_ids_list)

        # Labels
        labels = torch.tensor(sample.labels, dtype=torch.long)

        # Aspect mask
        num_aspects = len(sample.aspects)
        aspect_mask = torch.ones(num_aspects, dtype=torch.bool)

        # Virtual aspect markers
        is_virtual = torch.tensor(sample.is_virtual, dtype=torch.bool)

        # Soft labels (if available)
        soft_labels = None
        if self.soft_labels_dict is not None:
            soft_labels_list = []
            for aspect in sample.aspects:
                # 移除虛擬標記
                aspect_text = aspect.replace('<VIRTUAL>', '').strip()
                key = (sample.text, aspect_text)

                if key in self.soft_labels_dict:
                    soft_labels_list.append(self.soft_labels_dict[key])
                else:
                    # 如果找不到，使用 hard label 轉換的 one-hot
                    label_idx = sample.labels[len(soft_labels_list)]
                    one_hot = [0.0, 0.0, 0.0]
                    if 0 <= label_idx <= 2:
                        one_hot[label_idx] = 1.0
                    soft_labels_list.append(one_hot)

            soft_labels = torch.tensor(soft_labels_list, dtype=torch.float)

        result = {
            'pair_input_ids': pair_input_ids,
            'pair_attention_mask': pair_attention_mask,
            'pair_token_type_ids': pair_token_type_ids,
            'labels': labels,
            'aspect_mask': aspect_mask,
            'num_aspects': torch.tensor(num_aspects, dtype=torch.long),
            'is_virtual': is_virtual
        }

        if soft_labels is not None:
            result['soft_labels'] = soft_labels

        return result


def multiaspect_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length aspects

    處理變長 aspects 的 batch

    新格式：每個 aspect 對應一個完整的 sentence-pair 編碼
        [CLS] text [SEP] aspect [SEP]

    參數:
        batch: List of samples from __getitem__

    返回:
        Batched tensors with padding
    """
    # 找到最大 aspect 數量
    max_aspects_in_batch = max(item['num_aspects'].item() for item in batch)

    # 檢查是否有 soft labels
    has_soft_labels = 'soft_labels' in batch[0]

    # 準備 batch tensors
    batch_size = len(batch)

    pair_input_ids = []
    pair_attention_mask = []
    pair_token_type_ids = []
    labels = []
    aspect_mask = []
    num_aspects = []
    is_virtual = []
    soft_labels = [] if has_soft_labels else None

    for item in batch:
        # 獲取當前樣本的資訊
        curr_aspects = item['num_aspects'].item()
        seq_len = item['pair_input_ids'].size(1)

        if curr_aspects < max_aspects_in_batch:
            # Need padding
            pad_count = max_aspects_in_batch - curr_aspects

            # Pad pair_input_ids
            padded_pair_ids = torch.cat([
                item['pair_input_ids'],
                torch.zeros(pad_count, seq_len, dtype=torch.long)
            ], dim=0)

            # Pad pair_attention_mask
            padded_pair_mask = torch.cat([
                item['pair_attention_mask'],
                torch.zeros(pad_count, seq_len, dtype=torch.long)
            ], dim=0)

            # Pad pair_token_type_ids
            padded_pair_type_ids = torch.cat([
                item['pair_token_type_ids'],
                torch.zeros(pad_count, seq_len, dtype=torch.long)
            ], dim=0)

            # Pad labels (用 -100 表示 ignore)
            padded_labels = torch.cat([
                item['labels'],
                torch.full((pad_count,), -100, dtype=torch.long)
            ], dim=0)

            # Pad aspect_mask
            padded_valid_mask = torch.cat([
                item['aspect_mask'],
                torch.zeros(pad_count, dtype=torch.bool)
            ], dim=0)

            # Pad is_virtual
            padded_is_virtual = torch.cat([
                item['is_virtual'],
                torch.zeros(pad_count, dtype=torch.bool)
            ], dim=0)

            # Pad soft_labels (if present)
            if has_soft_labels:
                padded_soft_labels = torch.cat([
                    item['soft_labels'],
                    torch.zeros(pad_count, 3, dtype=torch.float)  # 3 classes
                ], dim=0)

        else:
            padded_pair_ids = item['pair_input_ids']
            padded_pair_mask = item['pair_attention_mask']
            padded_pair_type_ids = item['pair_token_type_ids']
            padded_labels = item['labels']
            padded_valid_mask = item['aspect_mask']
            padded_is_virtual = item['is_virtual']
            if has_soft_labels:
                padded_soft_labels = item['soft_labels']

        pair_input_ids.append(padded_pair_ids)
        pair_attention_mask.append(padded_pair_mask)
        pair_token_type_ids.append(padded_pair_type_ids)
        labels.append(padded_labels)
        aspect_mask.append(padded_valid_mask)
        num_aspects.append(item['num_aspects'])
        is_virtual.append(padded_is_virtual)
        if has_soft_labels:
            soft_labels.append(padded_soft_labels)

    # Stack into batch
    result = {
        'pair_input_ids': torch.stack(pair_input_ids),          # [batch, max_aspects, seq_len]
        'pair_attention_mask': torch.stack(pair_attention_mask), # [batch, max_aspects, seq_len]
        'pair_token_type_ids': torch.stack(pair_token_type_ids), # [batch, max_aspects, seq_len]
        'labels': torch.stack(labels),                           # [batch, max_aspects]
        'aspect_mask': torch.stack(aspect_mask),                 # [batch, max_aspects]
        'num_aspects': torch.stack(num_aspects),                 # [batch]
        'is_virtual': torch.stack(is_virtual)                    # [batch, max_aspects]
    }

    if has_soft_labels:
        result['soft_labels'] = torch.stack(soft_labels)         # [batch, max_aspects, 3]

    return result


def _build_stratified_sampler(dataset: 'MultiAspectBERTDataset'):
    """
    Multi-aspect 版 WeightedRandomSampler。
    每個樣本的 weight = 其所有有效 aspect 標籤中最稀少類別的 inverse freq。
    自動判斷：若 minority class 佔比 >= 5%，class_weights 已足夠，回傳 None（不採樣）。
    """
    from torch.utils.data import WeightedRandomSampler

    # 第一遍：統計全域類別頻率
    class_counts = [0, 0, 0]  # [Neg, Neu, Pos]
    for sample in dataset.samples:
        for lbl in sample.labels:
            if lbl in (0, 1, 2):
                class_counts[lbl] += 1

    total = sum(class_counts)
    minority_ratio = min(class_counts) / max(total, 1)

    # 自動判斷：minority >= 5% 時 class_weights 已足夠，不需要 sampler
    MINORITY_THRESHOLD = 0.05
    if minority_ratio >= MINORITY_THRESHOLD:
        print(f"[Stratified Sampler] 跳過（minority={minority_ratio*100:.1f}% >= {MINORITY_THRESHOLD*100:.0f}%，class_weights 已足夠）")
        return None

    # 反比頻率 weight（避免除以 0）
    class_weights = [total / max(c, 1) for c in class_counts]
    print(f"[Stratified Sampler] 啟用（minority={minority_ratio*100:.1f}% < {MINORITY_THRESHOLD*100:.0f}%）class_counts={class_counts}, class_weights={[f'{w:.2f}' for w in class_weights]}")

    # 第二遍：每個樣本取其最稀少 class 的 weight
    sample_weights = []
    for sample in dataset.samples:
        valid_labels = [lbl for lbl in sample.labels if lbl in (0, 1, 2)]
        if valid_labels:
            w = max(class_weights[lbl] for lbl in valid_labels)
        else:
            w = 1.0
        sample_weights.append(w)

    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_multiaspect_dataloaders(
    train_samples: List[MultiAspectSample],
    val_samples: List[MultiAspectSample],
    test_samples: List[MultiAspectSample],
    tokenizer: AutoTokenizer,
    batch_size: int = 16,
    max_text_len: int = 128,
    max_aspect_len: int = 10,
    max_num_aspects: int = 8,
    num_workers: int = 0,  # Windows subprocess 環境下 num_workers>0 會造成 deadlock
    soft_labels_dict: dict = None,  # {(text, aspect): [neg, neu, pos]}
    use_stratified_sampler: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    創建 DataLoader

    參數:
        train_samples/val_samples/test_samples: 各分割樣本
        tokenizer: BERT tokenizer
        batch_size: Batch size
        max_text_len: 最大序列長度（text + aspect）
        max_aspect_len: 已不使用，保留介面兼容性
        max_num_aspects: 最大 aspect 數量（用於 padding）
        num_workers: DataLoader workers（Windows 建議設 0）
        soft_labels_dict: 知識蒸餾用 soft labels（loss_type='distill' 時啟用）
        use_stratified_sampler: 啟用 Stratified Batch Sampler 均衡類別訓練訊號

    返回:
        (train_loader, val_loader, test_loader)
    """
    # 創建 Datasets
    # 只有訓練集使用 soft labels
    train_dataset = MultiAspectBERTDataset(
        train_samples, tokenizer, max_text_len, max_aspect_len, max_num_aspects,
        soft_labels_dict=soft_labels_dict
    )

    # 驗證集和測試集不使用 soft labels
    val_dataset = MultiAspectBERTDataset(
        val_samples, tokenizer, max_text_len, max_aspect_len, max_num_aspects,
        soft_labels_dict=None
    )

    test_dataset = MultiAspectBERTDataset(
        test_samples, tokenizer, max_text_len, max_aspect_len, max_num_aspects,
        soft_labels_dict=None
    )

    # 創建 DataLoaders
    sampler = _build_stratified_sampler(train_dataset) if use_stratified_sampler else None
    if sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=multiaspect_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=multiaspect_collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multiaspect_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multiaspect_collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    # 簡化輸出：DataLoader 統計已在主訓練腳本中顯示

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # 測試 Dataset 和 DataLoader
    from pathlib import Path
    from datasets.loader_semeval14 import load_multiaspect_data
    from transformers import AutoTokenizer

    # 設置路徑
    project_root = Path(__file__).parent.parent
    train_path = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Train_v2.xml'
    test_path = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Test_Gold.xml'

    # 加載數據
    print("加載 Multi-Aspect 數據...")
    train_samples, test_samples = load_multiaspect_data(
        train_path=str(train_path),
        test_path=str(test_path),
        min_aspects=2,
        include_single_aspect=True,
        virtual_aspect_mode='overall'
    )

    # 創建驗證集（從訓練集分割）
    val_size = int(0.1 * len(train_samples))
    val_samples = train_samples[:val_size]
    train_samples = train_samples[val_size:]

    print(f"\n數據分割:")
    print(f"  訓練: {len(train_samples)}")
    print(f"  驗證: {len(val_samples)}")
    print(f"  測試: {len(test_samples)}")

    # 加載 tokenizer
    print("\n加載 DeBERTa tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-base')

    # 創建 DataLoaders
    print("\n創建 DataLoaders...")
    train_loader, val_loader, test_loader = create_multiaspect_dataloaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        tokenizer=tokenizer,
        batch_size=8,
        max_text_len=128,
        max_aspect_len=10,
        max_num_aspects=8
    )

    # 測試一個 batch
    print("\n測試一個 batch (新 sentence-pair 格式):")
    for batch in train_loader:
        print(f"  pair_input_ids: {batch['pair_input_ids'].shape}")
        print(f"  pair_attention_mask: {batch['pair_attention_mask'].shape}")
        print(f"  pair_token_type_ids: {batch['pair_token_type_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        print(f"  aspect_mask: {batch['aspect_mask'].shape}")
        print(f"  num_aspects: {batch['num_aspects']}")

        # 檢查第一個樣本
        print(f"\n第一個樣本:")
        print(f"  序列長度: {batch['pair_attention_mask'][0, 0].sum().item()}")
        print(f"  Aspects 數量: {batch['num_aspects'][0].item()}")
        print(f"  Labels: {batch['labels'][0]}")
        print(f"  Aspect mask: {batch['aspect_mask'][0]}")
        print(f"  Is virtual: {batch['is_virtual'][0]}")

        # 驗證 token_type_ids 格式
        print(f"\n  Token Type IDs 驗證 (第一個 aspect):")
        type_ids = batch['pair_token_type_ids'][0, 0]  # [seq_len]
        sep_positions = (type_ids[:-1] != type_ids[1:]).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            print(f"    Segment 切換位置: {sep_positions.tolist()}")
            print(f"    Type 0 (text) 長度: {(type_ids == 0).sum().item()}")
            print(f"    Type 1 (aspect) 長度: {(type_ids == 1).sum().item()}")

        break

    print("\n✅ Dataset 和 DataLoader 測試通過！")
