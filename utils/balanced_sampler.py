"""
Balanced Batch Sampler
確保每個 batch 都有均衡的類別分布
"""

import torch
import numpy as np
from torch.utils.data import Sampler
from typing import Iterator, List
import random


class BalancedBatchSampler(Sampler):
    """
    平衡批次採樣器

    確保每個 batch 中包含相對均衡的三個類別樣本，
    避免某個類別被過度忽略或過度關注。

    策略：
    - 將數據按類別分組
    - 每個 batch 從每個類別中均勻採樣
    - 如果某個類別樣本不足，則從其他類別補充
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        drop_last: bool = False,
        seed: int = None
    ):
        """
        初始化平衡批次採樣器

        參數:
            labels: 標籤列表
            batch_size: 批次大小
            drop_last: 是否丟棄最後不完整的 batch
            seed: 隨機種子（確保可重現性）
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed

        # 按類別分組索引
        self.label_to_indices = {}
        for label in np.unique(self.labels):
            self.label_to_indices[label] = np.where(self.labels == label)[0].tolist()

        # 計算每個類別在一個 batch 中的樣本數
        num_classes = len(self.label_to_indices)
        self.samples_per_class = batch_size // num_classes
        self.remainder = batch_size % num_classes

        # 改進：使用最大類別的樣本數來計算批次數，確保使用所有數據
        max_samples = max(len(indices) for indices in self.label_to_indices.values())
        # 確保每個類別的樣本都至少被使用一次
        self.num_batches = (max_samples + self.samples_per_class - 1) // self.samples_per_class

        # 打印統計信息
        print(f"[BalancedBatchSampler] 類別分布: {', '.join([f'{k}:{len(v)}' for k, v in self.label_to_indices.items()])}")
        print(f"[BalancedBatchSampler] 每批次每類樣本數: {self.samples_per_class}, 總批次數: {self.num_batches}")

    def __iter__(self) -> Iterator[List[int]]:
        """生成平衡的批次索引"""
        # 設置隨機種子以確保可重現性
        if self.seed is not None:
            rng = random.Random(self.seed)
            np_rng = np.random.RandomState(self.seed)
        else:
            rng = random.Random()
            np_rng = np.random.RandomState()

        # 為每個類別創建擴展的索引列表（確保所有類別有相同數量的可用樣本）
        max_class_size = max(len(indices) for indices in self.label_to_indices.values())
        expanded_indices = {}

        for label, indices in self.label_to_indices.items():
            # 計算需要重複的次數
            repeat_times = (max_class_size + len(indices) - 1) // len(indices)
            # 創建重複的索引列表
            expanded = indices * repeat_times
            # 截斷到最大類別大小
            expanded = expanded[:max_class_size]
            # 打亂
            rng.shuffle(expanded)
            expanded_indices[label] = expanded

        # 生成每個 batch
        batch_indices_list = []
        for batch_idx in range(self.num_batches):
            batch = []

            # 從每個類別中採樣
            for label in sorted(self.label_to_indices.keys()):
                start_idx = batch_idx * self.samples_per_class
                end_idx = start_idx + self.samples_per_class

                # 處理最後一個批次的餘數
                if batch_idx == self.num_batches - 1 and not self.drop_last:
                    if label < self.remainder:
                        end_idx += 1

                # 從擴展的索引列表中取樣本
                if start_idx < len(expanded_indices[label]):
                    batch.extend(expanded_indices[label][start_idx:min(end_idx, len(expanded_indices[label]))])

            # 打亂 batch 內的順序
            rng.shuffle(batch)
            batch_indices_list.append(batch)

        # 返回批次
        for batch in batch_indices_list:
            if len(batch) > 0:  # 確保不返回空批次
                yield batch

    def __len__(self) -> int:
        """返回批次數"""
        return self.num_batches


class WeightedBalancedBatchSampler(Sampler):
    """
    加權平衡批次採樣器

    類似 BalancedBatchSampler，但允許為不同類別設置不同的採樣權重。
    例如：[1.5, 2.0, 1.5] 會讓中性類別有更多樣本出現在 batch 中。
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int,
        class_weights: List[float] = None,
        drop_last: bool = False
    ):
        """
        初始化加權平衡批次採樣器

        參數:
            labels: 標籤列表
            batch_size: 批次大小
            class_weights: 類別權重 [負面, 中性, 正面]
            drop_last: 是否丟棄最後不完整的 batch
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 按類別分組索引
        self.label_to_indices = {}
        for label in np.unique(self.labels):
            self.label_to_indices[label] = np.where(self.labels == label)[0].tolist()

        num_classes = len(self.label_to_indices)

        # 處理類別權重
        if class_weights is None:
            class_weights = [1.0] * num_classes

        # 標準化權重
        total_weight = sum(class_weights)
        normalized_weights = [w / total_weight for w in class_weights]

        # 計算每個類別在一個 batch 中的樣本數
        self.samples_per_class = {}
        allocated = 0
        for label, weight in enumerate(normalized_weights):
            n_samples = int(batch_size * weight)
            self.samples_per_class[label] = n_samples
            allocated += n_samples

        # 處理剩餘樣本（分配給權重最高的類別）
        remainder = batch_size - allocated
        if remainder > 0:
            max_weight_label = np.argmax(normalized_weights)
            self.samples_per_class[max_weight_label] += remainder

        # 計算總批次數
        min_samples = min(len(indices) for indices in self.label_to_indices.values())
        min_samples_per_class = min(self.samples_per_class.values())
        if min_samples_per_class > 0:
            self.num_batches = min_samples // min_samples_per_class
        else:
            self.num_batches = 0

    def __iter__(self) -> Iterator[List[int]]:
        """生成加權平衡的批次索引"""
        # 為每個類別創建打亂的索引列表
        shuffled_indices = {}
        for label, indices in self.label_to_indices.items():
            shuffled = indices.copy()
            random.shuffle(shuffled)
            shuffled_indices[label] = shuffled

        # 追蹤每個類別已使用的樣本數
        counters = {label: 0 for label in self.label_to_indices.keys()}

        # 生成每個 batch
        for _ in range(self.num_batches):
            batch = []

            # 從每個類別中按權重採樣
            for label in sorted(self.label_to_indices.keys()):
                indices = shuffled_indices[label]
                counter = counters[label]
                n_samples = self.samples_per_class[label]

                # 採樣（循環使用如果樣本不足）
                for _ in range(n_samples):
                    if counter >= len(indices):
                        # 重新打亂並從頭開始
                        random.shuffle(indices)
                        counter = 0
                    batch.append(indices[counter])
                    counter += 1

                counters[label] = counter

            # 打亂 batch 內的順序
            random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        """返回批次數"""
        return self.num_batches


if __name__ == "__main__":
    # 測試平衡批次採樣器
    print("測試 BalancedBatchSampler...")

    # 創建模擬數據（不平衡）
    labels = [0] * 100 + [1] * 50 + [2] * 150  # 0: 100, 1: 50, 2: 150
    print(f"數據分布: 0={labels.count(0)}, 1={labels.count(1)}, 2={labels.count(2)}")

    # 創建採樣器
    sampler = BalancedBatchSampler(labels, batch_size=30, drop_last=False)
    print(f"批次數: {len(sampler)}")

    # 檢查前幾個 batch 的類別分布
    for i, batch_indices in enumerate(sampler):
        if i >= 3:  # 只檢查前 3 個 batch
            break
        batch_labels = [labels[idx] for idx in batch_indices]
        dist = {
            0: batch_labels.count(0),
            1: batch_labels.count(1),
            2: batch_labels.count(2)
        }
        print(f"Batch {i+1}: {dist} (總數: {len(batch_indices)})")

    # 測試加權平衡採樣器
    print("\n測試 WeightedBalancedBatchSampler...")
    weighted_sampler = WeightedBalancedBatchSampler(
        labels,
        batch_size=30,
        class_weights=[1.5, 2.0, 1.5],  # 中性類別權重更高
        drop_last=False
    )
    print(f"批次數: {len(weighted_sampler)}")

    # 檢查前幾個 batch 的類別分布
    for i, batch_indices in enumerate(weighted_sampler):
        if i >= 3:
            break
        batch_labels = [labels[idx] for idx in batch_indices]
        dist = {
            0: batch_labels.count(0),
            1: batch_labels.count(1),
            2: batch_labels.count(2)
        }
        print(f"Batch {i+1}: {dist} (總數: {len(batch_indices)})")

    print("\n✓ 平衡批次採樣器測試完成！")
