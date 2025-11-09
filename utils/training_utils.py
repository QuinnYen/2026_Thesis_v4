"""
訓練工具模組
包含學習率調度器、資料增強等訓練相關功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math


class WarmupCosineAnnealingLR(_LRScheduler):
    """
    帶 Warm-up 的 Cosine Annealing 學習率調度器

    階段:
        1. Warm-up: 線性增長 (0 -> initial_lr)
        2. Cosine Annealing: 餘弦衰減 (initial_lr -> min_lr)

    優點:
        - Warm-up 防止訓練初期梯度爆炸
        - Cosine 衰減提供平滑的學習率下降
        - 在訓練後期保持較小的學習率細調
    """

    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        """
        初始化調度器

        參數:
            optimizer: 優化器
            warmup_epochs: warm-up 輪數
            max_epochs: 總訓練輪數
            min_lr: 最小學習率
            last_epoch: 上一個 epoch（用於恢復訓練）
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """計算當前學習率"""
        if self.last_epoch < self.warmup_epochs:
            # Warm-up 階段：線性增長
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine Annealing 階段
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_decay
                for base_lr in self.base_lrs
            ]


class WarmupLinearScheduler(_LRScheduler):
    """
    帶 Warm-up 的線性學習率調度器

    BERT 常用的調度策略
    """

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0, last_epoch=-1):
        """
        初始化調度器

        參數:
            optimizer: 優化器
            warmup_steps: warm-up 步數
            total_steps: 總訓練步數
            min_lr: 最小學習率
            last_epoch: 上一個 epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupLinearScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """計算當前學習率"""
        if self.last_epoch < self.warmup_steps:
            # Warm-up 階段
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 線性衰減階段
            progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return [
                self.min_lr + (base_lr - self.min_lr) * (1 - progress)
                for base_lr in self.base_lrs
            ]


class EmbeddingMixup:
    """
    Embedding 層面的 Mixup 資料增強

    在 embedding 空間進行樣本混合，而非原始輸入空間
    適用於 NLP 任務，不會產生無意義的詞彙組合

    論文: mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    """

    def __init__(self, alpha=0.2, enabled=True):
        """
        初始化 Mixup

        參數:
            alpha: Beta 分佈參數（alpha 越大，混合越均勻）
            enabled: 是否啟用
        """
        self.alpha = alpha
        self.enabled = enabled

    def __call__(self, embeddings, labels):
        """
        對 embeddings 進行 Mixup

        參數:
            embeddings: [batch, seq_len, hidden_dim]
            labels: [batch]

        返回:
            mixed_embeddings: [batch, seq_len, hidden_dim]
            mixed_labels: [batch, num_classes] (one-hot with mixing)
            lam: 混合係數
        """
        if not self.enabled or not self.training:
            return embeddings, labels, 1.0

        batch_size = embeddings.size(0)

        # 從 Beta 分佈採樣混合係數
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 隨機打亂索引
        index = torch.randperm(batch_size).to(embeddings.device)

        # 混合 embeddings
        mixed_embeddings = lam * embeddings + (1 - lam) * embeddings[index]

        # 混合標籤（轉為 one-hot 後混合）
        num_classes = labels.max().item() + 1
        labels_a = F.one_hot(labels, num_classes).float()
        labels_b = F.one_hot(labels[index], num_classes).float()
        mixed_labels = lam * labels_a + (1 - lam) * labels_b

        return mixed_embeddings, mixed_labels, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """
        Mixup 的損失函數

        參數:
            criterion: 原始損失函數
            pred: 模型預測 [batch, num_classes]
            y_a: 第一個樣本的標籤
            y_b: 第二個樣本的標籤
            lam: 混合係數
        """
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ManifoldMixup:
    """
    Manifold Mixup - 在隱藏層進行混合

    比標準 Mixup 更強的正則化效果
    論文: Manifold Mixup (https://arxiv.org/abs/1806.05236)
    """

    def __init__(self, alpha=0.2, enabled=True, mix_prob=0.5):
        """
        初始化 Manifold Mixup

        參數:
            alpha: Beta 分佈參數
            enabled: 是否啟用
            mix_prob: 執行混合的機率
        """
        self.alpha = alpha
        self.enabled = enabled
        self.mix_prob = mix_prob

    def __call__(self, hidden_state, labels):
        """
        對隱藏狀態進行 Mixup

        參數:
            hidden_state: 任意形狀的隱藏狀態
            labels: [batch]

        返回:
            mixed_hidden: 混合後的隱藏狀態
            labels_a: 第一個樣本標籤
            labels_b: 第二個樣本標籤
            lam: 混合係數
        """
        if not self.enabled or not self.training or np.random.rand() > self.mix_prob:
            return hidden_state, labels, labels, 1.0

        batch_size = hidden_state.size(0)

        # 採樣混合係數
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
        else:
            lam = 1.0

        # 隨機打亂
        index = torch.randperm(batch_size).to(hidden_state.device)

        # 混合隱藏狀態
        mixed_hidden = lam * hidden_state + (1 - lam) * hidden_state[index]

        labels_a = labels
        labels_b = labels[index]

        return mixed_hidden, labels_a, labels_b, lam


class AdversarialTraining:
    """
    對抗訓練 (Adversarial Training)

    在 embedding 上添加對抗性擾動，提升模型魯棒性

    方法: Fast Gradient Method (FGM) 或 Projected Gradient Descent (PGD)
    """

    def __init__(self, model, epsilon=1.0, method='fgm'):
        """
        初始化對抗訓練

        參數:
            model: 模型
            epsilon: 擾動幅度
            method: 'fgm' 或 'pgd'
        """
        self.model = model
        self.epsilon = epsilon
        self.method = method
        self.backup = {}

    def attack(self, emb_name='bert_absa.bert.embeddings.word_embeddings'):
        """
        生成對抗樣本

        參數:
            emb_name: embedding 層的名稱
        """
        # 找到 embedding 層
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # 保存原始值
                self.backup[name] = param.data.clone()

                # 計算梯度方向
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    # 添加對抗擾動
                    r_adv = self.epsilon * param.grad / norm
                    param.data.add_(r_adv)

    def restore(self, emb_name='bert_absa.bert.embeddings.word_embeddings'):
        """
        恢復原始 embedding

        參數:
            emb_name: embedding 層的名稱
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}


def get_scheduler(optimizer, scheduler_type='cosine', **kwargs):
    """
    工廠函數：創建學習率調度器

    參數:
        optimizer: 優化器
        scheduler_type: 調度器類型
        **kwargs: 其他參數

    返回:
        調度器實例
    """
    if scheduler_type == 'cosine':
        return WarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=kwargs.get('warmup_epochs', 2),
            max_epochs=kwargs.get('max_epochs', 20),
            min_lr=kwargs.get('min_lr', 1e-6)
        )

    elif scheduler_type == 'linear':
        return WarmupLinearScheduler(
            optimizer,
            warmup_steps=kwargs.get('warmup_steps', 100),
            total_steps=kwargs.get('total_steps', 1000),
            min_lr=kwargs.get('min_lr', 0)
        )

    elif scheduler_type == 'reduce_on_plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'max'),
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 3),
            min_lr=kwargs.get('min_lr', 1e-6)
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


if __name__ == "__main__":
    print("測試訓練工具...")

    # 測試學習率調度器
    print("\n1. 測試 Cosine Annealing with Warmup")
    model = nn.Linear(10, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    scheduler = WarmupCosineAnnealingLR(optimizer, warmup_epochs=2, max_epochs=10, min_lr=1e-6)

    print("Epoch | Learning Rate")
    print("-" * 30)
    for epoch in range(10):
        lr = scheduler.get_last_lr()[0]
        print(f"{epoch:5d} | {lr:.6e}")
        scheduler.step()

    # 測試 Mixup
    print("\n2. 測試 Embedding Mixup")
    mixup = EmbeddingMixup(alpha=0.2)
    embeddings = torch.randn(8, 50, 768)  # [batch, seq_len, hidden]
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    mixed_emb, mixed_labels, lam = mixup(embeddings, labels)
    print(f"Original shape: {embeddings.shape}")
    print(f"Mixed shape: {mixed_emb.shape}")
    print(f"Mixed labels shape: {mixed_labels.shape}")
    print(f"Lambda: {lam:.3f}")

    print("\n測試完成！")
