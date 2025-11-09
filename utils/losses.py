"""
自定義損失函數模組
提供 Focal Loss 等專門針對不平衡分類的損失函數
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance

    論文: Focal Loss for Dense Object Detection (https://arxiv.org/abs/1708.02002)

    公式: FL(pt) = -α_t * (1-pt)^γ * log(pt)

    參數:
        alpha: 類別權重，可以是單個值或每個類別的權重列表
               - 單個值: 所有類別使用相同權重
               - 列表: [負面權重, 中性權重, 正面權重]
        gamma: focusing 參數，控制難易樣本的權重差異
               - gamma=0: 等同於 CrossEntropyLoss
               - gamma=2: 標準設置（推薦）
               - gamma越大: 越關注難分類樣本
        reduction: 'none' | 'mean' | 'sum'

    用途:
        - 解決類別不平衡問題
        - 關注難分類樣本（如中性情感）
        - 減少簡單樣本對損失的主導
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        初始化 Focal Loss

        參數:
            alpha: 類別權重 [num_classes] 或單個浮點數
            gamma: focusing 參數
            reduction: 損失聚合方式
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # 處理 alpha 參數
        if alpha is not None:
            if isinstance(alpha, (list, tuple)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = torch.tensor([alpha], dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        """
        前向傳播

        參數:
            inputs: 模型輸出 logits [batch_size, num_classes]
            targets: 真實標籤 [batch_size]

        返回:
            focal loss 值
        """
        # 計算交叉熵損失（不進行 reduction）
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')

        # 計算 pt（模型對正確類別的預測概率）
        pt = torch.exp(-ce_loss)

        # 計算 focal loss
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # 應用類別權重 alpha
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)

            # 為每個樣本選擇對應類別的 alpha
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss

        # 聚合損失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ClassBalancedLoss(nn.Module):
    """
    類別平衡損失（基於樣本數量自動計算權重）

    使用有效樣本數（Effective Number of Samples）來計算類別權重
    論文: Class-Balanced Loss Based on Effective Number of Samples

    公式: weight_i = (1 - β) / (1 - β^n_i)
    其中 n_i 是類別 i 的樣本數
    """

    def __init__(self, samples_per_class, beta=0.9999, loss_type='focal', gamma=2.0):
        """
        初始化類別平衡損失

        參數:
            samples_per_class: 每個類別的樣本數 [num_classes]
            beta: 平衡參數（0-1之間，越接近1越關注少數類）
            loss_type: 'focal' 或 'ce'
            gamma: focal loss 的 gamma 參數
        """
        super(ClassBalancedLoss, self).__init__()

        # 計算有效樣本數權重
        effective_num = 1.0 - torch.pow(beta, torch.tensor(samples_per_class, dtype=torch.float32))
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(weights)  # 歸一化

        self.loss_type = loss_type
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=weights.tolist(), gamma=gamma)
        else:
            self.register_buffer('weight', weights)
            self.loss_fn = nn.CrossEntropyLoss(weight=weights)

    def forward(self, inputs, targets):
        return self.loss_fn(inputs, targets)


class LabelSmoothingCrossEntropy(nn.Module):
    """
    帶 Label Smoothing 的交叉熵損失

    將硬標籤 [0, 1, 0] 平滑為 [ε/K, 1-ε+ε/K, ε/K]
    可以防止模型過度自信，改善泛化能力
    """

    def __init__(self, epsilon=0.1, reduction='mean'):
        """
        初始化 Label Smoothing 損失

        參數:
            epsilon: 平滑參數（通常 0.1）
            reduction: 損失聚合方式
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        前向傳播

        參數:
            inputs: 模型輸出 logits [batch_size, num_classes]
            targets: 真實標籤 [batch_size]
        """
        num_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)

        # 創建平滑標籤
        targets_one_hot = F.one_hot(targets, num_classes).float()
        targets_smooth = targets_one_hot * (1 - self.epsilon) + self.epsilon / num_classes

        # 計算損失
        loss = -(targets_smooth * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_loss_function(loss_type='focal', num_classes=3, class_weights=None,
                      gamma=2.0, label_smoothing=0.0):
    """
    工廠函數：根據配置創建損失函數

    參數:
        loss_type: 'focal', 'ce', 'weighted_ce', 'cb_focal'
        num_classes: 類別數量
        class_weights: 類別權重（如果提供）
        gamma: focal loss 的 gamma 參數
        label_smoothing: label smoothing 參數

    返回:
        損失函數實例
    """
    if loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=gamma)

    elif loss_type == 'weighted_ce':
        if class_weights is not None:
            weights = torch.tensor(class_weights, dtype=torch.float32)
            return nn.CrossEntropyLoss(weight=weights, label_smoothing=label_smoothing)
        else:
            return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    elif loss_type == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


if __name__ == "__main__":
    # 測試 Focal Loss
    print("測試 Focal Loss...")

    # 模擬數據
    batch_size = 8
    num_classes = 3
    logits = torch.randn(batch_size, num_classes)
    targets = torch.tensor([0, 1, 2, 1, 1, 0, 2, 1])

    # 測試不同配置
    print("\n1. 標準 Focal Loss (gamma=2.0):")
    focal_loss = FocalLoss(gamma=2.0)
    loss = focal_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    print("\n2. Focal Loss with class weights [1.0, 2.0, 1.0]:")
    focal_loss_weighted = FocalLoss(alpha=[1.0, 2.0, 1.0], gamma=2.0)
    loss = focal_loss_weighted(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    print("\n3. 比較 CrossEntropyLoss:")
    ce_loss = nn.CrossEntropyLoss()
    loss = ce_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    print("\n4. Label Smoothing CrossEntropy:")
    ls_loss = LabelSmoothingCrossEntropy(epsilon=0.1)
    loss = ls_loss(logits, targets)
    print(f"   Loss: {loss.item():.4f}")

    print("\n測試完成！")
