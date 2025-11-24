"""
Focal Loss Implementation for Multi-Aspect ABSA

Focal Loss 專注於難分類樣本，自動降低易分類樣本的權重。
特別適合處理類別不平衡問題，如 Neutral 類別性能低的情況。

Reference:
    Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification

    Args:
        alpha: Weighting factor in range (0,1) or list of weights per class
               - If float: same weight for all classes
               - If list: [alpha_neg, alpha_neu, alpha_pos]
        gamma: Focusing parameter (gamma >= 0)
               - gamma=0: equivalent to CrossEntropyLoss
               - gamma=2: recommended default
        reduction: 'mean', 'sum', or 'none'

    Formula:
        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

        where p_t is the model's estimated probability for the true class
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

        # Handle alpha parameter
        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, alpha, alpha])
        elif isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha

    def forward(self, logits, labels):
        """
        Args:
            logits: [batch_size, num_classes] - raw logits
            labels: [batch_size] - ground truth labels

        Returns:
            loss: scalar loss value
        """
        # Compute softmax probabilities
        probs = F.softmax(logits, dim=-1)  # [batch, num_classes]

        # Get probability of the true class
        num_classes = logits.size(-1)
        labels_one_hot = F.one_hot(labels, num_classes).float()  # [batch, num_classes]
        p_t = (probs * labels_one_hot).sum(dim=-1)  # [batch]

        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma  # [batch]

        # Compute cross entropy: -log(p_t)
        ce_loss = -torch.log(p_t + 1e-8)  # [batch]

        # Combine focal weight and CE loss
        focal_loss = focal_weight * ce_loss  # [batch]

        # Apply alpha weighting if provided
        if self.alpha is not None:
            if self.alpha.device != logits.device:
                self.alpha = self.alpha.to(logits.device)

            # Get alpha for each sample's true class
            alpha_t = self.alpha[labels]  # [batch]
            focal_loss = alpha_t * focal_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiAspectFocalLoss(nn.Module):
    """
    Focal Loss for Multi-Aspect ABSA with masking and virtual aspect handling

    Args:
        alpha: Class weights [alpha_neg, alpha_neu, alpha_pos]
        gamma: Focusing parameter
        virtual_weight: Weight for virtual aspects (0.0-1.0)
    """

    def __init__(self, alpha=None, gamma=2.0, virtual_weight=0.5):
        super(MultiAspectFocalLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='none')
        self.virtual_weight = virtual_weight

    def forward(self, logits, labels, aspect_mask, is_virtual=None):
        """
        Args:
            logits: [batch, max_aspects, num_classes]
            labels: [batch, max_aspects]
            aspect_mask: [batch, max_aspects] - True for valid aspects
            is_virtual: [batch, max_aspects] - True for virtual aspects (optional)

        Returns:
            loss: scalar loss value
        """
        batch_size, max_aspects, num_classes = logits.shape

        # Flatten for loss computation
        logits_flat = logits.view(-1, num_classes)  # [batch*max_aspects, num_classes]
        labels_flat = labels.view(-1)  # [batch*max_aspects]
        aspect_mask_flat = aspect_mask.view(-1)  # [batch*max_aspects]

        # Filter out padding (-100 labels)
        valid_mask = (labels_flat != -100) & aspect_mask_flat

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute focal loss for valid aspects
        focal_loss = self.focal_loss(
            logits_flat[valid_mask],
            labels_flat[valid_mask]
        )  # [num_valid]

        # Apply virtual aspect weighting if provided
        if is_virtual is not None:
            is_virtual_flat = is_virtual.view(-1)[valid_mask]  # [num_valid]

            # Create weight tensor
            weights = torch.ones_like(focal_loss)
            weights[is_virtual_flat] = self.virtual_weight

            focal_loss = focal_loss * weights

        return focal_loss.mean()


class AdaptiveWeightedLoss(nn.Module):
    """
    Dynamically adjust class weights based on per-epoch performance

    更進階的損失函數，根據每個 epoch 的性能動態調整類別權重
    """

    def __init__(self, base_alpha=[1.0, 1.0, 1.0], gamma=2.0,
                 virtual_weight=0.5, adaptation_rate=0.1):
        super(AdaptiveWeightedLoss, self).__init__()
        self.base_alpha = torch.tensor(base_alpha)
        self.current_alpha = self.base_alpha.clone()
        self.gamma = gamma
        self.virtual_weight = virtual_weight
        self.adaptation_rate = adaptation_rate

        self.focal_loss = None
        self._update_focal_loss()

    def _update_focal_loss(self):
        """Update focal loss with current alpha"""
        self.focal_loss = MultiAspectFocalLoss(
            alpha=self.current_alpha.tolist(),
            gamma=self.gamma,
            virtual_weight=self.virtual_weight
        )

    def update_weights(self, class_f1_scores):
        """
        Update class weights based on F1 scores

        Args:
            class_f1_scores: [f1_neg, f1_neu, f1_pos]
        """
        # Inverse F1 weighting: lower F1 -> higher weight
        f1_tensor = torch.tensor(class_f1_scores)
        inverse_f1 = 1.0 / (f1_tensor + 0.1)  # Add 0.1 to avoid division by zero

        # Normalize
        new_weights = inverse_f1 / inverse_f1.sum() * 3

        # Smooth update with adaptation rate
        self.current_alpha = (1 - self.adaptation_rate) * self.current_alpha + \
                             self.adaptation_rate * new_weights

        # Update focal loss
        self._update_focal_loss()

    def forward(self, logits, labels, aspect_mask, is_virtual=None):
        """Forward pass"""
        return self.focal_loss(logits, labels, aspect_mask, is_virtual)


def compute_dynamic_class_weights(train_samples, num_classes=3, smooth=True):
    """
    根據訓練數據動態計算 class weights

    使用 inverse frequency weighting:
        weight_c = N / (num_classes * count_c)

    這樣少數類別會獲得更高的權重，多數類別權重較低

    Args:
        train_samples: 訓練樣本列表 (MultiAspectSample)
        num_classes: 類別數量 (default: 3 for Neg/Neu/Pos)
        smooth: 是否使用平滑 (避免極端權重)

    Returns:
        class_weights: [weight_neg, weight_neu, weight_pos]
    """
    from collections import Counter

    # 統計每個類別的數量
    label_counts = Counter()
    for sample in train_samples:
        for label in sample.labels:
            if label >= 0:  # 忽略 padding (-100)
                label_counts[label] += 1

    total = sum(label_counts.values())

    # 計算 inverse frequency weights
    weights = []
    for c in range(num_classes):
        count = label_counts.get(c, 1)  # 避免除以零
        # Inverse frequency: N / (K * n_c)
        w = total / (num_classes * count)
        weights.append(w)

    # 平滑處理：限制權重範圍在 [0.5, 3.0]
    if smooth:
        weights = [max(0.5, min(3.0, w)) for w in weights]

    # 正規化：確保平均權重為 1.0
    avg_weight = sum(weights) / len(weights)
    weights = [w / avg_weight for w in weights]

    return weights


def get_loss_function(loss_type='ce', **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type: 'ce', 'focal', 'adaptive'
        **kwargs: Additional arguments for loss function
            - alpha: Class weights [alpha_neg, alpha_neu, alpha_pos]
                     If 'auto', will be computed from train_samples
            - gamma: Focal loss gamma parameter
            - virtual_weight: Weight for virtual aspects
            - train_samples: Training samples (required if alpha='auto')

    Returns:
        loss_fn: Loss function
    """
    alpha = kwargs.get('alpha', None)

    # 動態計算 class weights
    if alpha == 'auto':
        train_samples = kwargs.get('train_samples', None)
        if train_samples is None:
            print("  [Warning] alpha='auto' but train_samples not provided, using uniform weights")
            alpha = None
        else:
            alpha = compute_dynamic_class_weights(train_samples)
            print(f"  [Dynamic Weights] Neg:{alpha[0]:.2f} Neu:{alpha[1]:.2f} Pos:{alpha[2]:.2f}")

    if loss_type == 'ce':
        # Standard Cross Entropy (wrapped for compatibility)
        return MultiAspectCrossEntropyLoss(
            virtual_weight=kwargs.get('virtual_weight', 0.5)
        )

    elif loss_type == 'focal':
        return MultiAspectFocalLoss(
            alpha=alpha,
            gamma=kwargs.get('gamma', 2.0),
            virtual_weight=kwargs.get('virtual_weight', 0.5)
        )

    elif loss_type == 'adaptive':
        return AdaptiveWeightedLoss(
            base_alpha=alpha if alpha else [1.0, 1.0, 1.0],
            gamma=kwargs.get('gamma', 2.0),
            virtual_weight=kwargs.get('virtual_weight', 0.5),
            adaptation_rate=kwargs.get('adaptation_rate', 0.1)
        )

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


class MultiAspectCrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss for Multi-Aspect ABSA
    (For compatibility and baseline comparison)
    """

    def __init__(self, virtual_weight=0.5):
        super(MultiAspectCrossEntropyLoss, self).__init__()
        self.virtual_weight = virtual_weight

    def forward(self, logits, labels, aspect_mask, is_virtual=None):
        """
        Args:
            logits: [batch, max_aspects, num_classes]
            labels: [batch, max_aspects]
            aspect_mask: [batch, max_aspects]
            is_virtual: [batch, max_aspects] (optional)

        Returns:
            loss: scalar
        """
        batch_size, max_aspects, num_classes = logits.shape

        # Flatten
        logits_flat = logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        aspect_mask_flat = aspect_mask.view(-1)

        # Filter valid aspects
        valid_mask = (labels_flat != -100) & aspect_mask_flat

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # Compute CE loss
        loss = F.cross_entropy(
            logits_flat[valid_mask],
            labels_flat[valid_mask],
            reduction='none'
        )

        # Apply virtual weighting
        if is_virtual is not None:
            is_virtual_flat = is_virtual.view(-1)[valid_mask]
            weights = torch.ones_like(loss)
            weights[is_virtual_flat] = self.virtual_weight
            loss = loss * weights

        return loss.mean()


# ============================================================================
# 測試和使用範例
# ============================================================================

def test_focal_loss():
    """Test focal loss implementation"""
    print("Testing Focal Loss...")

    # Create dummy data
    batch_size = 4
    max_aspects = 3
    num_classes = 3

    logits = torch.randn(batch_size, max_aspects, num_classes)
    labels = torch.randint(0, num_classes, (batch_size, max_aspects))
    aspect_mask = torch.tensor([
        [True, True, False],
        [True, True, True],
        [True, False, False],
        [True, True, False]
    ])
    is_virtual = torch.tensor([
        [False, True, False],
        [False, False, True],
        [True, False, False],
        [False, True, False]
    ])

    # Test different loss functions
    print("\n1. Standard CE Loss:")
    ce_loss = MultiAspectCrossEntropyLoss(virtual_weight=0.5)
    loss_ce = ce_loss(logits, labels, aspect_mask, is_virtual)
    print(f"   Loss: {loss_ce.item():.4f}")

    print("\n2. Focal Loss (gamma=2.0, no alpha):")
    focal_loss = MultiAspectFocalLoss(alpha=None, gamma=2.0, virtual_weight=0.5)
    loss_focal = focal_loss(logits, labels, aspect_mask, is_virtual)
    print(f"   Loss: {loss_focal.item():.4f}")

    print("\n3. Focal Loss with class weighting (alpha=[1.0, 2.0, 1.0]):")
    focal_loss_weighted = MultiAspectFocalLoss(
        alpha=[1.0, 2.0, 1.0],  # Increase weight for Neutral class
        gamma=2.0,
        virtual_weight=0.5
    )
    loss_focal_weighted = focal_loss_weighted(logits, labels, aspect_mask, is_virtual)
    print(f"   Loss: {loss_focal_weighted.item():.4f}")

    print("\n4. Adaptive Weighted Loss:")
    adaptive_loss = AdaptiveWeightedLoss(
        base_alpha=[1.0, 1.0, 1.0],
        gamma=2.0,
        virtual_weight=0.5
    )
    loss_adaptive = adaptive_loss(logits, labels, aspect_mask, is_virtual)
    print(f"   Loss (before update): {loss_adaptive.item():.4f}")

    # Simulate updating weights based on F1 scores
    class_f1 = [0.75, 0.45, 0.88]  # Low Neutral F1
    adaptive_loss.update_weights(class_f1)
    loss_adaptive_updated = adaptive_loss(logits, labels, aspect_mask, is_virtual)
    print(f"   Loss (after update): {loss_adaptive_updated.item():.4f}")
    print(f"   Updated weights: {adaptive_loss.current_alpha.tolist()}")

    print("\n✓ All tests passed!")


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss for ABSA

    核心思想：
        - 拉近同類別的樣本表示（特別是 Neutral）
        - 推遠不同類別的樣本表示
        - 學習更具區分性的特徵空間

    公式：
        L = -log( exp(z_i · z_p / τ) / Σ exp(z_i · z_k / τ) )

        其中：
        - z_i: anchor 樣本的特徵表示
        - z_p: 同類別的 positive 樣本
        - z_k: 所有其他樣本（包括 negative）
        - τ: temperature 參數（控制分布的尖銳程度）

    參考：
        Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)

    Args:
        temperature: 溫度參數，越小分布越尖銳（default: 0.07）
        base_temperature: 基礎溫度（用於正規化，default: 0.07）
    """

    def __init__(self, temperature: float = 0.07, base_temperature: float = 0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        計算 Supervised Contrastive Loss

        Args:
            features: [batch_size, feature_dim] - L2 正規化後的特徵向量
            labels: [batch_size] - 類別標籤
            mask: [batch_size] - 有效樣本的 mask（可選）

        Returns:
            loss: scalar contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # L2 正規化（確保特徵在單位球面上）
        features = F.normalize(features, p=2, dim=1)

        # 創建標籤 mask：同類別為 1，不同類別為 0
        labels = labels.contiguous().view(-1, 1)  # [batch, 1]
        label_mask = torch.eq(labels, labels.T).float().to(device)  # [batch, batch]

        # 排除自己（對角線設為 0）
        eye_mask = torch.eye(batch_size, device=device)
        label_mask = label_mask - eye_mask  # 同類別但不是自己

        # 計算相似度矩陣：z_i · z_j / τ
        similarity_matrix = torch.matmul(features, features.T) / self.temperature
        # [batch, batch]

        # 移除對角線的影響（設為很大的負數，softmax 後接近 0）
        logits_mask = torch.ones_like(similarity_matrix) - eye_mask
        similarity_matrix = similarity_matrix * logits_mask + (-1e9) * eye_mask

        # 應用有效樣本 mask
        if mask is not None:
            mask = mask.float().unsqueeze(1)  # [batch, 1]
            # 只保留有效樣本的相似度
            valid_mask = mask * mask.T  # [batch, batch]
            similarity_matrix = similarity_matrix * valid_mask + (-1e9) * (1 - valid_mask)
            label_mask = label_mask * valid_mask

        # 計算 log softmax（分母包含所有非自己的樣本）
        exp_logits = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # 只對同類別的 positive pairs 計算平均
        # 避免除零：當沒有同類別樣本時，跳過
        num_positives = label_mask.sum(dim=1)  # [batch]
        has_positives = (num_positives > 0)

        if not has_positives.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 計算每個 anchor 的 contrastive loss
        mean_log_prob_pos = (label_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
        # [batch]

        # 只對有 positive pairs 的樣本計算 loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss[has_positives].mean()

        return loss


class MultiAspectContrastiveLoss(nn.Module):
    """
    Multi-Aspect ABSA 專用的 Contrastive Loss

    處理：
        - 多 aspect 格式 [batch, max_aspects, feature_dim]
        - 有效 aspect mask
        - 與 Focal Loss 組合使用

    Args:
        temperature: 溫度參數
        contrastive_weight: 對比損失的權重（相對於分類損失）
    """

    def __init__(self, temperature: float = 0.07, contrastive_weight: float = 0.1):
        super(MultiAspectContrastiveLoss, self).__init__()
        self.scl = SupervisedContrastiveLoss(temperature=temperature)
        self.contrastive_weight = contrastive_weight

    def forward(self, features: torch.Tensor, labels: torch.Tensor,
                aspect_mask: torch.Tensor) -> torch.Tensor:
        """
        計算 Multi-Aspect Contrastive Loss

        Args:
            features: [batch, max_aspects, feature_dim] - 階層特徵表示
            labels: [batch, max_aspects] - 類別標籤（-100 為 padding）
            aspect_mask: [batch, max_aspects] - 有效 aspect 的 mask

        Returns:
            loss: scalar contrastive loss
        """
        batch_size, max_aspects, feature_dim = features.shape
        device = features.device

        # 展平所有有效的 aspects
        # 過濾 padding 和無效 aspects
        valid_mask = aspect_mask & (labels != -100)  # [batch, max_aspects]

        # 收集有效的 features 和 labels
        valid_features = []
        valid_labels = []

        for i in range(batch_size):
            for j in range(max_aspects):
                if valid_mask[i, j]:
                    valid_features.append(features[i, j])  # [feature_dim]
                    valid_labels.append(labels[i, j])

        if len(valid_features) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Stack 為 tensor
        valid_features = torch.stack(valid_features)  # [num_valid, feature_dim]
        valid_labels = torch.tensor(valid_labels, device=device)  # [num_valid]

        # 計算 contrastive loss
        loss = self.scl(valid_features, valid_labels)

        return loss * self.contrastive_weight


def get_combined_loss_function(
    loss_type: str = 'focal',
    contrastive_weight: float = 0.1,
    contrastive_temperature: float = 0.07,
    **kwargs
):
    """
    創建組合損失函數（分類損失 + 對比損失）

    Args:
        loss_type: 分類損失類型 ('ce', 'focal', 'adaptive')
        contrastive_weight: 對比損失權重 (0.0 = 不使用對比損失)
        contrastive_temperature: 對比損失溫度參數
        **kwargs: 傳給分類損失函數的參數

    Returns:
        (cls_loss_fn, contrastive_loss_fn) tuple
    """
    # 分類損失
    cls_loss_fn = get_loss_function(loss_type=loss_type, **kwargs)

    # 對比損失（如果啟用）
    contrastive_loss_fn = None
    if contrastive_weight > 0:
        contrastive_loss_fn = MultiAspectContrastiveLoss(
            temperature=contrastive_temperature,
            contrastive_weight=contrastive_weight
        )
        print(f"  [Contrastive Learning] Enabled: weight={contrastive_weight}, temp={contrastive_temperature}")

    return cls_loss_fn, contrastive_loss_fn


if __name__ == '__main__':
    test_focal_loss()
