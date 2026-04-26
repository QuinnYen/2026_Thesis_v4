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

    # 截斷最大權重，防止極端類別分布（如 REST16 Positive 73%）造成梯度失衡
    # 保留原始 inverse frequency 比例，不做 normalize（normalize 會抵消截斷效果）
    # max_weight=3.0：以驗證集 F1 選定，允許少數類別最多獲得多數類別 6 倍的梯度貢獻
    if smooth:
        weights = [max(0.5, min(3.0, w)) for w in weights]

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


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss for Multi-Aspect ABSA

    結合 hard label loss 和 soft label loss（從 Claude 學習）

    公式：
        L = (1-α) * L_hard + α * L_soft
        L_hard = CrossEntropy(pred, hard_label)
        L_soft = KL(pred_soft, teacher_soft) * T^2

    Args:
        alpha: soft label 的權重（0.0-1.0），默認 0.3
        temperature: 軟化 temperature，較高值產生更平滑的分布
        virtual_weight: 虛擬 aspect 的權重
    """

    def __init__(self, alpha: float = 0.3, temperature: float = 2.0, virtual_weight: float = 0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.virtual_weight = virtual_weight

    def forward(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor,
        aspect_mask: torch.Tensor,
        is_virtual: torch.Tensor = None
    ) -> torch.Tensor:
        """
        計算知識蒸餾損失

        Args:
            logits: [batch, max_aspects, num_classes] - 模型輸出
            hard_labels: [batch, max_aspects] - ground truth 標籤
            soft_labels: [batch, max_aspects, num_classes] - Claude soft labels
            aspect_mask: [batch, max_aspects] - 有效 aspect mask
            is_virtual: [batch, max_aspects] - 虛擬 aspect 標記

        Returns:
            loss: scalar
        """
        batch_size, max_aspects, num_classes = logits.shape
        device = logits.device

        # Flatten
        logits_flat = logits.view(-1, num_classes)
        hard_labels_flat = hard_labels.view(-1)
        soft_labels_flat = soft_labels.view(-1, num_classes)
        aspect_mask_flat = aspect_mask.view(-1)

        # Valid mask (非 padding 且有效)
        valid_mask = (hard_labels_flat != -100) & aspect_mask_flat

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # 取出有效樣本
        valid_logits = logits_flat[valid_mask]
        valid_hard_labels = hard_labels_flat[valid_mask]
        valid_soft_labels = soft_labels_flat[valid_mask]

        # ============= Hard Label Loss =============
        hard_loss = F.cross_entropy(valid_logits, valid_hard_labels, reduction='none')

        # ============= Soft Label Loss (KL Divergence) =============
        # 軟化 logits
        soft_logits = valid_logits / self.temperature
        soft_targets = valid_soft_labels

        # 計算 KL divergence: KL(teacher || student)
        # 使用 log_softmax 和 softmax 計算
        log_probs = F.log_softmax(soft_logits, dim=-1)

        # KL(P||Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
        # 這裡 P = soft_targets (teacher), Q = model output
        soft_loss = F.kl_div(
            log_probs,
            soft_targets,
            reduction='none'
        ).sum(dim=-1)  # [num_valid]

        # 乘以 T^2 來平衡梯度
        soft_loss = soft_loss * (self.temperature ** 2)

        # ============= 組合損失 =============
        combined_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        # ============= Virtual Aspect Weighting =============
        if is_virtual is not None:
            is_virtual_flat = is_virtual.view(-1)[valid_mask]
            weights = torch.ones_like(combined_loss)
            weights[is_virtual_flat] = self.virtual_weight
            combined_loss = combined_loss * weights

        return combined_loss.mean()


class MultiAspectDistillationLoss(nn.Module):
    """
    Multi-Aspect 知識蒸餾損失（結合 Focal Loss）

    支持：
    - Focal Loss 作為 hard label loss
    - KL divergence 作為 soft label loss
    - 可選的 class weighting

    Args:
        alpha: soft label 權重
        temperature: 蒸餾溫度
        focal_gamma: Focal Loss 的 gamma 參數
        class_weights: 類別權重
        virtual_weight: 虛擬 aspect 權重
    """

    def __init__(
        self,
        alpha: float = 0.3,
        temperature: float = 2.0,
        focal_gamma: float = 2.0,
        class_weights: list = None,
        virtual_weight: float = 0.5
    ):
        super(MultiAspectDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.focal_gamma = focal_gamma
        self.virtual_weight = virtual_weight

        # Class weights for focal loss
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights)
        else:
            self.class_weights = None

    def forward(
        self,
        logits: torch.Tensor,
        hard_labels: torch.Tensor,
        soft_labels: torch.Tensor,
        aspect_mask: torch.Tensor,
        is_virtual: torch.Tensor = None
    ) -> torch.Tensor:
        """
        計算組合損失

        Args:
            logits: [batch, max_aspects, num_classes]
            hard_labels: [batch, max_aspects]
            soft_labels: [batch, max_aspects, num_classes]
            aspect_mask: [batch, max_aspects]
            is_virtual: [batch, max_aspects]

        Returns:
            loss: scalar
        """
        batch_size, max_aspects, num_classes = logits.shape
        device = logits.device

        # Flatten
        logits_flat = logits.view(-1, num_classes)
        hard_labels_flat = hard_labels.view(-1)
        soft_labels_flat = soft_labels.view(-1, num_classes)
        aspect_mask_flat = aspect_mask.view(-1)

        # Valid mask
        valid_mask = (hard_labels_flat != -100) & aspect_mask_flat

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        valid_logits = logits_flat[valid_mask]
        valid_hard_labels = hard_labels_flat[valid_mask]
        valid_soft_labels = soft_labels_flat[valid_mask]

        # ============= Focal Loss (Hard Label) =============
        probs = F.softmax(valid_logits, dim=-1)
        labels_one_hot = F.one_hot(valid_hard_labels, num_classes).float()
        p_t = (probs * labels_one_hot).sum(dim=-1)

        focal_weight = (1 - p_t) ** self.focal_gamma
        ce_loss = -torch.log(p_t + 1e-8)
        hard_loss = focal_weight * ce_loss

        # Apply class weights
        if self.class_weights is not None:
            if self.class_weights.device != device:
                self.class_weights = self.class_weights.to(device)
            alpha_t = self.class_weights[valid_hard_labels]
            hard_loss = alpha_t * hard_loss

        # ============= Soft Label Loss (KL Divergence) =============
        soft_logits = valid_logits / self.temperature
        log_probs = F.log_softmax(soft_logits, dim=-1)

        soft_loss = F.kl_div(
            log_probs,
            valid_soft_labels,
            reduction='none'
        ).sum(dim=-1) * (self.temperature ** 2)

        # ============= 組合 =============
        combined_loss = (1 - self.alpha) * hard_loss + self.alpha * soft_loss

        # Virtual weighting
        if is_virtual is not None:
            is_virtual_flat = is_virtual.view(-1)[valid_mask]
            weights = torch.ones_like(combined_loss)
            weights[is_virtual_flat] = self.virtual_weight
            combined_loss = combined_loss * weights

        return combined_loss.mean()


def get_distillation_loss_function(
    alpha: float = 0.3,
    temperature: float = 2.0,
    use_focal: bool = True,
    focal_gamma: float = 2.0,
    class_weights: list = None,
    virtual_weight: float = 0.5
):
    """
    Factory function 創建知識蒸餾損失函數

    Args:
        alpha: soft label 權重 (0.0 = 純 hard label, 1.0 = 純 soft label)
        temperature: 蒸餾溫度
        use_focal: 是否使用 Focal Loss 作為 hard label loss
        focal_gamma: Focal Loss gamma
        class_weights: 類別權重
        virtual_weight: 虛擬 aspect 權重

    Returns:
        loss function
    """
    print(f"  [Knowledge Distillation] alpha={alpha}, T={temperature}, focal={use_focal}")

    if use_focal:
        return MultiAspectDistillationLoss(
            alpha=alpha,
            temperature=temperature,
            focal_gamma=focal_gamma,
            class_weights=class_weights,
            virtual_weight=virtual_weight
        )
    else:
        return KnowledgeDistillationLoss(
            alpha=alpha,
            temperature=temperature,
            virtual_weight=virtual_weight
        )


if __name__ == '__main__':
    test_focal_loss()
