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


def get_loss_function(loss_type='ce', **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type: 'ce', 'focal', 'adaptive'
        **kwargs: Additional arguments for loss function
            - alpha: Class weights [alpha_neg, alpha_neu, alpha_pos]
            - gamma: Focal loss gamma parameter
            - virtual_weight: Weight for virtual aspects

    Returns:
        loss_fn: Loss function
    """
    if loss_type == 'ce':
        # Standard Cross Entropy (wrapped for compatibility)
        return MultiAspectCrossEntropyLoss(
            virtual_weight=kwargs.get('virtual_weight', 0.5)
        )

    elif loss_type == 'focal':
        return MultiAspectFocalLoss(
            alpha=kwargs.get('alpha', None),
            gamma=kwargs.get('gamma', 2.0),
            virtual_weight=kwargs.get('virtual_weight', 0.5)
        )

    elif loss_type == 'adaptive':
        return AdaptiveWeightedLoss(
            base_alpha=kwargs.get('alpha', [1.0, 1.0, 1.0]),
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


if __name__ == '__main__':
    test_focal_loss()
