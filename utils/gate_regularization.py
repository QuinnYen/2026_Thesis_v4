"""
Gate 稀疏性正則化

用於鼓勵 Selective PMAC 的 gate 值趨向稀疏（接近 0）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_gate_sparsity_loss(
    gate_values: torch.Tensor,
    sparsity_type: str = 'l1',
    target_sparsity: float = None
) -> torch.Tensor:
    """
    計算 Gate 稀疏性損失

    參數:
        gate_values: [batch, num_aspects, num_aspects] Gate 值矩陣
        sparsity_type: 稀疏性類型
            'l1': L1 正則化（鼓勵所有 gate → 0）
            'l2': L2 正則化（較溫和）
            'hoyer': Hoyer 稀疏性（L1/L2 比率）
            'target': 目標稀疏性約束
        target_sparsity: 目標稀疏性比例（僅用於 'target' 類型）

    返回:
        sparsity_loss: 標量損失值
    """
    # 去掉對角線（自己對自己的 gate）
    batch_size, num_aspects, _ = gate_values.size()
    mask = ~torch.eye(num_aspects, dtype=torch.bool, device=gate_values.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    valid_gates = gate_values[mask]  # [batch * num_aspects * (num_aspects-1)]

    if sparsity_type == 'l1':
        # L1 正則化：Σ|gate|
        # 鼓勵 gate 值接近 0
        loss = torch.mean(torch.abs(valid_gates))

    elif sparsity_type == 'l2':
        # L2 正則化：Σ(gate²)
        # 較溫和，允許少數較大的 gate
        loss = torch.mean(valid_gates ** 2)

    elif sparsity_type == 'hoyer':
        # Hoyer 稀疏性：√n × (||x||₁ / ||x||₂)
        # 衡量分佈的稀疏程度
        l1_norm = torch.sum(torch.abs(valid_gates))
        l2_norm = torch.sqrt(torch.sum(valid_gates ** 2))
        n = valid_gates.numel()
        hoyer = (torch.sqrt(torch.tensor(n, device=valid_gates.device)) -
                (l1_norm / (l2_norm + 1e-8)))
        # 最大化 hoyer → 最小化 -hoyer
        loss = -hoyer / torch.sqrt(torch.tensor(n, device=valid_gates.device))

    elif sparsity_type == 'target':
        # 目標稀疏性約束
        # 懲罰偏離目標稀疏性的程度
        if target_sparsity is None:
            target_sparsity = 0.7  # 預設 70% 稀疏

        current_sparsity = (valid_gates < 0.1).float().mean()
        # 希望 current_sparsity → target_sparsity
        loss = (current_sparsity - target_sparsity) ** 2

    else:
        raise ValueError(f"Unknown sparsity_type: {sparsity_type}")

    return loss


def compute_gate_entropy_loss(gate_values: torch.Tensor) -> torch.Tensor:
    """
    計算 Gate 熵損失

    鼓勵 gate 值分佈更加確定（接近 0 或 1，而非中間值）

    參數:
        gate_values: [batch, num_aspects, num_aspects]

    返回:
        entropy_loss: 標量損失值
    """
    # 去掉對角線
    batch_size, num_aspects, _ = gate_values.size()
    mask = ~torch.eye(num_aspects, dtype=torch.bool, device=gate_values.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    valid_gates = gate_values[mask]

    # 計算二元熵：-[p*log(p) + (1-p)*log(1-p)]
    # 當 p→0 或 p→1 時，熵→0（確定性高）
    # 當 p→0.5 時，熵最大（不確定）
    eps = 1e-8
    entropy = -(valid_gates * torch.log(valid_gates + eps) +
                (1 - valid_gates) * torch.log(1 - valid_gates + eps))

    # 最大化熵 → 最小化 -熵（鼓勵確定性）
    # 但這裡我們希望 gate → 0，所以直接用熵
    # 實際上，熵損失可能不適合稀疏性，主要用於二元決策
    loss = torch.mean(entropy)

    return loss


def compute_gate_kl_loss(
    gate_values: torch.Tensor,
    target_distribution: str = 'sparse'
) -> torch.Tensor:
    """
    計算 Gate 值與目標分佈之間的 KL 散度

    參數:
        gate_values: [batch, num_aspects, num_aspects]
        target_distribution: 目標分佈類型
            'sparse': Beta(0.5, 5) - 偏向小值
            'uniform': Uniform(0, 1) - 均勻分佈

    返回:
        kl_loss: KL 散度損失
    """
    # 去掉對角線
    batch_size, num_aspects, _ = gate_values.size()
    mask = ~torch.eye(num_aspects, dtype=torch.bool, device=gate_values.device)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    valid_gates = gate_values[mask]

    if target_distribution == 'sparse':
        # 目標：Beta(0.5, 5) 分佈（偏向 0）
        # 使用簡化的近似：期望 gate 接近 0.1
        target_mean = 0.05
        # MSE 作為簡化的 KL 散度
        loss = F.mse_loss(valid_gates,
                         torch.full_like(valid_gates, target_mean))

    elif target_distribution == 'uniform':
        # 目標：均勻分佈 [0, 1]
        # 最小化方差（鼓勵分佈均勻）
        loss = -torch.var(valid_gates)

    else:
        raise ValueError(f"Unknown target_distribution: {target_distribution}")

    return loss


class GateRegularizedLoss(nn.Module):
    """
    帶 Gate 正則化的損失函數

    結合分類損失和 gate 稀疏性損失
    """

    def __init__(
        self,
        base_loss_fn: nn.Module,
        sparsity_weight: float = 0.01,
        sparsity_type: str = 'l1',
        target_sparsity: float = None
    ):
        """
        參數:
            base_loss_fn: 基礎損失函數（如 FocalLoss）
            sparsity_weight: 稀疏性損失的權重（推薦 0.001-0.1）
            sparsity_type: 稀疏性類型
            target_sparsity: 目標稀疏性（僅用於 'target' 類型）
        """
        super(GateRegularizedLoss, self).__init__()

        self.base_loss_fn = base_loss_fn
        self.sparsity_weight = sparsity_weight
        self.sparsity_type = sparsity_type
        self.target_sparsity = target_sparsity

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        aspect_mask: torch.Tensor,
        is_virtual: torch.Tensor = None,
        gate_values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        計算總損失

        參數:
            logits: [batch, max_aspects, num_classes]
            labels: [batch, max_aspects]
            aspect_mask: [batch, max_aspects]
            is_virtual: [batch, max_aspects]
            gate_values: [batch, max_aspects, max_aspects] (可選)

        返回:
            total_loss: 總損失
        """
        # 1. 基礎分類損失
        if hasattr(self.base_loss_fn, 'forward'):
            # 如果是自定義 loss（如 FocalLoss）
            if is_virtual is not None:
                cls_loss = self.base_loss_fn(logits, labels, aspect_mask, is_virtual)
            else:
                cls_loss = self.base_loss_fn(logits, labels, aspect_mask)
        else:
            # 標準 PyTorch loss
            cls_loss = self.base_loss_fn(logits, labels)

        # 2. Gate 稀疏性損失（如果提供）
        if gate_values is not None and self.sparsity_weight > 0:
            sparsity_loss = compute_gate_sparsity_loss(
                gate_values,
                sparsity_type=self.sparsity_type,
                target_sparsity=self.target_sparsity
            )
            total_loss = cls_loss + self.sparsity_weight * sparsity_loss
        else:
            total_loss = cls_loss
            sparsity_loss = torch.tensor(0.0, device=logits.device)

        # 儲存各項損失（用於監控）
        self.last_cls_loss = cls_loss.item()
        self.last_sparsity_loss = sparsity_loss.item() if isinstance(sparsity_loss, torch.Tensor) else 0.0
        self.last_total_loss = total_loss.item()

        return total_loss

    def get_loss_components(self):
        """返回各項損失的值（用於 logging）"""
        return {
            'cls_loss': self.last_cls_loss,
            'sparsity_loss': self.last_sparsity_loss,
            'total_loss': self.last_total_loss
        }


if __name__ == '__main__':
    # 測試代碼
    print("測試 Gate 稀疏性正則化...")

    batch_size = 4
    num_aspects = 5

    # 模擬 gate 值
    gate_values = torch.rand(batch_size, num_aspects, num_aspects) * 0.3

    # 測試不同的稀疏性損失
    for stype in ['l1', 'l2', 'hoyer', 'target']:
        loss = compute_gate_sparsity_loss(gate_values, sparsity_type=stype)
        print(f"{stype:10s} loss: {loss.item():.6f}")

    # 測試組合損失
    from utils.focal_loss import FocalLoss

    base_loss = FocalLoss(alpha=[1.0, 2.0, 1.0], gamma=2.0)
    gate_reg_loss = GateRegularizedLoss(
        base_loss_fn=base_loss,
        sparsity_weight=0.01,
        sparsity_type='l1'
    )

    # 模擬數據
    logits = torch.randn(batch_size, num_aspects, 3)
    labels = torch.randint(0, 3, (batch_size, num_aspects))
    aspect_mask = torch.ones(batch_size, num_aspects, dtype=torch.bool)
    is_virtual = torch.zeros(batch_size, num_aspects, dtype=torch.bool)

    total_loss = gate_reg_loss(logits, labels, aspect_mask, is_virtual, gate_values)

    print(f"\n總損失: {total_loss.item():.6f}")
    components = gate_reg_loss.get_loss_components()
    for k, v in components.items():
        print(f"  {k}: {v:.6f}")

    print("\n✓ Gate 正則化測試通過！")
