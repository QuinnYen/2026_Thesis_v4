"""
Selective PMAC (Progressive Multi-Aspect Composition)

核心創新：
- 可學習的 Gate 機制決定是否組合 aspects
- 稀疏的影響建模（大部分時候 gate ≈ 0）
- 殘差連接保留原始特徵
- 可解釋性：gate 值展示 aspect 影響關係
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectivePMAC(nn.Module):
    """
    Selective Progressive Multi-Aspect Composition

    與傳統 PMAC 的區別：
    1. 傳統 PMAC：強制組合所有 aspects
    2. Selective PMAC：學習何時組合、何時獨立

    優勢：
    - 自適應：模型自己學習影響關係
    - 稀疏性：不相關的 aspects 不會互相干擾
    - 保留獨立性：Aspect-level 分類的本質需求
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        use_layer_norm: bool = True,
        gate_activation: str = 'sigmoid'  # 'sigmoid', 'tanh', 'relu'
    ):
        super(SelectivePMAC, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_layer_norm = use_layer_norm

        # Gate 網絡：學習是否需要組合
        self.relation_gate = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

        # 根據 activation 選擇最後的激活函數
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'tanh':
            self.gate_activation = nn.Tanh()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown gate_activation: {gate_activation}")

        # Composition 網絡：如何組合兩個 aspects
        self.composition = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )

        # Layer Normalization（可選）
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(input_dim)

        # 初始化：讓 gate 初始時偏向小值（稀疏性）
        with torch.no_grad():
            for module in self.relation_gate.modules():
                if isinstance(module, nn.Linear):
                    # 小的初始化讓 gate 初始接近 0
                    nn.init.xavier_uniform_(module.weight, gain=0.1)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, aspects, aspect_mask=None):
        """
        Args:
            aspects: [batch, num_aspects, input_dim]
            aspect_mask: [batch, num_aspects], True 表示有效 aspect

        Returns:
            composed_aspects: [batch, num_aspects, input_dim]
            gate_values: [batch, num_aspects, num_aspects] - 用於分析
        """
        batch_size, num_aspects, _ = aspects.size()

        # 存儲所有 gate 值（用於分析和可解釋性）
        all_gates = torch.zeros(batch_size, num_aspects, num_aspects, device=aspects.device)

        composed_aspects = []

        for i in range(num_aspects):
            # 當前 aspect
            current = aspects[:, i]  # [batch, input_dim]

            # 計算受其他 aspects 的影響
            influences = []
            gates = []

            for j in range(num_aspects):
                if i == j:
                    continue

                other = aspects[:, j]  # [batch, input_dim]

                # 計算是否需要組合（Gate）
                gate_input = torch.cat([current, other], dim=-1)  # [batch, input_dim*2]
                gate_logit = self.relation_gate(gate_input)  # [batch, 1]
                gate = self.gate_activation(gate_logit)  # [batch, 1], 0-1 之間

                # 存儲 gate 值
                all_gates[:, i, j] = gate.squeeze(-1)

                # 組合表示
                comp_input = torch.cat([current, other], dim=-1)
                composed = self.composition(comp_input)  # [batch, input_dim]

                # 加權影響
                weighted_influence = gate * composed  # [batch, input_dim]

                influences.append(weighted_influence)
                gates.append(gate)

            # 當前 aspect + 所有其他 aspects 的加權影響
            if len(influences) > 0:
                total_influence = torch.stack(influences).sum(dim=0)  # [batch, input_dim]

                # 殘差連接：保留原始特徵
                final = current + total_influence
            else:
                final = current

            # Layer Normalization
            if self.use_layer_norm:
                final = self.layer_norm(final)

            composed_aspects.append(final)

        # 堆疊回 [batch, num_aspects, input_dim]
        composed_aspects = torch.stack(composed_aspects, dim=1)

        # 應用 aspect_mask（如果提供）
        if aspect_mask is not None:
            mask_expanded = aspect_mask.unsqueeze(-1)  # [batch, num_aspects, 1]
            composed_aspects = composed_aspects * mask_expanded

        return composed_aspects, all_gates

    def get_gate_statistics(self, gate_values):
        """
        分析 gate 值的統計信息

        Args:
            gate_values: [batch, num_aspects, num_aspects]

        Returns:
            dict: 統計信息
        """
        # 去掉對角線（自己對自己）
        batch_size, num_aspects, _ = gate_values.size()
        mask = ~torch.eye(num_aspects, dtype=torch.bool, device=gate_values.device)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

        valid_gates = gate_values[mask]  # [batch * num_aspects * (num_aspects-1)]

        return {
            'mean': valid_gates.mean().item(),
            'std': valid_gates.std().item(),
            'min': valid_gates.min().item(),
            'max': valid_gates.max().item(),
            'sparsity': (valid_gates < 0.1).float().mean().item(),  # 多少比例 < 0.1
            'active': (valid_gates > 0.5).float().mean().item()  # 多少比例 > 0.5
        }


class SelectivePMACMultiAspect(nn.Module):
    """
    Multi-Aspect 版本的 Selective PMAC
    適配原有的 HMACNetMultiAspect 架構
    """

    def __init__(
        self,
        input_dim: int = 768,
        fusion_dim: int = 512,
        num_composition_layers: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        use_layer_norm: bool = True
    ):
        super(SelectivePMACMultiAspect, self).__init__()

        self.input_dim = input_dim
        self.num_composition_layers = num_composition_layers

        # 多層 Selective PMAC
        self.composition_layers = nn.ModuleList([
            SelectivePMAC(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_layer_norm=use_layer_norm
            )
            for _ in range(num_composition_layers)
        ])

        # 最終的 fusion layer（可選）
        self.final_fusion = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, input_dim)
        )

    def forward(self, aspect_contexts, aspect_mask=None):
        """
        Args:
            aspect_contexts: [batch, max_aspects, input_dim] - AAHA 的輸出
            aspect_mask: [batch, max_aspects]

        Returns:
            composed: [batch, max_aspects, input_dim]
            all_gate_stats: list of dicts - 每層的 gate 統計
        """
        composed = aspect_contexts
        all_gate_stats = []

        # 逐層組合
        all_gate_values = []  # 收集所有gate值用於返回
        for layer_idx, comp_layer in enumerate(self.composition_layers):
            composed, gate_values = comp_layer(composed, aspect_mask)

            # 收集所有gate值
            all_gate_values.append(gate_values)

            # 收集 gate 統計（用於分析）
            if self.training:
                gate_stats = comp_layer.get_gate_statistics(gate_values)
                gate_stats['layer'] = layer_idx
                all_gate_stats.append(gate_stats)

        # 最終 fusion
        composed = self.final_fusion(composed)

        # 應用 mask
        if aspect_mask is not None:
            mask_expanded = aspect_mask.unsqueeze(-1)
            composed = composed * mask_expanded

        # 返回gate值（所有層的平均）用於分析
        # 如果不是training，返回gate值而不是統計
        if not self.training and len(all_gate_values) > 0:
            # 取所有層的平均gate值
            avg_gates = torch.stack(all_gate_values).mean(dim=0)
            return composed, avg_gates
        else:
            return composed, all_gate_stats


# 為了向後兼容，提供一個簡單的接口
def create_selective_pmac(
    input_dim: int = 768,
    num_layers: int = 2,
    dropout: float = 0.3,
    **kwargs
):
    """
    創建 Selective PMAC 模組的便捷函數
    """
    return SelectivePMACMultiAspect(
        input_dim=input_dim,
        fusion_dim=512,
        num_composition_layers=num_layers,
        hidden_dim=256,
        dropout=dropout,
        use_layer_norm=True
    )


if __name__ == '__main__':
    # 測試代碼
    print("測試 Selective PMAC...")

    batch_size = 4
    num_aspects = 5
    input_dim = 768

    # 創建測試數據
    aspects = torch.randn(batch_size, num_aspects, input_dim)
    aspect_mask = torch.ones(batch_size, num_aspects, dtype=torch.bool)
    aspect_mask[:, 3:] = False  # 前 3 個 aspects 有效

    # 創建模型
    model = SelectivePMACMultiAspect(
        input_dim=input_dim,
        num_composition_layers=2,
        dropout=0.3
    )

    # Forward
    model.train()
    composed, gate_stats = model(aspects, aspect_mask)

    print(f"Input shape: {aspects.shape}")
    print(f"Output shape: {composed.shape}")
    print(f"\nGate Statistics:")
    for stats in gate_stats:
        print(f"  Layer {stats['layer']}:")
        print(f"    Mean: {stats['mean']:.4f}")
        print(f"    Sparsity (<0.1): {stats['sparsity']:.2%}")
        print(f"    Active (>0.5): {stats['active']:.2%}")

    print("\n✓ Selective PMAC 測試通過！")
