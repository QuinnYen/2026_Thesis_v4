"""
增強版 PMAC 模組: Progressive Multi-Aspect Composition (Enhanced)
漸進式多面向組合（增強版）

新增功能:
    1. Enhanced Gating Mechanism - 多層門控 + 自注意力
    2. Aspect-specific Batch Normalization - 針對不同 aspect 的 BN
    3. Progressive Training Support - 支持漸進式訓練策略
    4. Residual Connections - 防止資訊丟失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class EnhancedGatingMechanism(nn.Module):
    """
    增強版門控機制

    改進:
        1. 多層門控網絡（而非單層）
        2. 加入 aspect-awareness（考慮 aspect 資訊）
        3. 使用自注意力動態調整門控強度
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_self_attention: bool = True
    ):
        """
        初始化增強版門控機制

        參數:
            input_dim: 輸入特徵維度
            hidden_dim: 隱藏層維度
            dropout: Dropout 比率
            use_self_attention: 是否使用自注意力
        """
        super(EnhancedGatingMechanism, self).__init__()

        self.use_self_attention = use_self_attention

        # 多層門控網絡
        self.gate_network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, input_dim),
            nn.Sigmoid()  # 門控值在 [0, 1]
        )

        # 自注意力（可選）
        if use_self_attention:
            self.query_proj = nn.Linear(input_dim, input_dim)
            self.key_proj = nn.Linear(input_dim, input_dim)
            self.value_proj = nn.Linear(input_dim, input_dim)
            self.scale = input_dim ** 0.5

        # Layer Normalization
        self.ln = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            feature1: 累積特徵 [batch, input_dim]
            feature2: 新特徵 [batch, input_dim]

        返回:
            融合後的特徵 [batch, input_dim]
        """
        batch_size = feature1.size(0)

        # 1. 計算門控值
        concat = torch.cat([feature1, feature2], dim=1)
        gate = self.gate_network(concat)  # [batch, input_dim]

        # 2. 門控融合
        gated_fusion = gate * feature1 + (1 - gate) * feature2

        # 3. 自注意力增強（可選）
        if self.use_self_attention:
            # 將兩個特徵堆疊起來
            stacked = torch.stack([feature1, feature2], dim=1)  # [batch, 2, input_dim]

            # 計算注意力
            Q = self.query_proj(gated_fusion).unsqueeze(1)  # [batch, 1, input_dim]
            K = self.key_proj(stacked)  # [batch, 2, input_dim]
            V = self.value_proj(stacked)  # [batch, 2, input_dim]

            # 注意力分數
            attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [batch, 1, 2]
            attn_weights = F.softmax(attn_scores, dim=-1)

            # 注意力輸出
            attn_output = torch.matmul(attn_weights, V).squeeze(1)  # [batch, input_dim]

            # 殘差連接
            output = self.ln(gated_fusion + attn_output)
        else:
            output = self.ln(gated_fusion)

        return self.dropout(output)


class AspectSpecificBatchNorm(nn.Module):
    """
    Aspect-specific Batch Normalization

    為不同的 aspect 維護不同的 BN 統計量
    這樣可以更好地處理不同 aspect 的分佈差異
    """

    def __init__(
        self,
        num_features: int,
        num_aspects: int = 3,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True
    ):
        """
        初始化 Aspect-specific Batch Normalization

        參數:
            num_features: 特徵維度
            num_aspects: aspect 數量（最大）
            eps: 數值穩定性參數
            momentum: 移動平均動量
            affine: 是否學習 scale 和 shift 參數
        """
        super(AspectSpecificBatchNorm, self).__init__()

        self.num_features = num_features
        self.num_aspects = num_aspects
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # 為每個 aspect 創建獨立的 BN
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(num_features, eps=eps, momentum=momentum, affine=affine)
            for _ in range(num_aspects)
        ])

    def forward(
        self,
        x: torch.Tensor,
        aspect_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入特徵 [batch, num_features]
            aspect_ids: aspect 索引 [batch] (可選，預設為 0)

        返回:
            歸一化後的特徵 [batch, num_features]
        """
        if aspect_ids is None:
            # 如果沒有提供 aspect_ids，使用第一個 BN
            return self.bn_layers[0](x)

        # 為不同 aspect 使用不同的 BN
        batch_size = x.size(0)
        output = torch.zeros_like(x)

        for aspect_id in range(self.num_aspects):
            # 找到屬於這個 aspect 的樣本
            mask = (aspect_ids == aspect_id)
            num_samples = mask.sum()

            if num_samples > 1:
                # 有足夠樣本，使用 BN
                output[mask] = self.bn_layers[aspect_id](x[mask])
            elif num_samples == 1:
                # 只有一個樣本，直接通過（不做歸一化）
                output[mask] = x[mask]

        return output


class ResidualGatedFusion(nn.Module):
    """
    帶殘差連接的門控融合
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        use_aspect_bn: bool = True,
        num_aspects: int = 3
    ):
        super(ResidualGatedFusion, self).__init__()

        # 增強版門控
        self.gating = EnhancedGatingMechanism(
            input_dim, hidden_dim, dropout, use_self_attention=True
        )

        # Aspect-specific BN
        self.use_aspect_bn = use_aspect_bn
        if use_aspect_bn:
            self.aspect_bn = AspectSpecificBatchNorm(input_dim, num_aspects)
        else:
            self.bn = nn.BatchNorm1d(input_dim)

        # Feed-Forward Network (增加表徵能力)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(dropout)
        )

        # Layer Normalization
        self.ln1 = nn.LayerNorm(input_dim)
        self.ln2 = nn.LayerNorm(input_dim)

    def forward(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor,
        aspect_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播（帶殘差連接）

        參數:
            feature1: 累積特徵 [batch, input_dim]
            feature2: 新特徵 [batch, input_dim]
            aspect_ids: aspect 索引 [batch] (可選)

        返回:
            融合後的特徵 [batch, input_dim]
        """
        # 1. 門控融合
        gated = self.gating(feature1, feature2)

        # 2. 殘差連接 + Layer Norm
        residual1 = self.ln1(gated + feature1)

        # 3. Aspect-specific BN
        if self.use_aspect_bn and aspect_ids is not None:
            normalized = self.aspect_bn(residual1, aspect_ids)
        else:
            if self.use_aspect_bn:
                normalized = self.aspect_bn(residual1, None)
            else:
                normalized = self.bn(residual1)

        # 4. Feed-Forward Network + 殘差連接
        ffn_output = self.ffn(normalized)
        output = self.ln2(ffn_output + normalized)

        return output


class PMACEnhanced(nn.Module):
    """
    增強版 PMAC 模組

    改進:
        1. ✅ Enhanced Gating Mechanism - 多層門控 + 自注意力
        2. ✅ Aspect-specific Batch Normalization
        3. ✅ Residual Connections
        4. ✅ Progressive Training Support
    """

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int,
        num_composition_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.4,
        use_aspect_bn: bool = True,
        num_aspects: int = 3,
        progressive_training: bool = False
    ):
        """
        初始化增強版 PMAC

        參數:
            input_dim: 輸入維度（來自 AAHA）
            fusion_dim: 融合後的維度
            num_composition_layers: 組合層數
            hidden_dim: 門控網絡隱藏層維度
            dropout: Dropout 比率
            use_aspect_bn: 是否使用 aspect-specific BN
            num_aspects: aspect 數量
            progressive_training: 是否支持漸進式訓練
        """
        super(PMACEnhanced, self).__init__()

        self.num_composition_layers = num_composition_layers
        self.progressive_training = progressive_training

        # 輸入投影（將 input_dim 投影到 fusion_dim）
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 多粒度表示（保留原有設計）
        self.multi_granular_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(3)  # 3 個粒度
        ])

        # 粒度融合
        self.granularity_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 漸進式組合層（增強版）
        self.composition_layers = nn.ModuleList([
            ResidualGatedFusion(
                input_dim=fusion_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_aspect_bn=use_aspect_bn,
                num_aspects=num_aspects
            )
            for _ in range(num_composition_layers)
        ])

        # 最終輸出投影
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def multi_granular_representation(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        計算多粒度表示

        參數:
            x: 輸入特徵 [batch, fusion_dim]

        返回:
            多粒度表示 [batch, fusion_dim]
        """
        # 不同粒度的轉換
        granular_features = []
        for transform in self.multi_granular_transforms:
            granular_features.append(transform(x))

        # 拼接
        concat = torch.cat(granular_features, dim=1)

        # 融合
        multi_granular = self.granularity_fusion(concat)

        # 殘差連接
        return multi_granular + x

    def forward(
        self,
        aspect_features: torch.Tensor,
        aspect_ids: Optional[torch.Tensor] = None,
        num_aspects: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            aspect_features: [batch, input_dim] 或 [batch, num_aspects, input_dim]
            aspect_ids: aspect 索引 [batch] (用於 aspect-specific BN)
            num_aspects: 每個樣本的實際 aspect 數 [batch] (可選)

        返回:
            組合後的表示 [batch, fusion_dim]
        """
        # 處理輸入維度
        if aspect_features.dim() == 2:
            # 單個 aspect: [batch, input_dim]
            aspect_features = aspect_features.unsqueeze(1)  # [batch, 1, input_dim]
            single_aspect = True
        else:
            single_aspect = False

        batch_size, num_aspects_dim, input_dim = aspect_features.size()

        # 投影到 fusion_dim
        # [batch, num_aspects, input_dim] -> [batch, num_aspects, fusion_dim]
        projected = torch.stack([
            self.input_projection(aspect_features[:, i, :])
            for i in range(num_aspects_dim)
        ], dim=1)

        # 單個 aspect 的情況
        if single_aspect or num_aspects_dim == 1:
            aspect_repr = projected[:, 0, :]  # [batch, fusion_dim]

            # 多粒度表示
            multi_granular = self.multi_granular_representation(aspect_repr)

            # 輸出投影
            output = self.output_projection(multi_granular)

            return output

        # 多個 aspect 的漸進式組合
        # 1. 初始化：使用第一個 aspect
        accumulated = self.multi_granular_representation(projected[:, 0, :])

        # 2. 漸進式組合其他 aspects
        for i in range(1, num_aspects_dim):
            # 當前 aspect 的多粒度表示
            current_aspect = self.multi_granular_representation(projected[:, i, :])

            # 通過所有組合層
            for comp_layer in self.composition_layers:
                accumulated = comp_layer(
                    accumulated,
                    current_aspect,
                    aspect_ids=aspect_ids if aspect_ids is not None else None
                )

        # 3. 最終投影
        output = self.output_projection(accumulated)

        return output

    def forward_progressive(
        self,
        aspect_features: torch.Tensor,
        max_num_aspects: int = 1,
        aspect_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        漸進式訓練的前向傳播

        在訓練初期只使用單個 aspect，逐步增加 aspect 數量

        參數:
            aspect_features: [batch, num_aspects, input_dim]
            max_num_aspects: 當前階段使用的最大 aspect 數
            aspect_ids: aspect 索引 [batch]

        返回:
            組合後的表示 [batch, fusion_dim]
        """
        if aspect_features.dim() == 2:
            return self.forward(aspect_features, aspect_ids)

        batch_size, total_aspects, input_dim = aspect_features.size()

        # 限制使用的 aspect 數量（漸進式訓練）
        num_aspects_to_use = min(max_num_aspects, total_aspects)

        # 只使用前 N 個 aspects
        limited_features = aspect_features[:, :num_aspects_to_use, :]

        return self.forward(limited_features, aspect_ids)


if __name__ == "__main__":
    print("測試增強版 PMAC 模組...")

    batch_size = 4
    num_aspects = 3
    input_dim = 256
    fusion_dim = 256

    # 創建增強版 PMAC
    pmac_enhanced = PMACEnhanced(
        input_dim=input_dim,
        fusion_dim=fusion_dim,
        num_composition_layers=2,
        hidden_dim=128,
        dropout=0.4,
        use_aspect_bn=True,
        num_aspects=3,
        progressive_training=True
    )

    # 測試單個 aspect
    print("\n1. 測試單個 aspect:")
    single_aspect = torch.randn(batch_size, input_dim)
    output_single = pmac_enhanced(single_aspect)
    print(f"  輸入: {single_aspect.shape}")
    print(f"  輸出: {output_single.shape}")

    # 測試多個 aspects
    print("\n2. 測試多個 aspects:")
    multi_aspects = torch.randn(batch_size, num_aspects, input_dim)
    aspect_ids = torch.randint(0, 3, (batch_size,))
    output_multi = pmac_enhanced(multi_aspects, aspect_ids)
    print(f"  輸入: {multi_aspects.shape}")
    print(f"  Aspect IDs: {aspect_ids}")
    print(f"  輸出: {output_multi.shape}")

    # 測試漸進式訓練
    print("\n3. 測試漸進式訓練:")
    for stage in [1, 2, 3]:
        output_prog = pmac_enhanced.forward_progressive(
            multi_aspects,
            max_num_aspects=stage,
            aspect_ids=aspect_ids
        )
        print(f"  Stage {stage} (使用 {stage} 個 aspects): {output_prog.shape}")

    # 計算參數量
    total_params = sum(p.numel() for p in pmac_enhanced.parameters())
    print(f"\n總參數量: {total_params:,}")

    print("\n✅ 增強版 PMAC 測試完成!")
    print("\n改進總結:")
    print("  ✅ 1. Enhanced Gating Mechanism - 多層門控 + 自注意力")
    print("  ✅ 2. Aspect-specific Batch Normalization")
    print("  ✅ 3. Residual Connections - 防止資訊丟失")
    print("  ✅ 4. Progressive Training Support")


# ============================================================================
# Multi-Aspect PMAC (真正的多面向組合)
# ============================================================================

class PMACMultiAspect(nn.Module):
    """
    Multi-Aspect Progressive Composition

    真正的多面向漸進式組合模組
    設計用於句子級別的多 aspect 建模
    """

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int = None,
        num_composition_layers: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        composition_mode: str = 'sequential'  # 'sequential', 'pairwise', 'attention'
    ):
        """
        初始化 Multi-Aspect PMAC

        參數:
            input_dim: 輸入維度（來自 AAHA 的 context vectors）
            fusion_dim: 融合維度（如果為 None，則使用 input_dim）
            num_composition_layers: 組合層數
            hidden_dim: 門控網絡隱藏維度
            dropout: Dropout 比率
            composition_mode: 組合模式
                - 'sequential': 順序組合 (asp1 + asp2 -> c12, c12 + asp3 -> c123, ...)
                - 'pairwise': 兩兩組合（每個 aspect 與其他所有 aspects 組合）
                - 'attention': 基於注意力的組合
        """
        super(PMACMultiAspect, self).__init__()

        self.input_dim = input_dim
        self.fusion_dim = fusion_dim if fusion_dim is not None else input_dim
        self.num_composition_layers = num_composition_layers
        self.composition_mode = composition_mode

        # 投影層（如果需要維度轉換）
        if self.input_dim != self.fusion_dim:
            self.input_projection = nn.Linear(input_dim, self.fusion_dim)
        else:
            self.input_projection = nn.Identity()

        # 門控融合層（用於組合兩個 aspects）
        self.gated_fusion_layers = nn.ModuleList([
            EnhancedGatingMechanism(
                input_dim=self.fusion_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_self_attention=True
            )
            for _ in range(num_composition_layers)
        ])

        # Attention-based 組合（如果使用 attention 模式）
        if composition_mode == 'attention':
            self.attention_query = nn.Linear(self.fusion_dim, self.fusion_dim)
            self.attention_key = nn.Linear(self.fusion_dim, self.fusion_dim)
            self.attention_value = nn.Linear(self.fusion_dim, self.fusion_dim)
            self.scale = self.fusion_dim ** 0.5

        # Layer Normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.fusion_dim)
            for _ in range(num_composition_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        aspect_features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Multi-Aspect 組合

        參數:
            aspect_features: [batch, num_aspects, dim]
            aspect_mask: [batch, num_aspects] - bool mask，True 表示有效 aspect

        返回:
            composed_features: [batch, num_aspects, fusion_dim]
        """
        batch_size, num_aspects, _ = aspect_features.shape

        # 投影
        features = self.input_projection(aspect_features)  # [batch, N, fusion_dim]

        if self.composition_mode == 'sequential':
            return self._sequential_composition(features, aspect_mask)
        elif self.composition_mode == 'pairwise':
            return self._pairwise_composition(features, aspect_mask)
        elif self.composition_mode == 'attention':
            return self._attention_composition(features, aspect_mask)
        else:
            raise ValueError(f"Unknown composition_mode: {self.composition_mode}")

    def _sequential_composition(
        self,
        features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        順序組合: asp1 + asp2 -> c12, c12 + asp3 -> c123, ...

        這是真正的「漸進式組合」- PMAC 的核心創新點！
        """
        batch_size, num_aspects, dim = features.shape

        # 初始化輸出
        composed_features = torch.zeros_like(features)

        # 對每個樣本單獨處理（因為 aspect 數量不同）
        for b in range(batch_size):
            # 找到有效 aspects
            valid_mask = aspect_mask[b]
            valid_indices = torch.where(valid_mask)[0]
            num_valid = len(valid_indices)

            if num_valid == 0:
                continue
            elif num_valid == 1:
                # 只有一個 aspect，直接複製
                composed_features[b, valid_indices[0]] = features[b, valid_indices[0]]
            else:
                # 多個 aspects，漸進式組合
                # 從第一個開始
                composed = features[b, valid_indices[0]].clone().unsqueeze(0)  # [1, dim]

                # 逐個組合後續 aspects
                for layer_idx, gating_layer in enumerate(self.gated_fusion_layers):
                    for i in range(1, num_valid):
                        curr_aspect = features[b, valid_indices[i]].unsqueeze(0)  # [1, dim]

                        # 門控融合
                        composed = gating_layer(composed, curr_aspect)

                        # Layer norm
                        composed = self.layer_norms[layer_idx](composed)

                # 轉回 1D
                composed = composed.squeeze(0)  # [dim]

                # 將組合結果廣播到所有有效 aspects
                # 每個 aspect 都獲得完整的組合信息
                for idx in valid_indices:
                    composed_features[b, idx] = composed

        return composed_features

    def _pairwise_composition(
        self,
        features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        兩兩組合: 每個 aspect 與其他所有 aspects 組合

        這種方式可以捕捉更豐富的 aspect 交互
        """
        batch_size, num_aspects, dim = features.shape
        composed_features = torch.zeros_like(features)

        for b in range(batch_size):
            valid_mask = aspect_mask[b]
            valid_indices = torch.where(valid_mask)[0]
            num_valid = len(valid_indices)

            if num_valid <= 1:
                # 單個或無 aspect
                if num_valid == 1:
                    composed_features[b, valid_indices[0]] = features[b, valid_indices[0]]
                continue

            # 對每個 aspect，與其他所有 aspects 組合
            for i, idx_i in enumerate(valid_indices):
                composed = features[b, idx_i].clone()

                # 與其他 aspects 組合
                for j, idx_j in enumerate(valid_indices):
                    if i == j:
                        continue

                    other_aspect = features[b, idx_j]

                    # 使用第一個門控層組合
                    composed = self.gated_fusion_layers[0](composed, other_aspect)

                composed_features[b, idx_i] = self.layer_norms[0](composed)

        return composed_features

    def _attention_composition(
        self,
        features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        基於注意力的組合: 使用 self-attention 聚合所有 aspects

        每個 aspect 通過注意力機制查詢其他 aspects
        """
        batch_size, num_aspects, dim = features.shape

        # Multi-head self-attention
        Q = self.attention_query(features)  # [batch, N, dim]
        K = self.attention_key(features)
        V = self.attention_value(features)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, N, N]

        # 應用 mask
        mask_expanded = aspect_mask.unsqueeze(1)  # [batch, 1, N]
        scores = scores.masked_fill(~mask_expanded, -1e9)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)  # [batch, N, N]

        # 應用 attention
        composed = torch.matmul(attn_weights, V)  # [batch, N, dim]

        # 殘差連接
        composed = features + self.dropout(composed)
        composed = self.layer_norms[0](composed)

        return composed
