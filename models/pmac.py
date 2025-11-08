"""
PMAC 模組: Progressive Multi-Aspect Composition
漸進式多面向組合

功能:
    - 多粒度面向表示
    - 漸進式特徵融合
    - 動態組合多個面向資訊
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatedFusion(nn.Module):
    """
    門控融合機制
    動態控制兩個特徵向量的融合比例
    """

    def __init__(self, input_dim: int, dropout: float = 0.3):
        """
        初始化門控融合

        參數:
            input_dim: 輸入特徵維度
            dropout: Dropout 比率
        """
        super(GatedFusion, self).__init__()

        # 門控網路
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        feature1: torch.Tensor,
        feature2: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            feature1: 第一個特徵 [batch, dim]
            feature2: 第二個特徵 [batch, dim]

        返回:
            融合後的特徵 [batch, dim]
        """
        # 拼接兩個特徵 [batch, dim * 2]
        concat = torch.cat([feature1, feature2], dim=1)

        # 計算門控值 [batch, dim]
        gate_value = self.gate(concat)

        # 門控融合
        fused = gate_value * feature1 + (1 - gate_value) * feature2

        return self.dropout(fused)


class MultiGranularRepresentation(nn.Module):
    """
    多粒度表示層
    從不同粒度提取面向特徵
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_granularities: int = 3,
        dropout: float = 0.3
    ):
        """
        初始化多粒度表示

        參數:
            input_dim: 輸入維度
            output_dim: 輸出維度
            num_granularities: 粒度數量
            dropout: Dropout 比率
        """
        super(MultiGranularRepresentation, self).__init__()

        self.num_granularities = num_granularities

        # 不同粒度的變換
        self.granularity_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_granularities)
        ])

        # 融合不同粒度
        self.fusion = nn.Linear(output_dim * num_granularities, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, aspect_feature: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            aspect_feature: 面向特徵 [batch, input_dim]

        返回:
            多粒度表示 [batch, output_dim]
        """
        # 計算不同粒度的表示
        granular_features = []
        for transform in self.granularity_transforms:
            granular_features.append(transform(aspect_feature))

        # 拼接所有粒度 [batch, output_dim * num_granularities]
        concat = torch.cat(granular_features, dim=1)

        # 融合 [batch, output_dim]
        multi_granular = self.fusion(concat)

        return self.dropout(multi_granular)


class ProgressiveCompositionLayer(nn.Module):
    """
    漸進式組合層
    逐步組合多個面向的特徵
    """

    def __init__(
        self,
        feature_dim: int,
        fusion_method: str = "gated",
        dropout: float = 0.3
    ):
        """
        初始化漸進式組合層

        參數:
            feature_dim: 特徵維度
            fusion_method: 融合方法 ("gated", "concat", "weighted")
            dropout: Dropout 比率
        """
        super(ProgressiveCompositionLayer, self).__init__()

        self.fusion_method = fusion_method

        if fusion_method == "gated":
            self.fusion = GatedFusion(feature_dim, dropout)

        elif fusion_method == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        elif fusion_method == "weighted":
            self.weight_net = nn.Sequential(
                nn.Linear(feature_dim * 2, 2),
                nn.Softmax(dim=1)
            )
            self.dropout = nn.Dropout(dropout)

        else:
            raise ValueError(f"不支援的融合方法: {fusion_method}")

    def forward(
        self,
        accumulated_feature: torch.Tensor,
        new_feature: torch.Tensor
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            accumulated_feature: 累積的特徵 [batch, feature_dim]
            new_feature: 新加入的特徵 [batch, feature_dim]

        返回:
            組合後的特徵 [batch, feature_dim]
        """
        if self.fusion_method == "gated":
            return self.fusion(accumulated_feature, new_feature)

        elif self.fusion_method == "concat":
            concat = torch.cat([accumulated_feature, new_feature], dim=1)
            return self.fusion(concat)

        elif self.fusion_method == "weighted":
            concat = torch.cat([accumulated_feature, new_feature], dim=1)
            weights = self.weight_net(concat)  # [batch, 2]

            weighted = (
                weights[:, 0:1] * accumulated_feature +
                weights[:, 1:2] * new_feature
            )
            return self.dropout(weighted)


class PMAC(nn.Module):
    """
    PMAC 模組: Progressive Multi-Aspect Composition
    漸進式多面向組合

    功能:
        - 多粒度面向表示
        - 漸進式特徵融合
        - 處理多個面向的情況
    """

    def __init__(
        self,
        input_dim: int,
        fusion_dim: int,
        num_composition_layers: int = 2,
        fusion_method: str = "gated",
        dropout: float = 0.4
    ):
        """
        初始化 PMAC 模組

        參數:
            input_dim: 輸入特徵維度（來自 AAHA）
            fusion_dim: 融合後的維度
            num_composition_layers: 組合層數
            fusion_method: 融合方法
            dropout: Dropout 比率
        """
        super(PMAC, self).__init__()

        self.num_composition_layers = num_composition_layers

        # 多粒度表示
        self.multi_granular = MultiGranularRepresentation(
            input_dim=input_dim,
            output_dim=fusion_dim,
            num_granularities=3,
            dropout=dropout
        )

        # 漸進式組合層
        self.composition_layers = nn.ModuleList([
            ProgressiveCompositionLayer(
                feature_dim=fusion_dim,
                fusion_method=fusion_method,
                dropout=dropout
            )
            for _ in range(num_composition_layers)
        ])

        # 最終投影
        self.output_projection = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        aspect_features: torch.Tensor,
        num_aspects: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            aspect_features: 面向特徵 [batch, num_aspects, input_dim] 或 [batch, input_dim]
            num_aspects: 每個樣本的實際面向數 [batch] (可選)

        返回:
            組合後的表示 [batch, fusion_dim]
        """
        # 處理單個面向的情況
        if aspect_features.dim() == 2:
            # [batch, input_dim] -> [batch, 1, input_dim]
            aspect_features = aspect_features.unsqueeze(1)
            single_aspect = True
        else:
            single_aspect = False

        batch_size, num_aspects_dim, input_dim = aspect_features.size()

        # 如果是單個面向，直接處理
        if single_aspect or num_aspects_dim == 1:
            aspect_feature = aspect_features[:, 0, :]  # [batch, input_dim]

            # 多粒度表示
            multi_granular_repr = self.multi_granular(aspect_feature)

            # 最終投影
            output = self.output_projection(multi_granular_repr)

            return output

        # 多個面向的漸進式組合
        # 初始化累積特徵（使用第一個面向）
        accumulated = self.multi_granular(aspect_features[:, 0, :])

        # 漸進式組合其他面向
        for i in range(1, num_aspects_dim):
            # 獲取當前面向的多粒度表示
            current_aspect = self.multi_granular(aspect_features[:, i, :])

            # 通過組合層
            for comp_layer in self.composition_layers:
                accumulated = comp_layer(accumulated, current_aspect)

        # 最終投影
        output = self.output_projection(accumulated)

        return output

    def forward_with_attention(
        self,
        aspect_features: torch.Tensor,
        aspect_importance: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        帶注意力權重的前向傳播
        可以根據面向重要性進行加權組合

        參數:
            aspect_features: 面向特徵 [batch, num_aspects, input_dim]
            aspect_importance: 面向重要性分數 [batch, num_aspects] (可選)

        返回:
            (組合後的表示 [batch, fusion_dim], 面向權重 [batch, num_aspects])
        """
        batch_size, num_aspects_dim, input_dim = aspect_features.size()

        # 獲取所有面向的多粒度表示
        aspect_representations = []
        for i in range(num_aspects_dim):
            repr = self.multi_granular(aspect_features[:, i, :])
            aspect_representations.append(repr)

        # 堆疊 [batch, num_aspects, fusion_dim]
        stacked_reprs = torch.stack(aspect_representations, dim=1)

        # 如果沒有提供重要性分數，則計算注意力權重
        if aspect_importance is None:
            # 簡單的注意力機制
            attention_scores = torch.mean(stacked_reprs, dim=2)  # [batch, num_aspects]
            aspect_weights = F.softmax(attention_scores, dim=1)  # [batch, num_aspects]
        else:
            aspect_weights = F.softmax(aspect_importance, dim=1)

        # 加權求和 [batch, fusion_dim]
        weighted_sum = torch.sum(
            stacked_reprs * aspect_weights.unsqueeze(2),
            dim=1
        )

        # 最終投影
        output = self.output_projection(weighted_sum)

        return output, aspect_weights


if __name__ == "__main__":
    # 測試 PMAC 模組
    print("測試 PMAC 模組...")

    batch_size = 4
    num_aspects = 3
    input_dim = 256
    fusion_dim = 256

    # 創建 PMAC 模組
    pmac = PMAC(
        input_dim=input_dim,
        fusion_dim=fusion_dim,
        num_composition_layers=2,
        fusion_method="gated",
        dropout=0.4
    )

    # 測試單個面向
    print("\n1. 測試單個面向:")
    single_aspect = torch.randn(batch_size, input_dim)
    output_single = pmac(single_aspect)
    print(f"  輸入: {single_aspect.shape}")
    print(f"  輸出: {output_single.shape}")

    # 測試多個面向
    print("\n2. 測試多個面向:")
    multi_aspects = torch.randn(batch_size, num_aspects, input_dim)
    output_multi = pmac(multi_aspects)
    print(f"  輸入: {multi_aspects.shape}")
    print(f"  輸出: {output_multi.shape}")

    # 測試帶注意力的前向傳播
    print("\n3. 測試帶注意力的前向傳播:")
    output_attn, aspect_weights = pmac.forward_with_attention(multi_aspects)
    print(f"  輸出: {output_attn.shape}")
    print(f"  面向權重: {aspect_weights.shape}")
    print(f"  面向權重範例:\n{aspect_weights[0]}")

    print("\nPMAC 模組測試完成！")
