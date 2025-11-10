"""
IARM Enhanced 模組: Enhanced Inter-Aspect Relation Modeling
增強版面向間關係建模

增強功能:
    - Relation-aware pooling 機制
    - Contrastive loss 增強 aspect 區分度
    - 改進的 GAT 實現
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
import math


class RelationAwarePooling(nn.Module):
    """
    Relation-aware pooling 機制
    根據面向間的關係動態調整池化權重
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        初始化 Relation-aware pooling

        參數:
            input_dim: 輸入特徵維度
            num_heads: 注意力頭數
            dropout: Dropout 比率
        """
        super(RelationAwarePooling, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim 必須能被 num_heads 整除"

        # Query, Key, Value 投影
        self.q_proj = nn.Linear(input_dim, input_dim)
        self.k_proj = nn.Linear(input_dim, input_dim)
        self.v_proj = nn.Linear(input_dim, input_dim)

        # 輸出投影
        self.out_proj = nn.Linear(input_dim, input_dim)

        # 門控機制
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            x: 輸入特徵 [batch, num_aspects, input_dim]
            mask: 面向掩碼 [batch, num_aspects]

        返回:
            (池化後的表示 [batch, input_dim], 注意力權重 [batch, num_aspects])
        """
        batch_size, num_aspects, _ = x.size()

        # Multi-head attention
        q = self.q_proj(x).view(batch_size, num_aspects, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, num_aspects, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, num_aspects, self.num_heads, self.head_dim)

        # Transpose for attention: [batch, num_heads, num_aspects, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, num_aspects]
            scores = scores.masked_fill(mask_expanded == 0, -1e9)

        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        attn_output = torch.matmul(attn_weights, v)  # [batch, num_heads, num_aspects, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [batch, num_aspects, num_heads, head_dim]
        attn_output = attn_output.view(batch_size, num_aspects, self.input_dim)

        # Output projection
        attn_output = self.out_proj(attn_output)

        # 計算全局池化 - 使用平均和最大池化的組合
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            # Average pooling
            avg_pool = (attn_output * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True)
            # Max pooling
            max_pool = (attn_output + (1 - mask_expanded) * (-1e9)).max(dim=1)[0]
        else:
            avg_pool = attn_output.mean(dim=1)
            max_pool = attn_output.max(dim=1)[0]

        # 門控融合
        gate_input = torch.cat([avg_pool, max_pool], dim=-1)
        gate_weight = self.gate(gate_input)
        pooled = gate_weight * avg_pool + (1 - gate_weight) * max_pool

        pooled = self.layer_norm(pooled)

        # 返回池化結果和平均注意力權重
        avg_attn_weights = attn_weights.mean(dim=1).mean(dim=1)  # [batch, num_aspects]

        return pooled, avg_attn_weights


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss 用於增強 aspect 間的區分度
    使用 InfoNCE 損失
    """

    def __init__(
        self,
        temperature: float = 0.07,
        reduction: str = 'mean'
    ):
        """
        初始化 Contrastive Loss

        參數:
            temperature: 溫度參數，控制分布的平滑度
            reduction: 損失歸約方式
        """
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        計算對比損失

        參數:
            features: 特徵表示 [batch, num_aspects, dim]
            labels: 標籤 [batch, num_aspects] - 相同標籤的為正樣本
            mask: 面向掩碼 [batch, num_aspects]

        返回:
            對比損失值
        """
        batch_size, num_aspects, dim = features.size()
        device = features.device

        # 展平為 [batch * num_aspects, dim]
        features_flat = features.view(-1, dim)
        labels_flat = labels.view(-1)

        if mask is not None:
            mask_flat = mask.view(-1).bool()
            features_flat = features_flat[mask_flat]
            labels_flat = labels_flat[mask_flat]

        # 正規化特徵
        features_norm = F.normalize(features_flat, dim=1)

        # 計算相似度矩陣
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature

        # 創建標籤匹配矩陣
        labels_expanded = labels_flat.unsqueeze(0)
        labels_match = (labels_expanded == labels_expanded.T).float()

        # 移除對角線（自己與自己）
        mask_diag = torch.eye(labels_match.size(0), device=device).bool()
        labels_match = labels_match.masked_fill(mask_diag, 0)

        # 計算 InfoNCE 損失
        # 對於每個樣本，正樣本的相似度應該高於負樣本
        exp_sim = torch.exp(similarity_matrix)
        exp_sim = exp_sim.masked_fill(mask_diag, 0)

        # 正樣本的概率
        pos_sim = (exp_sim * labels_match).sum(dim=1)
        all_sim = exp_sim.sum(dim=1)

        # 避免除以零
        loss = -torch.log((pos_sim + 1e-8) / (all_sim + 1e-8))

        # 只計算有正樣本的損失
        has_positive = labels_match.sum(dim=1) > 0
        if has_positive.sum() > 0:
            loss = loss[has_positive]

            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss
        else:
            return torch.tensor(0.0, device=device, requires_grad=True)


class EnhancedGraphAttentionLayer(nn.Module):
    """
    增強的圖注意力層
    改進:
    - 使用多層感知器計算注意力
    - 加入邊特徵
    - 殘差連接和層歸一化
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        use_edge_features: bool = True,
        use_residual: bool = True
    ):
        """
        初始化增強的圖注意力層

        參數:
            in_features: 輸入特徵維度
            out_features: 輸出特徵維度
            dropout: Dropout 比率
            alpha: LeakyReLU 的負斜率
            use_edge_features: 是否使用邊特徵
            use_residual: 是否使用殘差連接
        """
        super(EnhancedGraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_edge_features = use_edge_features
        self.use_residual = use_residual

        # 節點特徵投影
        self.W = nn.Linear(in_features, out_features, bias=False)

        # 注意力機制 - 使用 MLP
        attn_input_dim = 2 * out_features
        if use_edge_features:
            attn_input_dim += out_features  # 加入邊特徵

        self.attention_mlp = nn.Sequential(
            nn.Linear(attn_input_dim, out_features),
            nn.LeakyReLU(alpha),
            nn.Dropout(dropout),
            nn.Linear(out_features, 1)
        )

        # 邊特徵編碼
        if use_edge_features:
            self.edge_encoder = nn.Sequential(
                nn.Linear(in_features * 2, out_features),
                nn.LeakyReLU(alpha),
                nn.Dropout(dropout)
            )

        # 殘差連接的投影（如果維度不同）
        if use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = None

        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            h: 輸入特徵 [batch, N, in_features]
            adj: 鄰接矩陣 [batch, N, N]

        返回:
            輸出特徵 [batch, N, out_features]
        """
        batch_size, N, _ = h.size()

        # 保存殘差
        residual = h

        # 節點特徵變換
        Wh = self.W(h)  # [batch, N, out_features]

        # 計算注意力
        # [batch, N, 1, out_features] -> [batch, N, N, out_features]
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)

        # 拼接節點特徵
        attn_input = torch.cat([Wh_i, Wh_j], dim=-1)  # [batch, N, N, 2*out_features]

        # 加入邊特徵
        if self.use_edge_features:
            h_i = h.unsqueeze(2).expand(-1, -1, N, -1)
            h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
            edge_features = torch.cat([h_i, h_j], dim=-1)  # [batch, N, N, 2*in_features]
            edge_features = self.edge_encoder(edge_features)  # [batch, N, N, out_features]
            attn_input = torch.cat([attn_input, edge_features], dim=-1)

        # 計算注意力分數
        e = self.attention_mlp(attn_input).squeeze(-1)  # [batch, N, N]

        # 應用鄰接矩陣
        if adj is not None:
            e = e.masked_fill(adj == 0, -1e9)

        # 注意力權重
        attention = F.softmax(e, dim=-1)  # [batch, N, N]
        attention = self.dropout(attention)

        # 加權聚合
        h_prime = torch.matmul(attention, Wh)  # [batch, N, out_features]

        # 殘差連接
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            h_prime = h_prime + residual

        # Layer Normalization
        h_prime = self.layer_norm(h_prime)

        return h_prime


class MultiHeadEnhancedGraphAttention(nn.Module):
    """
    多頭增強圖注意力
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_edge_features: bool = True
    ):
        """
        初始化多頭增強圖注意力

        參數:
            in_features: 輸入特徵維度
            out_features: 輸出特徵維度
            num_heads: 注意力頭數
            dropout: Dropout 比率
            use_edge_features: 是否使用邊特徵
        """
        super(MultiHeadEnhancedGraphAttention, self).__init__()

        self.num_heads = num_heads
        self.out_features = out_features

        assert out_features % num_heads == 0, "out_features 必須能被 num_heads 整除"
        head_dim = out_features // num_heads

        # 多個注意力頭
        self.attentions = nn.ModuleList([
            EnhancedGraphAttentionLayer(
                in_features=in_features,
                out_features=head_dim,
                dropout=dropout,
                use_edge_features=use_edge_features,
                use_residual=False  # 在整個模組層面處理殘差
            )
            for _ in range(num_heads)
        ])

        # 輸出投影
        self.output_projection = nn.Linear(out_features, out_features)

        # 殘差投影
        if in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features)
        else:
            self.residual_proj = None

        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            h: 輸入特徵 [batch, N, in_features]
            adj: 鄰接矩陣 [batch, N, N]

        返回:
            輸出特徵 [batch, N, out_features]
        """
        residual = h

        # 多頭注意力
        head_outputs = [att(h, adj) for att in self.attentions]

        # 拼接 [batch, N, out_features]
        concat_output = torch.cat(head_outputs, dim=-1)

        # 輸出投影
        output = self.output_projection(concat_output)
        output = self.dropout(output)

        # 殘差連接
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)
        output = output + residual

        # Layer Normalization
        output = self.layer_norm(output)

        return output


class IARMEnhanced(nn.Module):
    """
    IARM Enhanced 模組: 增強版面向間關係建模

    增強功能:
        1. Relation-aware pooling 機制
        2. Contrastive loss 支援
        3. 改進的 GAT 實現
    """

    def __init__(
        self,
        input_dim: int,
        relation_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_edge_features: bool = True,
        use_relation_pooling: bool = True,
        pooling_heads: int = 4,
        contrastive_temperature: float = 0.07
    ):
        """
        初始化 IARMEnhanced 模組

        參數:
            input_dim: 輸入特徵維度
            relation_dim: 關係嵌入維度
            num_heads: GAT 注意力頭數
            num_layers: GAT 層數
            dropout: Dropout 比率
            use_edge_features: 是否使用邊特徵
            use_relation_pooling: 是否使用 relation-aware pooling
            pooling_heads: Pooling 的注意力頭數
            contrastive_temperature: 對比學習的溫度參數
        """
        super(IARMEnhanced, self).__init__()

        self.use_relation_pooling = use_relation_pooling
        self.num_layers = num_layers

        # 輸入投影
        self.input_projection = nn.Linear(input_dim, relation_dim)

        # 多層增強圖注意力
        self.gat_layers = nn.ModuleList([
            MultiHeadEnhancedGraphAttention(
                in_features=relation_dim,
                out_features=relation_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_features=use_edge_features
            )
            for _ in range(num_layers)
        ])

        # Relation-aware pooling
        if use_relation_pooling:
            self.relation_pooling = RelationAwarePooling(
                input_dim=relation_dim,
                num_heads=pooling_heads,
                dropout=dropout
            )

        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(
            temperature=contrastive_temperature
        )

        # 輸出投影
        self.output_projection = nn.Linear(relation_dim, input_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def _create_adjacency_matrix(
        self,
        batch_size: int,
        num_aspects: int,
        device: torch.device,
        use_self_loop: bool = True
    ) -> torch.Tensor:
        """
        創建鄰接矩陣（全連接圖）

        參數:
            batch_size: 批次大小
            num_aspects: 面向數
            device: 設備
            use_self_loop: 是否使用自環

        返回:
            鄰接矩陣 [batch, num_aspects, num_aspects]
        """
        adj = torch.ones(batch_size, num_aspects, num_aspects, device=device)

        if not use_self_loop:
            mask = torch.eye(num_aspects, device=device).unsqueeze(0)
            adj = adj * (1 - mask)

        return adj

    def forward(
        self,
        aspect_representations: torch.Tensor,
        aspect_labels: Optional[torch.Tensor] = None,
        aspect_mask: Optional[torch.Tensor] = None,
        return_contrastive_loss: bool = False
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向傳播

        參數:
            aspect_representations: 面向表示 [batch, num_aspects, input_dim]
            aspect_labels: 面向標籤 [batch, num_aspects] - 用於對比學習
            aspect_mask: 面向掩碼 [batch, num_aspects]
            return_contrastive_loss: 是否返回對比損失

        返回:
            (增強的面向表示 [batch, num_aspects, input_dim], 資訊字典)
        """
        batch_size, num_aspects, input_dim = aspect_representations.size()
        device = aspect_representations.device

        # 輸入投影
        h = self.input_projection(aspect_representations)  # [batch, num_aspects, relation_dim]

        # 創建鄰接矩陣（全連接）
        adj = self._create_adjacency_matrix(batch_size, num_aspects, device)

        # 應用掩碼
        if aspect_mask is not None:
            mask_expanded = aspect_mask.unsqueeze(1) * aspect_mask.unsqueeze(2)
            adj = adj * mask_expanded

        # 多層 GAT
        gat_outputs = []
        for layer in self.gat_layers:
            h = layer(h, adj)
            gat_outputs.append(h)

        # 輸出投影
        output = self.output_projection(h)  # [batch, num_aspects, input_dim]
        output = self.dropout(output)

        # 殘差連接
        output = output + aspect_representations
        output = self.layer_norm(output)

        # 收集返回資訊
        info_dict = {}

        # Relation-aware pooling
        if self.use_relation_pooling:
            pooled, pooling_weights = self.relation_pooling(h, aspect_mask)
            info_dict['pooled_representation'] = pooled
            info_dict['pooling_weights'] = pooling_weights

        # Contrastive loss
        if return_contrastive_loss and aspect_labels is not None:
            contrastive_loss = self.contrastive_loss(h, aspect_labels, aspect_mask)
            info_dict['contrastive_loss'] = contrastive_loss

        return output, info_dict


if __name__ == "__main__":
    # 測試 IARMEnhanced 模組
    print("測試 IARMEnhanced 模組...")

    batch_size = 4
    num_aspects = 3
    input_dim = 256

    # 創建模型
    print("\n1. 創建 IARMEnhanced 模型:")
    iarm = IARMEnhanced(
        input_dim=input_dim,
        relation_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        use_edge_features=True,
        use_relation_pooling=True,
        pooling_heads=4
    )

    print(f"  參數量: {sum(p.numel() for p in iarm.parameters()):,}")

    # 測試輸入
    aspect_reprs = torch.randn(batch_size, num_aspects, input_dim)
    aspect_labels = torch.randint(0, 3, (batch_size, num_aspects))
    aspect_mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0],
        [1, 0, 0],
        [1, 1, 1]
    ], dtype=torch.float)

    # 前向傳播
    print("\n2. 測試前向傳播:")
    output, info = iarm(
        aspect_reprs,
        aspect_labels=aspect_labels,
        aspect_mask=aspect_mask,
        return_contrastive_loss=True
    )

    print(f"  輸入形狀: {aspect_reprs.shape}")
    print(f"  輸出形狀: {output.shape}")
    print(f"  池化表示形狀: {info['pooled_representation'].shape}")
    print(f"  池化權重形狀: {info['pooling_weights'].shape}")
    print(f"  對比損失: {info['contrastive_loss'].item():.4f}")

    # 測試 Relation-aware pooling
    print("\n3. 測試 Relation-aware pooling:")
    pooling = RelationAwarePooling(input_dim=128, num_heads=4)
    x = torch.randn(batch_size, num_aspects, 128)
    pooled, weights = pooling(x, aspect_mask)
    print(f"  輸入形狀: {x.shape}")
    print(f"  池化輸出形狀: {pooled.shape}")
    print(f"  池化權重形狀: {weights.shape}")
    print(f"  池化權重示例:\n{weights[0]}")

    # 測試 Contrastive Loss
    print("\n4. 測試 Contrastive Loss:")
    contrastive = ContrastiveLoss(temperature=0.07)
    features = torch.randn(batch_size, num_aspects, 128)
    labels = torch.tensor([
        [0, 0, 1],
        [1, 1, 2],
        [0, 1, 2],
        [2, 2, 2]
    ])
    loss = contrastive(features, labels, aspect_mask)
    print(f"  特徵形狀: {features.shape}")
    print(f"  標籤形狀: {labels.shape}")
    print(f"  對比損失: {loss.item():.4f}")

    print("\nIARMEnhanced 模組測試完成！")


# ============================================================================
# Multi-Aspect IARM (真正的面向間關係建模)
# ============================================================================

class IARMMultiAspect(nn.Module):
    """
    Multi-Aspect Inter-Aspect Relation Modeling

    真正的面向間關係建模模組
    設計用於句子級別的多 aspect 關係學習
    """

    def __init__(
        self,
        input_dim: int,
        relation_dim: int = None,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        relation_mode: str = 'transformer'  # 'transformer', 'gat', 'bilinear'
    ):
        """
        初始化 Multi-Aspect IARM

        參數:
            input_dim: 輸入維度（來自 PMAC 的組合特徵）
            relation_dim: 關係特徵維度（如果為 None，則使用 input_dim）
            num_heads: 注意力頭數
            num_layers: Transformer 層數
            dropout: Dropout 比率
            relation_mode: 關係建模模式
                - 'transformer': 使用 Transformer encoder
                - 'gat': 使用 Graph Attention Network
                - 'bilinear': 使用雙線性交互
        """
        super(IARMMultiAspect, self).__init__()

        self.input_dim = input_dim
        self.relation_dim = relation_dim if relation_dim is not None else input_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.relation_mode = relation_mode

        # 投影層
        if self.input_dim != self.relation_dim:
            self.input_projection = nn.Linear(input_dim, self.relation_dim)
        else:
            self.input_projection = nn.Identity()

        # 根據模式初始化關係建模組件
        if relation_mode == 'transformer':
            # Transformer Encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.relation_dim,
                nhead=num_heads,
                dim_feedforward=self.relation_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer,
                num_layers=num_layers
            )

        elif relation_mode == 'gat':
            # Graph Attention Network layers
            self.gat_layers = nn.ModuleList([
                self._build_gat_layer(self.relation_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])

        elif relation_mode == 'bilinear':
            # Bilinear interaction
            self.bilinear_layers = nn.ModuleList([
                nn.Bilinear(self.relation_dim, self.relation_dim, self.relation_dim)
                for _ in range(num_layers)
            ])
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(self.relation_dim)
                for _ in range(num_layers)
            ])

        # Relation-aware pooling（可選）
        self.relation_pooling = RelationAwarePooling(
            input_dim=self.relation_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 輸出投影
        self.output_projection = nn.Sequential(
            nn.Linear(self.relation_dim, self.relation_dim),
            nn.LayerNorm(self.relation_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

    def _build_gat_layer(self, dim: int, num_heads: int, dropout: float):
        """構建單層 GAT"""
        return nn.ModuleDict({
            'attention': nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ),
            'ffn': nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim),
                nn.Dropout(dropout)
            ),
            'norm1': nn.LayerNorm(dim),
            'norm2': nn.LayerNorm(dim)
        })

    def forward(
        self,
        aspect_features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-Aspect 關係建模

        參數:
            aspect_features: [batch, num_aspects, dim]
            aspect_mask: [batch, num_aspects] - bool mask

        返回:
            enhanced_features: [batch, num_aspects, relation_dim]
            relation_info: 字典，包含關係信息
        """
        batch_size, num_aspects, _ = aspect_features.shape

        # 投影
        features = self.input_projection(aspect_features)  # [batch, N, relation_dim]

        # 根據模式建模關係
        if self.relation_mode == 'transformer':
            enhanced, relation_info = self._transformer_relation(features, aspect_mask)
        elif self.relation_mode == 'gat':
            enhanced, relation_info = self._gat_relation(features, aspect_mask)
        elif self.relation_mode == 'bilinear':
            enhanced, relation_info = self._bilinear_relation(features, aspect_mask)
        else:
            raise ValueError(f"Unknown relation_mode: {self.relation_mode}")

        # 輸出投影
        output = self.output_projection(enhanced)

        # 殘差連接
        output = features + self.dropout(output)

        return output, relation_info

    def _transformer_relation(
        self,
        features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        使用 Transformer 建模 aspect 關係

        Transformer 的 self-attention 自然捕捉序列中的關係
        """
        # 創建 padding mask（True 表示 padding，需要被 mask）
        # PyTorch Transformer 期望 True = padding
        padding_mask = ~aspect_mask  # [batch, num_aspects]

        # Transformer forward
        enhanced = self.transformer(
            features,
            src_key_padding_mask=padding_mask
        )  # [batch, num_aspects, dim]

        # 提取關係信息（可以從最後一層的 attention 獲取）
        relation_info = {
            'mode': 'transformer',
            'num_layers': self.num_layers
        }

        return enhanced, relation_info

    def _gat_relation(
        self,
        features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        使用 GAT 建模 aspect 關係

        將 aspects 視為圖節點，建模它們之間的關係邊
        """
        enhanced = features
        attention_weights_list = []

        # 多層 GAT
        for layer in self.gat_layers:
            # Self-attention（作為圖注意力）
            attn_output, attn_weights = layer['attention'](
                enhanced, enhanced, enhanced,
                key_padding_mask=~aspect_mask,
                need_weights=True
            )

            # 殘差連接 + Norm
            enhanced = layer['norm1'](enhanced + self.dropout(attn_output))

            # FFN
            ffn_output = layer['ffn'](enhanced)
            enhanced = layer['norm2'](enhanced + ffn_output)

            attention_weights_list.append(attn_weights)

        relation_info = {
            'mode': 'gat',
            'attention_weights': attention_weights_list  # 可用於可視化
        }

        return enhanced, relation_info

    def _bilinear_relation(
        self,
        features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        使用雙線性交互建模 aspect 關係

        計算每對 aspects 之間的雙線性相似度
        """
        batch_size, num_aspects, dim = features.shape
        enhanced = features.clone()

        for layer_idx, (bilinear, norm) in enumerate(zip(self.bilinear_layers, self.layer_norms)):
            # 對每個樣本計算兩兩交互
            layer_output = torch.zeros_like(features)

            for b in range(batch_size):
                valid_mask = aspect_mask[b]
                valid_indices = torch.where(valid_mask)[0]
                num_valid = len(valid_indices)

                if num_valid <= 1:
                    layer_output[b] = enhanced[b]
                    continue

                # 計算所有有效 aspects 的兩兩交互
                for i, idx_i in enumerate(valid_indices):
                    interactions = []

                    for j, idx_j in enumerate(valid_indices):
                        if i == j:
                            continue

                        # 雙線性交互
                        interaction = bilinear(
                            enhanced[b, idx_i].unsqueeze(0),
                            enhanced[b, idx_j].unsqueeze(0)
                        ).squeeze(0)

                        interactions.append(interaction)

                    if interactions:
                        # 聚合交互信息
                        aggregated = torch.stack(interactions).mean(dim=0)
                        layer_output[b, idx_i] = aggregated
                    else:
                        layer_output[b, idx_i] = enhanced[b, idx_i]

            # 殘差 + Norm
            enhanced = norm(enhanced + self.dropout(layer_output))

        relation_info = {
            'mode': 'bilinear',
            'num_layers': self.num_layers
        }

        return enhanced, relation_info

    def compute_relation_matrix(
        self,
        aspect_features: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 aspect 關係矩陣（用於可視化）

        參數:
            aspect_features: [batch, num_aspects, dim]
            aspect_mask: [batch, num_aspects]

        返回:
            relation_matrix: [batch, num_aspects, num_aspects]
                表示 aspects 之間的關係強度
        """
        batch_size, num_aspects, dim = aspect_features.shape

        # 計算相似度矩陣
        # 使用 cosine similarity 或 dot product
        features_norm = F.normalize(aspect_features, p=2, dim=-1)
        relation_matrix = torch.matmul(
            features_norm,
            features_norm.transpose(-2, -1)
        )  # [batch, N, N]

        # 應用 mask
        mask_expanded = aspect_mask.unsqueeze(1) & aspect_mask.unsqueeze(2)
        relation_matrix = relation_matrix.masked_fill(~mask_expanded, 0)

        return relation_matrix
