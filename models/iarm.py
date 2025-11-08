"""
IARM 模組: Inter-Aspect Relation Modeling
面向間關係建模

功能:
    - 建模多個面向之間的依賴關係
    - 使用圖注意力網絡或 Transformer 式交互
    - 生成關係增強的面向表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class GraphAttentionLayer(nn.Module):
    """
    圖注意力層 (GAT)
    用於建模面向之間的關係
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.3,
        alpha: float = 0.2,
        concat: bool = True
    ):
        """
        初始化圖注意力層

        參數:
            in_features: 輸入特徵維度
            out_features: 輸出特徵維度
            dropout: Dropout 比率
            alpha: LeakyReLU 的負斜率
            concat: 是否拼接（True）或平均（False）
        """
        super(GraphAttentionLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 權重矩陣
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        # 注意力參數
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            h: 輸入特徵 [batch, N, in_features]
            adj: 鄰接矩陣 [batch, N, N] (None 表示全連接)

        返回:
            輸出特徵 [batch, N, out_features]
        """
        batch_size, N, _ = h.size()

        # 線性變換 [batch, N, out_features]
        Wh = torch.matmul(h, self.W)

        # 計算注意力係數
        # [batch, N, 1, out_features] 擴展為 [batch, N, N, out_features]
        Wh_i = Wh.unsqueeze(2).expand(-1, -1, N, -1)
        # [batch, 1, N, out_features] 擴展為 [batch, N, N, out_features]
        Wh_j = Wh.unsqueeze(1).expand(-1, N, -1, -1)

        # 拼接 [batch, N, N, 2*out_features]
        concat_features = torch.cat([Wh_i, Wh_j], dim=3)

        # 計算注意力能量 [batch, N, N, 1]
        e = self.leakyrelu(torch.matmul(concat_features, self.a))
        e = e.squeeze(3)  # [batch, N, N]

        # 應用鄰接矩陣（如果有）
        if adj is not None:
            e = e.masked_fill(adj == 0, -1e9)

        # 注意力權重 [batch, N, N]
        attention = F.softmax(e, dim=2)
        attention = self.dropout_layer(attention)

        # 加權求和 [batch, N, out_features]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime


class MultiHeadGraphAttention(nn.Module):
    """
    多頭圖注意力
    結合多個注意力頭的輸出
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        初始化多頭圖注意力

        參數:
            in_features: 輸入特徵維度
            out_features: 輸出特徵維度
            num_heads: 注意力頭數
            dropout: Dropout 比率
        """
        super(MultiHeadGraphAttention, self).__init__()

        self.num_heads = num_heads
        self.out_features = out_features

        # 多個注意力頭
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(
                in_features,
                out_features,
                dropout=dropout,
                concat=True
            )
            for _ in range(num_heads)
        ])

        # 輸出投影
        self.output_projection = nn.Linear(out_features * num_heads, out_features)
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
        # 多頭注意力
        head_outputs = [att(h, adj) for att in self.attentions]

        # 拼接 [batch, N, out_features * num_heads]
        concat_output = torch.cat(head_outputs, dim=2)

        # 投影到輸出維度 [batch, N, out_features]
        output = self.output_projection(concat_output)

        return self.dropout(output)


class TransformerInteraction(nn.Module):
    """
    Transformer 式交互
    使用自注意力建模面向間關係
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3
    ):
        """
        初始化 Transformer 交互層

        參數:
            d_model: 模型維度
            nhead: 注意力頭數
            dim_feedforward: 前饋網路維度
            dropout: Dropout 比率
        """
        super(TransformerInteraction, self).__init__()

        # 多頭自注意力
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )

        # 前饋網路
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            x: 輸入 [batch, N, d_model]
            mask: 注意力掩碼 [batch, N]

        返回:
            (輸出 [batch, N, d_model], 注意力權重 [batch, nhead, N, N])
        """
        # 自注意力
        attn_output, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=mask if mask is not None else None,
            need_weights=True
        )

        # 殘差連接 + Layer Norm
        x = self.norm1(x + self.dropout(attn_output))

        # 前饋網路
        ff_output = self.feed_forward(x)

        # 殘差連接 + Layer Norm
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class IARM(nn.Module):
    """
    IARM 模組: Inter-Aspect Relation Modeling
    面向間關係建模

    功能:
        - 建模多個面向之間的依賴關係
        - 支援圖注意力或 Transformer 兩種方式
        - 生成關係增強的面向表示
    """

    def __init__(
        self,
        input_dim: int,
        relation_dim: int = 128,
        relation_type: str = "graph_attention",
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        use_self_loop: bool = True
    ):
        """
        初始化 IARM 模組

        參數:
            input_dim: 輸入特徵維度
            relation_dim: 關係嵌入維度
            relation_type: 關係建模類型 ("graph_attention", "transformer")
            num_heads: 注意力頭數
            num_layers: 關係建模層數
            dropout: Dropout 比率
            use_self_loop: 是否使用自環
        """
        super(IARM, self).__init__()

        self.relation_type = relation_type
        self.use_self_loop = use_self_loop
        self.num_layers = num_layers

        # 輸入投影
        self.input_projection = nn.Linear(input_dim, relation_dim)

        # 選擇關係建模方式
        if relation_type == "graph_attention":
            self.relation_layers = nn.ModuleList([
                MultiHeadGraphAttention(
                    in_features=relation_dim,
                    out_features=relation_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])

        elif relation_type == "transformer":
            self.relation_layers = nn.ModuleList([
                TransformerInteraction(
                    d_model=relation_dim,
                    nhead=num_heads,
                    dim_feedforward=relation_dim * 2,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])

        else:
            raise ValueError(f"不支援的關係類型: {relation_type}")

        # 輸出投影
        self.output_projection = nn.Linear(relation_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def _create_adjacency_matrix(
        self,
        batch_size: int,
        num_aspects: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        創建鄰接矩陣（全連接圖）

        參數:
            batch_size: 批次大小
            num_aspects: 面向數
            device: 設備

        返回:
            鄰接矩陣 [batch, num_aspects, num_aspects]
        """
        # 全連接（除了對角線，如果不使用自環）
        adj = torch.ones(batch_size, num_aspects, num_aspects, device=device)

        if not self.use_self_loop:
            # 移除自環
            mask = torch.eye(num_aspects, device=device).unsqueeze(0)
            adj = adj * (1 - mask)

        return adj

    def forward(
        self,
        aspect_representations: torch.Tensor,
        aspect_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向傳播

        參數:
            aspect_representations: 面向表示 [batch, num_aspects, input_dim]
            aspect_mask: 面向掩碼 [batch, num_aspects] (1 表示有效，0 表示填充)

        返回:
            (關係增強的面向表示 [batch, num_aspects, input_dim], 注意力權重字典)
        """
        batch_size, num_aspects, input_dim = aspect_representations.size()
        device = aspect_representations.device

        # 輸入投影 [batch, num_aspects, relation_dim]
        h = self.input_projection(aspect_representations)

        # 創建鄰接矩陣（如果使用圖注意力）
        if self.relation_type == "graph_attention":
            adj = self._create_adjacency_matrix(batch_size, num_aspects, device)

            # 應用掩碼
            if aspect_mask is not None:
                # 將無效面向的邊設為 0
                mask_expanded = aspect_mask.unsqueeze(1) * aspect_mask.unsqueeze(2)
                adj = adj * mask_expanded

        attention_weights_list = []

        # 多層關係建模
        for i, layer in enumerate(self.relation_layers):
            if self.relation_type == "graph_attention":
                h = layer(h, adj)
                # 圖注意力層沒有返回注意力權重，這裡設為 None
                attention_weights_list.append(None)

            elif self.relation_type == "transformer":
                # Transformer 需要將 mask 反轉（True 表示忽略）
                padding_mask = ~aspect_mask.bool() if aspect_mask is not None else None
                h, attn_weights = layer(h, padding_mask)
                attention_weights_list.append(attn_weights)

        # 輸出投影 [batch, num_aspects, input_dim]
        output = self.output_projection(h)
        output = self.dropout(output)

        # 殘差連接
        output = output + aspect_representations

        # 收集注意力權重
        attention_weights = {
            f'layer_{i}': weights
            for i, weights in enumerate(attention_weights_list)
            if weights is not None
        }

        return output, attention_weights


if __name__ == "__main__":
    # 測試 IARM 模組
    print("測試 IARM 模組...")

    batch_size = 4
    num_aspects = 3
    input_dim = 256

    # 測試圖注意力版本
    print("\n1. 測試圖注意力版本:")
    iarm_gat = IARM(
        input_dim=input_dim,
        relation_dim=128,
        relation_type="graph_attention",
        num_heads=4,
        num_layers=2,
        dropout=0.3
    )

    aspect_reprs = torch.randn(batch_size, num_aspects, input_dim)
    output_gat, attn_gat = iarm_gat(aspect_reprs)

    print(f"  輸入: {aspect_reprs.shape}")
    print(f"  輸出: {output_gat.shape}")
    print(f"  注意力權重數量: {len(attn_gat)}")

    # 測試 Transformer 版本
    print("\n2. 測試 Transformer 版本:")
    iarm_trans = IARM(
        input_dim=input_dim,
        relation_dim=128,
        relation_type="transformer",
        num_heads=4,
        num_layers=2,
        dropout=0.3
    )

    output_trans, attn_trans = iarm_trans(aspect_reprs)

    print(f"  輸入: {aspect_reprs.shape}")
    print(f"  輸出: {output_trans.shape}")
    print(f"  注意力權重數量: {len(attn_trans)}")

    if 'layer_0' in attn_trans:
        print(f"  第一層注意力權重形狀: {attn_trans['layer_0'].shape}")

    # 測試帶掩碼
    print("\n3. 測試帶掩碼（處理變長面向）:")
    aspect_mask = torch.tensor([
        [1, 1, 1],  # 3 個有效面向
        [1, 1, 0],  # 2 個有效面向
        [1, 0, 0],  # 1 個有效面向
        [1, 1, 1]   # 3 個有效面向
    ], dtype=torch.float)

    output_masked, _ = iarm_trans(aspect_reprs, aspect_mask)
    print(f"  帶掩碼輸出: {output_masked.shape}")

    print("\nIARM 模組測試完成！")
