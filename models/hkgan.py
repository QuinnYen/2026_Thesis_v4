"""
HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network (v3.0)

核心創新：
1. 階層式 BERT 特徵提取 (Low/Mid/High layers)
2. SenticNet 知識增強 - 情感極性融入注意力計算
3. 純 PyTorch 實現的階層式 GAT
4. 跨面向關係建模
5. 動態知識門控 (v3.0) - 解決 MAMS 複雜句問題

v3.0 新增（解決 MAMS 複雜句問題）：
- 動態知識門控 (Dynamic Knowledge Gating)：
  * 舊版（硬性注入）：Feature_new = Feature_BERT + λ * SenticNet
  * 新版（軟性門控）：
      Gate = Sigmoid(Linear([Feature_BERT, SenticNet]))
      Feature_new = (1 - Gate) * Feature_BERT + Gate * SenticNet_embed
- 行為：
  * 簡單句（Laptops/Restaurants）：Gate 高，利用 SenticNet 增強
  * 複雜句（MAMS 轉折句 "But", "However"）：Gate 低，只相信 BERT
- 預期效果：MAMS F1 85%+

v2.3 新增（解決蹺蹺板效應）：
- 情感感知隔離 (Sentiment-Aware Isolation)：根據「情感一致性」動態調整隔離程度

v2.2 新增（解決情感洩漏問題）：
- 情感隔離機制 (Sentiment Isolation)：防止強烈情感透過 IARN 流向中性面向

v2.1 修復（解決 Neutral 識別問題）：
- ConfidenceGate 公式修復
- 移除 coverage_mask 的雙重抑制

論文創新點：
- 首次將情感知識庫 (SenticNet) 與階層式 GAT 結合
- 雙階層設計：BERT 內部階層 + 語言學階層
- 知識增強的圖注意力機制
- 動態知識門控：根據上下文自動決定是否信任外部知識
- 情感感知隔離：根據情感一致性動態調整跨面向信息流

Reference:
    - Veličković et al. "Graph Attention Networks" (ICLR 2018)
    - Cambria et al. "SenticNet 5" (AAAI 2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

from models.bert_embedding import BERTForABSA


class GraphAttentionLayer(nn.Module):
    """
    純 PyTorch 實現的圖注意力層 (不依賴 torch_geometric)

    公式：
        e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        α_ij = softmax_j(e_ij)
        h'_i = σ(Σ α_ij * Wh_j)

    Args:
        in_features: 輸入特徵維度
        out_features: 輸出特徵維度
        dropout: Dropout 比率
        alpha: LeakyReLU 負斜率
        concat: 是否拼接輸出（用於多頭）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        dropout: float = 0.1,
        alpha: float = 0.2,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # 線性變換 W
        self.W = nn.Linear(in_features, out_features, bias=False)

        # 注意力參數 a (分為兩部分便於計算)
        self.a_src = nn.Parameter(torch.zeros(out_features, 1))
        self.a_dst = nn.Parameter(torch.zeros(out_features, 1))

        self.leaky_relu = nn.LeakyReLU(alpha)
        self.dropout_layer = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        """初始化參數"""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        Args:
            h: [batch, seq_len, in_features] 節點特徵
            adj: [batch, seq_len, seq_len] 鄰接矩陣（可選，默認全連接）
            attention_mask: [batch, seq_len] 有效節點掩碼

        Returns:
            h': [batch, seq_len, out_features] 更新後的節點特徵
            attention: [batch, seq_len, seq_len] 注意力權重
        """
        batch_size, seq_len, _ = h.shape

        # 線性變換: [batch, seq_len, out_features]
        Wh = self.W(h)

        # 計算注意力分數
        # e_ij = a_src^T * Wh_i + a_dst^T * Wh_j
        e_src = torch.matmul(Wh, self.a_src)  # [batch, seq_len, 1]
        e_dst = torch.matmul(Wh, self.a_dst)  # [batch, seq_len, 1]

        # 廣播相加得到注意力分數矩陣
        e = e_src + e_dst.transpose(1, 2)  # [batch, seq_len, seq_len]
        e = self.leaky_relu(e)

        # 應用鄰接矩陣掩碼（如果提供）
        if adj is not None:
            # adj = 1 表示有邊，0 表示無邊
            e = e.masked_fill(adj == 0, float('-inf'))

        # 應用 attention_mask（padding mask）
        if attention_mask is not None:
            # attention_mask: [batch, seq_len]，1=有效，0=padding
            mask = attention_mask.unsqueeze(1)  # [batch, 1, seq_len]
            e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax 得到注意力權重
        attention = F.softmax(e, dim=-1)  # [batch, seq_len, seq_len]

        # 處理全 -inf 的情況（避免 NaN）
        attention = torch.nan_to_num(attention, nan=0.0)

        # Dropout
        attention = self.dropout_layer(attention)

        # 加權聚合
        h_prime = torch.bmm(attention, Wh)  # [batch, seq_len, out_features]

        if self.concat:
            return F.elu(h_prime), attention
        else:
            return h_prime, attention


class MultiHeadGAT(nn.Module):
    """
    多頭圖注意力網絡

    使用多個 GAT 頭捕捉不同類型的依賴關係，
    類似於 Multi-Head Self-Attention

    Args:
        in_features: 輸入特徵維度
        out_features: 每個頭的輸出特徵維度
        num_heads: 注意力頭數
        dropout: Dropout 比率
        concat: 是否拼接多頭輸出
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        # 多個 GAT 頭
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(
                in_features=in_features,
                out_features=out_features,
                dropout=dropout,
                concat=True
            )
            for _ in range(num_heads)
        ])

        # 輸出投影（如果拼接）
        if concat:
            self.out_proj = nn.Linear(out_features * num_heads, out_features)
        else:
            self.out_proj = nn.Linear(out_features, out_features)

        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        adj: torch.Tensor = None,
        attention_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        Args:
            h: [batch, seq_len, in_features]
            adj: [batch, seq_len, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            h': [batch, seq_len, out_features]
            attention: [batch, num_heads, seq_len, seq_len]
        """
        head_outputs = []
        head_attentions = []

        for head in self.attention_heads:
            h_head, attn_head = head(h, adj, attention_mask)
            head_outputs.append(h_head)
            head_attentions.append(attn_head)

        if self.concat:
            # 拼接所有頭
            h_cat = torch.cat(head_outputs, dim=-1)  # [batch, seq_len, out*num_heads]
            h_out = self.out_proj(h_cat)
        else:
            # 平均所有頭
            h_stack = torch.stack(head_outputs, dim=0)  # [num_heads, batch, seq_len, out]
            h_mean = h_stack.mean(dim=0)
            h_out = self.out_proj(h_mean)

        # 殘差連接和層正規化
        if h.shape[-1] == h_out.shape[-1]:
            h_out = self.layer_norm(h_out + h)
        else:
            h_out = self.layer_norm(h_out)

        h_out = self.dropout(h_out)

        # 堆疊注意力權重
        attention = torch.stack(head_attentions, dim=1)  # [batch, num_heads, seq_len, seq_len]

        return h_out, attention


class ConfidenceGate(nn.Module):
    """
    信心門控機制 (Confidence Gating) v2.1

    修復版本：解決 v2.0 的門控範圍壓縮問題

    核心思想：讓模型根據 BERT 上下文語義，動態決定是否信任 SenticNet 知識

    問題背景：
    - SenticNet 提供的是「通用領域」的情感極性
    - 在特定語境下（如技術描述），這些極性可能是噪聲
    - 例如："high resolution" 中的 "high" 在 SenticNet 中有正向極性，
      但在技術描述中是中性的

    解決方案：
    - 讓 BERT 的上下文表示學習一個 gate ∈ [0, 1]
    - gate ≈ 1：信任 SenticNet（這是情感表達）
    - gate ≈ 0：忽略 SenticNet（這是客觀陳述）

    v2.1 修復：
    - 移除錯誤的 base_trust 公式（會壓縮門控範圍到 [0.25, 0.75]）
    - 改用可學習的偏置項，讓門控可以達到完整的 [0, 1] 範圍
    - 添加溫度參數控制門控的銳利度

    公式：
        gate = σ((W₂ · ReLU(W₁ · h + b₁) + b₂ + bias) / temperature)
        effective_polarity = gate * senticnet_polarity

    Args:
        hidden_dim: BERT 隱藏層維度
        dropout: Dropout 比率
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        # 門控網絡：從 BERT 上下文預測「是否是情感表達」
        # 注意：最後不加 Sigmoid，我們在 forward 中手動處理
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
            # 移除 Sigmoid，改在 forward 中處理
        )

        # 可學習的偏置項（初始化為正值，讓默認傾向於信任 SenticNet）
        self.gate_bias = nn.Parameter(torch.tensor(1.0))

        # 溫度參數（控制門控的銳利度，較低的溫度產生更極端的 0/1 決策）
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        polarities: torch.Tensor,
        coverage_mask: torch.Tensor = None
    ) -> tuple:
        """
        前向傳播

        Args:
            hidden_states: [batch, seq_len, hidden_dim] BERT 上下文表示
            polarities: [batch, seq_len] SenticNet 原始極性值
            coverage_mask: [batch, seq_len] 知識庫覆蓋掩碼（1=已知，0=未知）

        Returns:
            gated_polarities: [batch, seq_len] 經過門控的極性值
            gate_values: [batch, seq_len] 門控值（用於分析和正則化）
        """
        # 計算 logits（未經 sigmoid）
        gate_logits = self.gate_network(hidden_states).squeeze(-1)  # [batch, seq_len]

        # 加上偏置並除以溫度，然後 sigmoid
        # temperature 需要 clamp 避免除以 0 或負數
        temp = torch.clamp(self.temperature, min=0.1)
        gate_values = torch.sigmoid((gate_logits + self.gate_bias) / temp)

        # 只在「有知識的位置」應用門控
        # 對於未知詞（coverage_mask=0），我們不應用門控抑制，
        # 因為極性本來就是 0，不需要額外處理
        # 這裡不再乘以 coverage_mask，避免雙重抑制

        # 應用門控
        gated_polarities = gate_values * polarities

        return gated_polarities, gate_values


class DynamicKnowledgeGate(nn.Module):
    """
    動態知識門控 (Dynamic Knowledge Gating) v3.0

    核心改進：從「硬性注入」改為「軟性門控」

    舊版（硬性注入）：
        Feature_new = Feature_BERT + λ * SenticNet
        問題：在 MAMS 這類複雜句子中，SenticNet 可能與上下文衝突

    新版（動態門控）：
        Gate = Sigmoid(Linear([Feature_BERT, SenticNet]))
        Feature_new = (1 - Gate) * Feature_BERT + Gate * SenticNet_embed

    行為：
        - 簡單句（Laptops/Restaurants）：Gate 高，利用 SenticNet 增強
        - 複雜句（MAMS 轉折句 "But", "However"）：Gate 低，只相信 BERT

    Args:
        hidden_dim: BERT 隱藏層維度
        dropout: Dropout 比率
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 將 SenticNet 極性嵌入到高維空間
        self.polarity_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        # 動態門控網絡：根據 BERT 特徵和 SenticNet 嵌入決定融合程度
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # 融合後的投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        polarities: torch.Tensor,
        coverage_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        Args:
            hidden_states: [batch, seq_len, hidden_dim] BERT 特徵
            polarities: [batch, seq_len] SenticNet 極性值 [-1, 1]
            coverage_mask: [batch, seq_len] 知識庫覆蓋掩碼（1=已知，0=未知）

        Returns:
            enhanced_features: [batch, seq_len, hidden_dim] 增強後的特徵
            gate_values: [batch, seq_len] 門控值（用於分析）
        """
        batch_size, seq_len, _ = hidden_states.shape

        # 1. 將極性嵌入到高維空間
        pol_input = polarities.unsqueeze(-1)  # [batch, seq_len, 1]
        polarity_embed = self.polarity_embed(pol_input)  # [batch, seq_len, hidden_dim]

        # 2. 計算動態門控值
        gate_input = torch.cat([hidden_states, polarity_embed], dim=-1)  # [batch, seq_len, hidden_dim*2]
        gate_values = self.gate_network(gate_input).squeeze(-1)  # [batch, seq_len]

        # 3. 對未知詞（coverage_mask=0）的門控值設為 0（不使用外部知識）
        if coverage_mask is not None:
            gate_values = gate_values * coverage_mask

        # 4. 軟性融合
        gate_expanded = gate_values.unsqueeze(-1)  # [batch, seq_len, 1]
        fused_features = (1 - gate_expanded) * hidden_states + gate_expanded * polarity_embed

        # 5. 輸出投影（殘差連接）
        enhanced_features = self.output_proj(fused_features) + hidden_states

        return enhanced_features, gate_values


class KnowledgeEnhancedGAT(nn.Module):
    """
    知識增強的圖注意力層（含動態知識門控 v3.0）

    核心改進：
    v3.0 新增「動態知識門控」：
        - 取代原有的「硬性注入」（λ * SenticNet）
        - 改用軟性門控：Gate * SenticNet_embed + (1-Gate) * BERT
        - Gate 由 BERT 上下文動態計算

    原有設計保留：
    1. 原始：e'_ij = e_ij + λ * knowledge_bias_ij（固定 λ）
    2. 改進：e'_ij = e_ij + gate_i * gate_j * knowledge_bias_ij（動態門控）

    其中 gate 由 ConfidenceGate 根據 BERT 上下文計算：
    - 如果上下文顯示這是情感表達 → gate ≈ 1 → 注入知識
    - 如果上下文顯示這是客觀陳述 → gate ≈ 0 → 忽略知識

    這有助於捕捉:
        - "not good" 中 not 和 good 的關係（情感表達，gate 高）
        - "high resolution" 中的 high（技術描述，gate 低）

    Args:
        hidden_dim: 隱藏層維度
        num_heads: 注意力頭數
        knowledge_weight: 知識注入權重 λ（基準值）
        dropout: Dropout 比率
        use_confidence_gate: 是否使用信心門控
        use_dynamic_gate: 是否使用動態知識門控（v3.0 新增）
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        knowledge_weight: float = 0.1,
        dropout: float = 0.1,
        use_confidence_gate: bool = True,
        use_dynamic_gate: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.knowledge_weight = knowledge_weight
        self.use_confidence_gate = use_confidence_gate
        self.use_dynamic_gate = use_dynamic_gate

        # 基礎 GAT
        self.gat = MultiHeadGAT(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # 動態知識門控（v3.0 新增）
        if use_dynamic_gate:
            self.dynamic_gate = DynamicKnowledgeGate(hidden_dim, dropout)
        else:
            self.dynamic_gate = None

        # 信心門控（用於邊權重調整，與動態門控互補）
        if use_confidence_gate:
            self.confidence_gate = ConfidenceGate(hidden_dim, dropout)
        else:
            self.confidence_gate = None

        # 知識偏置計算
        self.knowledge_proj = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(
        self,
        h: torch.Tensor,
        polarities: torch.Tensor = None,
        adj: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        coverage_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        前向傳播（v3.0 新增動態知識門控）

        Args:
            h: [batch, seq_len, hidden_dim] 節點特徵
            polarities: [batch, seq_len] SenticNet 極性值 [-1, 1]
            adj: [batch, seq_len, seq_len] 鄰接矩陣
            attention_mask: [batch, seq_len] 有效節點掩碼
            coverage_mask: [batch, seq_len] 知識庫覆蓋掩碼（用於區分中性與未知）

        Returns:
            h': [batch, seq_len, hidden_dim]
            attention: [batch, num_heads, seq_len, seq_len]
            gate_values: [batch, seq_len] 門控值（動態門控或信心門控）
        """
        batch_size, seq_len, _ = h.shape
        gate_values = None

        # v3.0: 動態知識門控（節點級別的軟性融合）
        if self.use_dynamic_gate and self.dynamic_gate is not None and polarities is not None:
            h, gate_values = self.dynamic_gate(h, polarities, coverage_mask)

        # 計算知識增強的鄰接矩陣（邊級別）
        if polarities is not None and self.knowledge_weight > 0:
            # 如果啟用信心門控，先對極性進行門控
            if self.use_confidence_gate and self.confidence_gate is not None:
                gated_polarities, conf_gate_values = self.confidence_gate(
                    h, polarities, coverage_mask
                )
                adj = self._compute_knowledge_adj(
                    gated_polarities, adj, attention_mask, conf_gate_values
                )
                # 如果沒有動態門控，使用信心門控的值
                if gate_values is None:
                    gate_values = conf_gate_values
            else:
                adj = self._compute_knowledge_adj(polarities, adj, attention_mask)

        # GAT 前向傳播
        h_out, attention = self.gat(h, adj, attention_mask)

        return h_out, attention, gate_values

    def _compute_knowledge_adj(
        self,
        polarities: torch.Tensor,
        adj: torch.Tensor,
        attention_mask: torch.Tensor,
        gate_values: torch.Tensor = None
    ) -> torch.Tensor:
        """
        計算知識增強的鄰接矩陣（支援信心門控 v2.1）

        Args:
            polarities: [batch, seq_len] 極性值（可能已經過門控）
            adj: [batch, seq_len, seq_len] 或 None
            attention_mask: [batch, seq_len]
            gate_values: [batch, seq_len] 門控值（可選）

        Returns:
            enhanced_adj: [batch, seq_len, seq_len]

        v2.1 修復：
        - 舊版使用 gate_i * gate_j（二次方效應），導致知識注入被過度抑制
        - 新版使用 (gate_i + gate_j) / 2（平均門控），保持線性關係
        - 當極性本身已經被門控時，邊權重不再額外乘以 gate_matrix
        """
        batch_size, seq_len = polarities.shape

        # 擴展極性以計算兩兩關係
        pol_i = polarities.unsqueeze(2)  # [batch, seq_len, 1]
        pol_j = polarities.unsqueeze(1)  # [batch, 1, seq_len]

        # 特徵組合: [polarity_i, polarity_j]
        pol_pairs = torch.stack([
            pol_i.expand(-1, -1, seq_len),
            pol_j.expand(-1, seq_len, -1)
        ], dim=-1)  # [batch, seq_len, seq_len, 2]

        # 計算知識偏置
        knowledge_bias = self.knowledge_proj(pol_pairs).squeeze(-1)  # [batch, seq_len, seq_len]

        # 縮放
        knowledge_bias = knowledge_bias * self.knowledge_weight

        # v2.1: 因為極性已經在 ConfidenceGate 中被門控過了，
        # 這裡不再對邊權重進行額外的門控調製
        # 如果真的需要邊級別的調製，使用平均而非乘積（避免二次方抑制）
        #
        # 舊版（問題）: gate_matrix = gate_i * gate_j  → 0.5 * 0.5 = 0.25
        # 新版（修復）: 直接使用已門控的極性，不再額外調製
        #
        # 如果需要邊級調製，可以取消下面的註釋：
        # if gate_values is not None:
        #     gate_i = gate_values.unsqueeze(2)
        #     gate_j = gate_values.unsqueeze(1)
        #     gate_matrix = (gate_i + gate_j) / 2  # 平均而非乘積
        #     knowledge_bias = knowledge_bias * gate_matrix

        # 與原始鄰接矩陣結合
        if adj is None:
            # 默認全連接
            enhanced_adj = torch.ones(batch_size, seq_len, seq_len, device=polarities.device)
        else:
            enhanced_adj = adj.clone()

        # 添加知識偏置（作為邊權重）
        enhanced_adj = enhanced_adj + knowledge_bias

        # 確保非負
        enhanced_adj = F.relu(enhanced_adj)

        return enhanced_adj


class HierarchicalGATLayer(nn.Module):
    """
    階層式圖注意力層（含信心門控和動態知識門控）

    設計理念：三層圖傳播，從局部到全局
        - Level 1 (Token): 局部依賴 (window=3)
        - Level 2 (Phrase): 短語級依賴 (window=5)
        - Level 3 (Clause): 子句級依賴 (全連接)

    每個層級都融入 SenticNet 極性信息，並通過信心門控
    動態調整知識注入強度

    v3.0 新增：動態知識門控
        - 根據 BERT 上下文動態決定是否信任 SenticNet
        - 複雜句（MAMS）：自動降低對外部知識的依賴

    Args:
        hidden_dim: 隱藏層維度
        num_heads: 注意力頭數
        num_levels: 層級數量（默認 3）
        knowledge_weight: 知識注入權重
        dropout: Dropout 比率
        use_confidence_gate: 是否使用信心門控
        use_dynamic_gate: 是否使用動態知識門控（v3.0）
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        num_levels: int = 3,
        knowledge_weight: float = 0.1,
        dropout: float = 0.1,
        use_confidence_gate: bool = True,
        use_dynamic_gate: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_levels = num_levels
        self.use_confidence_gate = use_confidence_gate
        self.use_dynamic_gate = use_dynamic_gate

        # 層級窗口大小
        self.windows = [3, 5, -1]  # -1 表示全連接

        # 鄰接矩陣緩存（避免重複計算）
        self._adj_cache = {}

        # 每個層級的 GAT（含動態門控）
        self.level_gats = nn.ModuleList([
            KnowledgeEnhancedGAT(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                knowledge_weight=knowledge_weight,
                dropout=dropout,
                use_confidence_gate=use_confidence_gate,
                use_dynamic_gate=use_dynamic_gate
            )
            for _ in range(num_levels)
        ])

        # 跨層級融合
        self.level_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 可學習的層級權重
        self.level_weights = nn.Parameter(torch.ones(num_levels) / num_levels)

    def _create_window_adj(
        self,
        seq_len: int,
        window_size: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        創建窗口鄰接矩陣（向量化版本 + 緩存）

        Args:
            seq_len: 序列長度
            window_size: 窗口大小（-1 表示全連接）
            device: 設備

        Returns:
            adj: [seq_len, seq_len]
        """
        # 檢查緩存
        cache_key = (seq_len, window_size, str(device))
        if cache_key in self._adj_cache:
            return self._adj_cache[cache_key]

        if window_size == -1:
            # 全連接
            adj = torch.ones(seq_len, seq_len, device=device)
        else:
            # 向量化：使用距離矩陣創建帶狀鄰接矩陣
            half_window = window_size // 2
            # 創建位置索引
            positions = torch.arange(seq_len, device=device)
            # 計算兩兩距離 |i - j|
            dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
            # 距離 <= half_window 的位置為 1
            adj = (dist <= half_window).float()

        # 緩存結果
        self._adj_cache[cache_key] = adj

        return adj

    def forward(
        self,
        h: torch.Tensor,
        polarities: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        coverage_mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        前向傳播

        Args:
            h: [batch, seq_len, hidden_dim]
            polarities: [batch, seq_len] SenticNet 極性
            attention_mask: [batch, seq_len]
            coverage_mask: [batch, seq_len] 知識庫覆蓋掩碼

        Returns:
            h': [batch, seq_len, hidden_dim]
            extras: 包含各層級注意力權重、門控值等的字典
        """
        batch_size, seq_len, _ = h.shape
        device = h.device

        level_outputs = []
        level_attentions = {}
        all_gate_values = []

        for level_idx, (gat, window) in enumerate(zip(self.level_gats, self.windows)):
            # 創建該層級的鄰接矩陣
            adj = self._create_window_adj(seq_len, window, device)
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, seq_len, seq_len]

            # GAT 前向傳播（含信心門控）
            h_level, attn_level, gate_values = gat(
                h, polarities, adj, attention_mask, coverage_mask
            )
            level_outputs.append(h_level)
            level_attentions[f'level_{level_idx}_attention'] = attn_level

            if gate_values is not None:
                all_gate_values.append(gate_values)

        # 加權融合
        weights = F.softmax(self.level_weights, dim=0)
        h_weighted = sum(w * h_l for w, h_l in zip(weights, level_outputs))

        # 拼接融合
        h_cat = torch.cat(level_outputs, dim=-1)  # [batch, seq_len, hidden*3]
        h_fused = self.level_fusion(h_cat)

        # 最終結合
        h_out = h_weighted + h_fused

        extras = {
            **level_attentions,
            'level_weights': weights.detach().cpu()
        }

        # 添加門控值統計（用於分析和正則化）
        if all_gate_values:
            avg_gate = torch.stack(all_gate_values, dim=0).mean(dim=0)  # [batch, seq_len]
            extras['confidence_gate_values'] = avg_gate.detach().cpu()
            extras['avg_confidence_gate'] = avg_gate.mean().item()

        return h_out, extras


class HKGAN(nn.Module):
    """
    HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network

    主模型架構：
        1. BERT with hierarchical output (Low/Mid/High layers)
        2. SenticNet knowledge enhancement（含領域過濾和信心門控）
        3. Hierarchical GAT layers
        4. Inter-aspect attention (from IARN)
        5. Classifier

    核心改進（解決 Neutral 識別問題）：
        - Domain Filtering: 針對特定領域的技術術語遮蔽通用情感極性
        - Confidence Gating: 讓模型根據上下文動態決定是否信任 SenticNet
        - Coverage Mask: 區分「中性詞」與「未登錄詞」

    統一接口，與現有訓練流程兼容

    Args:
        bert_model_name: BERT 模型名稱
        freeze_bert: 是否凍結 BERT
        hidden_dim: 隱藏層維度
        num_classes: 分類類別數
        dropout: Dropout 比率
        num_gat_heads: GAT 頭數
        num_gat_layers: GAT 層數
        knowledge_weight: 知識注入權重
        use_senticnet: 是否使用 SenticNet
        use_confidence_gate: 是否使用信心門控（默認 True）
        use_dynamic_gate: 是否使用動態知識門控 v3.0（默認 True）
        domain: 領域名稱，用於領域過濾（如 'laptops', 'restaurants'）
    """

    # BERT 層級劃分
    LOW_LAYERS = [1, 2, 3, 4]
    MID_LAYERS = [5, 6, 7, 8]
    HIGH_LAYERS = [9, 10, 11, 12]

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.3,
        num_gat_heads: int = 4,
        num_gat_layers: int = 2,
        knowledge_weight: float = 0.1,
        use_senticnet: bool = True,
        use_confidence_gate: bool = True,
        use_dynamic_gate: bool = True,
        domain: str = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_senticnet = use_senticnet
        self.knowledge_weight = knowledge_weight
        self.use_confidence_gate = use_confidence_gate
        self.use_dynamic_gate = use_dynamic_gate
        self.domain = domain

        # ========== 1. BERT Encoder ==========
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )
        bert_hidden_size = self.bert_absa.hidden_size  # 通常是 768

        # 啟用多層輸出
        self.bert_absa.bert_embedding.output_hidden_states = True
        self.bert_absa.bert_embedding.bert.config.output_hidden_states = True

        # ========== 2. SenticNet Knowledge（含領域過濾）==========
        if use_senticnet:
            from utils.senticnet_loader import get_senticnet, reset_senticnet
            # 重置單例以確保領域設置生效
            reset_senticnet()
            self.senticnet = get_senticnet()
            # 設置領域過濾
            if domain:
                self.senticnet.set_domain(domain)
            # 預建立 token_id → polarity 映射表（優化查詢速度）
            # 使用 register_buffer 註冊為 None，之後再填充
            self.register_buffer('_polarity_lookup', None)
            self.register_buffer('_coverage_lookup', None)  # 新增：覆蓋掩碼查找表
            self._bert_model_name = bert_model_name
        else:
            self.senticnet = None
            self.register_buffer('_polarity_lookup', None)
            self.register_buffer('_coverage_lookup', None)

        # 知識嵌入投影
        self.knowledge_projection = nn.Linear(1, hidden_dim)

        # ========== 3. Hierarchical Feature Fusion ==========
        # 從 4 層拼接降維到 hidden_dim
        self.low_fusion = nn.Sequential(
            nn.Linear(bert_hidden_size * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.mid_fusion = nn.Sequential(
            nn.Linear(bert_hidden_size * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.high_fusion = nn.Sequential(
            nn.Linear(bert_hidden_size * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # ========== 4. Hierarchical GAT（含信心門控 + 動態知識門控 v3.0）==========
        self.hierarchical_gat = HierarchicalGATLayer(
            hidden_dim=hidden_dim,
            num_heads=num_gat_heads,
            num_levels=3,
            knowledge_weight=knowledge_weight if use_senticnet else 0.0,
            dropout=dropout,
            use_confidence_gate=use_confidence_gate and use_senticnet,
            use_dynamic_gate=use_dynamic_gate and use_senticnet
        )

        # ========== 5. Cross-Level Fusion ==========
        # 融合 Low/Mid/High 三個層級
        self.level_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # 可學習的層級權重
        self.hierarchical_weights = nn.Parameter(torch.tensor([0.5, 1.0, 1.5]))

        # ========== 6. Inter-Aspect Attention (含情感隔離機制) ==========
        self.aspect_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_gat_heads,
            dropout=dropout,
            batch_first=True
        )

        # Relation-aware 門控
        self.relation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # ========== 情感感知隔離機制 (Sentiment-Aware Isolation) v2.3 ==========
        # 解決問題：v2.2 的隔離機制「盲目隔離」，可能阻擋有益的情感增強
        #
        # v2.2 問題：
        #   - 隔離機制只看自身特徵，不知道上下文情感
        #   - "battery drains quickly" 中，battery 可能被過度隔離，無法獲得 "quickly" 的負面增強
        #
        # v2.3 改進：情感感知的非對稱隔離
        #   - 如果 context 情感與 self 情感「一致」→ 允許流入（增強）
        #   - 如果 context 情感與 self 情感「衝突」→ 阻擋流入（隔離）
        #
        # 實現：
        #   1. 先預測每個面向的「情感傾向」（Neg/Neu/Pos 的 soft prediction）
        #   2. 計算 self 與 context 的情感一致性
        #   3. 一致性高 → 降低隔離；一致性低 → 提高隔離

        # 情感傾向預測器（輕量級，用於計算一致性）
        self.sentiment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # 3 classes: Neg, Neu, Pos
        )

        # 基礎隔離門控（保留原有設計）
        self.isolation_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

        # 可學習的基準獨立性（降低到 0.2，讓情感一致時更容易流通）
        self.base_isolation = nn.Parameter(torch.tensor(0.2))

        # 一致性調製強度（控制「一致性」對隔離的影響程度）
        self.consistency_strength = nn.Parameter(torch.tensor(0.5))

        # ========== 7. Classifier ==========
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Tokenizer（用於 SenticNet 查詢）
        self.tokenizer = None

    def _get_tokenizer(self):
        """懶加載 tokenizer"""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.bert_absa.bert_embedding.bert.config._name_or_path
            )
        return self.tokenizer

    def _extract_hierarchical_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        提取階層式 BERT 特徵

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]

        Returns:
            low_features: [batch, seq_len, hidden_dim]
            mid_features: [batch, seq_len, hidden_dim]
            high_features: [batch, seq_len, hidden_dim]
            all_hidden_states: 所有層的隱藏狀態
        """
        # BERT 前向傳播
        outputs = self.bert_absa.bert_embedding.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        all_hidden_states = outputs.hidden_states  # tuple of 13 tensors

        # 提取各層級特徵（使用完整序列而非僅 CLS）
        def extract_and_fuse(layer_indices, fusion_layer):
            features = [all_hidden_states[i] for i in layer_indices]
            concat_features = torch.cat(features, dim=-1)  # [batch, seq_len, hidden*4]
            return fusion_layer(concat_features)

        low_features = extract_and_fuse(self.LOW_LAYERS, self.low_fusion)
        mid_features = extract_and_fuse(self.MID_LAYERS, self.mid_fusion)
        high_features = extract_and_fuse(self.HIGH_LAYERS, self.high_fusion)

        return low_features, mid_features, high_features, all_hidden_states

    def _build_polarity_lookup(self):
        """
        預建立 token_id → polarity 查找表和覆蓋掩碼

        這個方法只在首次調用時執行，將整個 tokenizer 詞彙表
        映射到 SenticNet 極性值，之後的查詢只需要簡單的索引操作

        同時建立覆蓋掩碼，用於區分「中性詞」與「未登錄詞」
        """
        if self._polarity_lookup is not None:
            return

        print("[HKGAN] Building polarity lookup table (one-time operation)...")
        if self.domain:
            print(f"[HKGAN] Domain filtering enabled: {self.domain}")

        tokenizer = self._get_tokenizer()
        vocab_size = tokenizer.vocab_size

        # 建立查找表和覆蓋掩碼
        polarity_list = []
        coverage_list = []

        for token_id in range(vocab_size):
            token = tokenizer.convert_ids_to_tokens(token_id)
            # 使用新的帶覆蓋信息的查詢方法
            polarity, is_known = self.senticnet.get_polarity_with_coverage(token)
            polarity_list.append(polarity)
            coverage_list.append(1.0 if is_known else 0.0)

        # 直接賦值給已註冊的 buffer（會自動移動到正確的設備）
        device = next(self.parameters()).device

        polarity_tensor = torch.tensor(polarity_list, dtype=torch.float32)
        self._polarity_lookup = polarity_tensor.to(device)

        coverage_tensor = torch.tensor(coverage_list, dtype=torch.float32)
        self._coverage_lookup = coverage_tensor.to(device)

        # 統計
        non_zero = sum(1 for p in polarity_list if p != 0.0)
        known_count = sum(coverage_list)
        print(f"[HKGAN] Polarity lookup ready: {non_zero}/{vocab_size} tokens have non-zero polarity")
        print(f"[HKGAN] Coverage: {int(known_count)}/{vocab_size} tokens are known ({100*known_count/vocab_size:.1f}%)")

    def _get_polarities(
        self,
        input_ids: torch.Tensor
    ) -> tuple:
        """
        獲取 SenticNet 極性和覆蓋掩碼（優化版本）

        使用預建立的查找表，通過向量化索引操作獲取極性值
        時間複雜度從 O(batch × seq_len) 降到 O(1) 的 GPU 操作

        Args:
            input_ids: [batch, seq_len]

        Returns:
            polarities: [batch, seq_len] 極性值
            coverage_mask: [batch, seq_len] 覆蓋掩碼（1=已知，0=未知）
        """
        if not self.use_senticnet or self.senticnet is None:
            return None, None

        # 確保查找表已建立
        self._build_polarity_lookup()

        # 向量化查詢：直接用 input_ids 作為索引
        # _polarity_lookup: [vocab_size]
        # input_ids: [batch, seq_len]
        # 結果: [batch, seq_len]
        polarities = self._polarity_lookup[input_ids]
        coverage_mask = self._coverage_lookup[input_ids]

        return polarities, coverage_mask

    def _aspect_pooling(
        self,
        sequence_features: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Aspect 級別池化（使用 [CLS] token）

        Args:
            sequence_features: [batch, seq_len, hidden_dim]
            attention_mask: [batch, seq_len]

        Returns:
            pooled: [batch, hidden_dim]
        """
        # 使用 [CLS] token（位置 0）
        return sequence_features[:, 0, :]

    def forward(
        self,
        pair_input_ids: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        pair_token_type_ids: torch.Tensor,
        aspect_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向傳播

        Args:
            pair_input_ids: [batch, max_aspects, seq_len]
            pair_attention_mask: [batch, max_aspects, seq_len]
            pair_token_type_ids: [batch, max_aspects, seq_len] (未使用)
            aspect_mask: [batch, max_aspects] bool

        Returns:
            logits: [batch, max_aspects, num_classes]
            extras: 包含注意力權重等額外信息的字典
        """
        batch_size, max_aspects, seq_len = pair_input_ids.shape
        device = pair_input_ids.device

        # ========== 批次化處理（優化版本）==========
        # 將所有 aspects 展平成一個大 batch，只跑一次 BERT
        # [batch, max_aspects, seq_len] -> [batch * max_aspects, seq_len]
        flat_input_ids = pair_input_ids.view(-1, seq_len)
        flat_attention_mask = pair_attention_mask.view(-1, seq_len)

        # 1. 一次性提取所有 aspects 的階層式 BERT 特徵
        low_feat, mid_feat, high_feat, _ = self._extract_hierarchical_features(
            flat_input_ids, flat_attention_mask
        )
        # 結果: [batch * max_aspects, seq_len, hidden_dim]

        # 2. 獲取 SenticNet 極性和覆蓋掩碼（已優化為向量化操作）
        polarities, coverage_mask = self._get_polarities(flat_input_ids)

        # 3. 階層式 GAT 處理（批次化，含信心門控）
        low_gat, low_extras = self.hierarchical_gat(
            low_feat, polarities, flat_attention_mask, coverage_mask
        )
        mid_gat, mid_extras = self.hierarchical_gat(
            mid_feat, polarities, flat_attention_mask, coverage_mask
        )
        high_gat, high_extras = self.hierarchical_gat(
            high_feat, polarities, flat_attention_mask, coverage_mask
        )

        # 4. 跨層級融合
        weights = F.softmax(self.hierarchical_weights, dim=0)

        # 池化
        low_pooled = self._aspect_pooling(low_gat, flat_attention_mask)
        mid_pooled = self._aspect_pooling(mid_gat, flat_attention_mask)
        high_pooled = self._aspect_pooling(high_gat, flat_attention_mask)
        # 結果: [batch * max_aspects, hidden_dim]

        weighted_features = (
            weights[0] * low_pooled +
            weights[1] * mid_pooled +
            weights[2] * high_pooled
        )

        concat_features = torch.cat([low_pooled, mid_pooled, high_pooled], dim=-1)
        fused_features = self.level_fusion(concat_features)

        # 結合加權和拼接
        flat_aspect_features = weighted_features + fused_features  # [batch * max_aspects, hidden_dim]

        # 重塑回原始形狀
        aspect_features = flat_aspect_features.view(batch_size, max_aspects, -1)  # [batch, max_aspects, hidden_dim]

        # GAT extras（只保存一份用於分析）
        all_gat_extras = [{'low': low_extras, 'mid': mid_extras, 'high': high_extras}]

        # 5. Inter-Aspect Attention (含情感隔離機制)
        # 創建 attention mask
        attn_mask = ~aspect_mask  # True = mask out

        # Self-attention across aspects
        context_features, aspect_attn_weights = self.aspect_attention(
            query=aspect_features,
            key=aspect_features,
            value=aspect_features,
            key_padding_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )

        # ========== 情感感知隔離機制 (v2.3) ==========
        # 核心改進：根據「情感一致性」動態調整隔離程度
        #
        # Step 1: 預測每個面向的情感傾向（soft prediction）
        self_sentiment = F.softmax(self.sentiment_predictor(aspect_features), dim=-1)  # [batch, max_aspects, 3]
        context_sentiment = F.softmax(self.sentiment_predictor(context_features), dim=-1)  # [batch, max_aspects, 3]

        # Step 2: 計算情感一致性（使用餘弦相似度或點積）
        # 一致性高 = 兩個情感分布相似 → 應該允許信息流動
        # 一致性低 = 兩個情感分布不同 → 應該隔離（可能是污染）
        # 使用點積作為一致性度量（範圍 [0, 1]，因為 softmax 輸出都是正數且和為 1）
        sentiment_consistency = (self_sentiment * context_sentiment).sum(dim=-1, keepdim=True)  # [batch, max_aspects, 1]
        # sentiment_consistency 範圍約 [0.33, 1]（完全不同到完全一致）

        # Step 3: 基礎隔離門控（看自身特徵）
        base_isolation_scores = self.isolation_gate(aspect_features)  # [batch, max_aspects, 1]

        # Step 4: 根據一致性調整隔離程度
        # 一致性高 → 降低隔離（乘以較小的因子）
        # 一致性低 → 保持/提高隔離
        #
        # 公式: adjusted_isolation = base_isolation * (1 - consistency_strength * consistency)
        # 當 consistency=1（完全一致）且 strength=0.5 時，隔離降低 50%
        # 當 consistency=0.33（完全衝突）時，隔離幾乎不變
        consistency_factor = 1 - self.consistency_strength * sentiment_consistency
        adjusted_isolation = base_isolation_scores * consistency_factor

        # Step 5: 與基準獨立性結合
        effective_isolation = adjusted_isolation * (1 - self.base_isolation) + self.base_isolation
        # 確保範圍在 [base_isolation, 1]

        # 原有的關係門控
        gate_input = torch.cat([aspect_features, context_features], dim=-1)
        relation_gate_values = self.relation_gate(gate_input)  # [batch, max_aspects, 1]

        # 組合兩種門控（公式不變）
        self_weight = effective_isolation + (1 - effective_isolation) * relation_gate_values
        context_weight = (1 - effective_isolation) * (1 - relation_gate_values)

        gated_features = self_weight * aspect_features + context_weight * context_features

        # 保存門控值用於分析
        gate_values = relation_gate_values  # 向後兼容
        isolation_scores = base_isolation_scores  # 用於 extras

        # 6. 分類
        logits = self.classifier(gated_features)  # [batch, max_aspects, num_classes]

        # 對無效 aspect 的 logits 設為 0
        logits = logits.masked_fill(~aspect_mask.unsqueeze(-1), 0.0)

        # 準備 extras
        extras = {
            'aspect_attention_weights': aspect_attn_weights.detach().cpu(),
            'gate_values': gate_values.squeeze(-1).detach().cpu(),
            'avg_gate': gate_values[aspect_mask.unsqueeze(-1).expand_as(gate_values)].mean().item() if aspect_mask.any() else 0.0,
            'hierarchical_weights': F.softmax(self.hierarchical_weights, dim=0).detach().cpu(),
            'gat_extras': all_gat_extras[0] if all_gat_extras else {},  # 只保存第一個 aspect 的
            'mode': 'hkgan',
            # 供對比學習使用的特徵表示（分類器前的特徵）
            'features': gated_features,  # [batch, max_aspects, hidden_dim]
            # 情感感知隔離機制統計 (v2.3)
            'isolation_scores': isolation_scores.squeeze(-1).detach().cpu(),
            'effective_isolation': effective_isolation.squeeze(-1).detach().cpu(),
            'avg_isolation': effective_isolation[aspect_mask.unsqueeze(-1).expand_as(effective_isolation)].mean().item() if aspect_mask.any() else 0.0,
            'self_weight': self_weight.squeeze(-1).detach().cpu(),
            'context_weight': context_weight.squeeze(-1).detach().cpu(),
            'base_isolation': self.base_isolation.item(),
            # v2.3 新增：情感一致性統計
            'sentiment_consistency': sentiment_consistency.squeeze(-1).detach().cpu(),
            'avg_sentiment_consistency': sentiment_consistency[aspect_mask.unsqueeze(-1).expand_as(sentiment_consistency)].mean().item() if aspect_mask.any() else 0.0,
            'consistency_strength': self.consistency_strength.item()
        }

        # 添加信心門控統計（用於分析 Neutral 識別效果）
        if self.use_confidence_gate and 'avg_confidence_gate' in low_extras:
            avg_conf_gate = (
                low_extras.get('avg_confidence_gate', 0) +
                mid_extras.get('avg_confidence_gate', 0) +
                high_extras.get('avg_confidence_gate', 0)
            ) / 3.0
            extras['avg_confidence_gate'] = avg_conf_gate
            extras['confidence_gate_enabled'] = True
        else:
            extras['confidence_gate_enabled'] = False

        # 添加領域過濾統計
        if self.domain:
            extras['domain'] = self.domain
            extras['domain_filter_enabled'] = True
        else:
            extras['domain_filter_enabled'] = False

        return logits, extras


def create_hkgan_model(args, num_classes: int = 3) -> HKGAN:
    """
    創建 HKGAN 模型的工廠函數

    Args:
        args: 命令行參數
        num_classes: 分類類別數

    Returns:
        HKGAN 模型實例

    新增參數：
        - use_confidence_gate: 是否使用信心門控（默認 True）
        - use_dynamic_gate: 是否使用動態知識門控 v3.0（默認 True）
        - domain: 領域名稱，用於領域過濾（如 'laptops', 'restaurants'）
    """
    return HKGAN(
        bert_model_name=getattr(args, 'bert_model', 'bert-base-uncased'),
        freeze_bert=getattr(args, 'freeze_bert', False),
        hidden_dim=getattr(args, 'hidden_dim', 768),
        num_classes=num_classes,
        dropout=getattr(args, 'dropout', 0.3),
        num_gat_heads=getattr(args, 'gat_heads', 4),
        num_gat_layers=getattr(args, 'gat_layers', 2),
        knowledge_weight=getattr(args, 'knowledge_weight', 0.1),
        use_senticnet=getattr(args, 'use_senticnet', True),
        use_confidence_gate=getattr(args, 'use_confidence_gate', True),
        use_dynamic_gate=getattr(args, 'use_dynamic_gate', True),
        domain=getattr(args, 'domain', None)
    )
