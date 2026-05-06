"""
HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network

架構組成：
    1. 階層式 BERT 特徵提取（Low / Mid / High 三段層）
    2. 動態知識門控（Dynamic Knowledge Gating）
       Gate = Sigmoid(Linear([h_BERT, h_SenticNet]))
       h_out = (1 - Gate) * h_BERT + Gate * h_SenticNet
    3. 信心門控（Confidence Gate）：調整邊級別知識注入強度
    4. 階層式 GAT（三級窗口：token / phrase / clause）
    5. 跨面向注意力（Inter-Aspect Attention）
       含情感感知隔離（Sentiment-Aware Isolation）：
       依情感一致性動態決定跨面向資訊流通程度

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
    純 PyTorch 實現的圖注意力層

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
    信心門控（Confidence Gate）：根據 BERT 上下文動態決定是否信任 SenticNet 知識

    gate ≈ 1：情感表達（信任 SenticNet）
    gate ≈ 0：客觀陳述（忽略 SenticNet，如技術描述中的 "high resolution"）

    公式：
        gate = σ((W₂ · ReLU(W₁ · h) + bias) / temperature)
        effective_polarity = gate * senticnet_polarity

    Args:
        hidden_dim: BERT 隱藏層維度
        dropout: Dropout 比率
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

        # 正值初始化使默認傾向於信任 SenticNet
        self.gate_bias = nn.Parameter(torch.tensor(1.0))
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
        gate_logits = self.gate_network(hidden_states).squeeze(-1)  # [batch, seq_len]

        # clamp 避免除以 0 或負數
        temp = torch.clamp(self.temperature, min=0.1)
        gate_values = torch.sigmoid((gate_logits + self.gate_bias) / temp)

        gated_polarities = gate_values * polarities

        return gated_polarities, gate_values


class DynamicKnowledgeGate(nn.Module):
    """
    動態知識門控（Dynamic Knowledge Gating）

    公式：
        Gate = Sigmoid(Linear([h_BERT, h_SenticNet]))
        h_out = (1 - Gate) * h_BERT + Gate * h_SenticNet

    Args:
        hidden_dim: BERT 隱藏層維度
        dropout: Dropout 比率
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.polarity_embed = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, hidden_dim)
        )

        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

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
        pol_input = polarities.unsqueeze(-1)  # [batch, seq_len, 1]
        polarity_embed = self.polarity_embed(pol_input)  # [batch, seq_len, hidden_dim]

        gate_input = torch.cat([hidden_states, polarity_embed], dim=-1)
        gate_values = self.gate_network(gate_input).squeeze(-1)  # [batch, seq_len]

        if coverage_mask is not None:
            gate_values = gate_values * coverage_mask

        gate_expanded = gate_values.unsqueeze(-1)
        fused_features = (1 - gate_expanded) * hidden_states + gate_expanded * polarity_embed
        enhanced_features = self.output_proj(fused_features) + hidden_states

        return enhanced_features, gate_values


class KnowledgeEnhancedGAT(nn.Module):
    """
    知識增強的圖注意力層

    節點級：DynamicKnowledgeGate 軟性融合 BERT 與 SenticNet 特徵
    邊級：ConfidenceGate 調整知識偏置強度，影響鄰接矩陣權重：
        e'_ij = e_ij + knowledge_bias_ij（極性已門控）

    Args:
        hidden_dim: 隱藏層維度
        num_heads: 注意力頭數
        knowledge_weight: 知識注入基準權重 λ
        dropout: Dropout 比率
        use_confidence_gate: 是否使用信心門控
        use_dynamic_gate: 是否使用動態知識門控
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

        self.gat = MultiHeadGAT(
            in_features=hidden_dim,
            out_features=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            concat=True
        )

        self.dynamic_gate = DynamicKnowledgeGate(hidden_dim, dropout) if use_dynamic_gate else None
        self.confidence_gate = ConfidenceGate(hidden_dim, dropout) if use_confidence_gate else None

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
        gate_values = None

        if self.use_dynamic_gate and self.dynamic_gate is not None and polarities is not None:
            h, gate_values = self.dynamic_gate(h, polarities, coverage_mask)

        if polarities is not None and self.knowledge_weight > 0:
            if self.use_confidence_gate and self.confidence_gate is not None:
                gated_polarities, conf_gate_values = self.confidence_gate(h, polarities, coverage_mask)
                adj = self._compute_knowledge_adj(gated_polarities, adj, attention_mask, conf_gate_values)
                if gate_values is None:
                    gate_values = conf_gate_values
            else:
                adj = self._compute_knowledge_adj(polarities, adj, attention_mask)

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
        計算知識增強的鄰接矩陣

        極性已在 ConfidenceGate 中門控，此處直接計算邊級知識偏置。

        Args:
            polarities: [batch, seq_len] 已門控的極性值
            adj: [batch, seq_len, seq_len] 或 None（None 視為全連接）
            attention_mask: [batch, seq_len]（未使用，保留介面一致性）
            gate_values: 未使用（保留介面一致性）

        Returns:
            enhanced_adj: [batch, seq_len, seq_len]
        """
        batch_size, seq_len = polarities.shape

        pol_i = polarities.unsqueeze(2)  # [batch, seq_len, 1]
        pol_j = polarities.unsqueeze(1)  # [batch, 1, seq_len]

        pol_pairs = torch.stack([
            pol_i.expand(-1, -1, seq_len),
            pol_j.expand(-1, seq_len, -1)
        ], dim=-1)  # [batch, seq_len, seq_len, 2]

        knowledge_bias = self.knowledge_proj(pol_pairs).squeeze(-1) * self.knowledge_weight

        if adj is None:
            enhanced_adj = torch.ones(batch_size, seq_len, seq_len, device=polarities.device)
        else:
            enhanced_adj = adj.clone()

        enhanced_adj = F.relu(enhanced_adj + knowledge_bias)

        return enhanced_adj


class HierarchicalGATLayer(nn.Module):
    """
    階層式圖注意力層：三級窗口圖傳播，從局部到全局

    Level 0 (Token):  window=3
    Level 1 (Phrase): window=5
    Level 2 (Clause): 全連接

    Args:
        hidden_dim: 隱藏層維度
        num_heads: 注意力頭數
        num_levels: 層級數量（默認 3）
        knowledge_weight: 知識注入權重
        dropout: Dropout 比率
        use_confidence_gate: 是否使用信心門控
        use_dynamic_gate: 是否使用動態知識門控
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

        self.windows = [3, 5, -1]  # -1 表示全連接
        self._adj_cache = {}

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

        self.level_fusion = nn.Sequential(
            nn.Linear(hidden_dim * num_levels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

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
        cache_key = (seq_len, window_size, str(device))
        if cache_key in self._adj_cache:
            return self._adj_cache[cache_key]

        if window_size == -1:
            adj = torch.ones(seq_len, seq_len, device=device)
        else:
            half_window = window_size // 2
            positions = torch.arange(seq_len, device=device)
            dist = torch.abs(positions.unsqueeze(0) - positions.unsqueeze(1))
            adj = (dist <= half_window).float()

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
            adj = self._create_window_adj(seq_len, window, device)
            adj = adj.unsqueeze(0).expand(batch_size, -1, -1)

            h_level, attn_level, gate_values = gat(h, polarities, adj, attention_mask, coverage_mask)
            level_outputs.append(h_level)
            level_attentions[f'level_{level_idx}_attention'] = attn_level

            if gate_values is not None:
                all_gate_values.append(gate_values)

        weights = F.softmax(self.level_weights, dim=0)
        h_weighted = sum(w * h_l for w, h_l in zip(weights, level_outputs))
        h_cat = torch.cat(level_outputs, dim=-1)
        h_out = h_weighted + self.level_fusion(h_cat)

        extras = {**level_attentions, 'level_weights': weights.detach().cpu()}

        if all_gate_values:
            avg_gate = torch.stack(all_gate_values, dim=0).mean(dim=0)
            extras['confidence_gate_values'] = avg_gate.detach().cpu()
            extras['avg_confidence_gate'] = avg_gate.mean().item()
            extras['gate_values_grad'] = avg_gate.mean(dim=-1)  # 供 gate penalty 使用，保留梯度

        return h_out, extras


class HKGAN(nn.Module):
    """
    HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network

    Args:
        bert_model_name: BERT 模型名稱
        freeze_bert: 是否凍結 BERT
        hidden_dim: 隱藏層維度
        num_classes: 分類類別數
        dropout: Dropout 比率
        num_gat_heads: GAT 頭數
        num_gat_layers: GAT 層數（保留參數，架構固定為 3 級）
        knowledge_weight: 知識注入基準權重
        use_senticnet: 是否使用 SenticNet
        use_confidence_gate: 是否使用信心門控
        use_dynamic_gate: 是否使用動態知識門控
        use_inter_aspect: 是否使用跨面向注意力模組
        use_hierarchical_features: 是否使用三段式 BERT 層特徵（消融用）
        domain: 領域名稱，用於 SenticNet 領域過濾
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
        use_inter_aspect: bool = True,
        use_hierarchical_features: bool = True,
        domain: str = None
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_senticnet = use_senticnet
        self.knowledge_weight = knowledge_weight
        self.use_confidence_gate = use_confidence_gate
        self.use_dynamic_gate = use_dynamic_gate
        self.use_inter_aspect = use_inter_aspect
        self.use_hierarchical_features = use_hierarchical_features
        self.domain = domain

        # ========== 1. BERT Encoder ==========
        self.bert_absa = BERTForABSA(model_name=bert_model_name, freeze_bert=freeze_bert)
        bert_hidden_size = self.bert_absa.hidden_size

        self.bert_absa.bert_embedding.output_hidden_states = True
        self.bert_absa.bert_embedding.bert.config.output_hidden_states = True

        # ========== 2. SenticNet Knowledge ==========
        if use_senticnet:
            from datasets.loader_knowledge import get_senticnet, reset_senticnet
            reset_senticnet()
            self.senticnet = get_senticnet()
            if domain:
                self.senticnet.set_domain(domain)
            self._bert_model_name = bert_model_name
        else:
            self.senticnet = None

        # token_id → polarity / coverage 查找表（懶建立）
        self.register_buffer('_polarity_lookup', None)
        self.register_buffer('_coverage_lookup', None)

        self.knowledge_projection = nn.Linear(1, hidden_dim)

        # ========== 3. Hierarchical Feature Fusion ==========
        # 每段 4 層 concat → hidden_dim
        _make_fusion = lambda: nn.Sequential(
            nn.Linear(bert_hidden_size * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.low_fusion = _make_fusion()
        self.mid_fusion = _make_fusion()
        self.high_fusion = _make_fusion()

        # ========== 4. Hierarchical GAT ==========
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
        self.level_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.hierarchical_weights = nn.Parameter(torch.tensor([0.5, 1.0, 1.5]))

        # ========== 6. Inter-Aspect Attention + Sentiment-Aware Isolation ==========
        self.aspect_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_gat_heads,
            dropout=dropout,
            batch_first=True
        )

        self.relation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 情感感知隔離：依情感一致性動態調整跨面向資訊流
        self.sentiment_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3),  # Neg / Neu / Pos
        )
        self.isolation_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        self.base_isolation = nn.Parameter(torch.tensor(0.2))
        self.consistency_strength = nn.Parameter(torch.tensor(0.5))

        # ========== 7. Classifier ==========
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

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
            if is_known:
                coverage_list.append(1.0)
            else:
                coverage_list.append(0.0)

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

        # 展平 aspects → 單次 BERT 前向傳播
        flat_input_ids = pair_input_ids.view(-1, seq_len)
        flat_attention_mask = pair_attention_mask.view(-1, seq_len)

        # 1. BERT 階層式特徵提取
        if self.use_hierarchical_features:
            low_feat, mid_feat, high_feat, _ = self._extract_hierarchical_features(
                flat_input_ids, flat_attention_mask
            )
        else:
            # 消融：僅用最後一層
            _, _, high_feat, _ = self._extract_hierarchical_features(flat_input_ids, flat_attention_mask)
            low_feat = mid_feat = high_feat

        # 2. SenticNet 極性與覆蓋掩碼
        polarities, coverage_mask = self._get_polarities(flat_input_ids)

        # 3. 階層式 GAT
        low_gat, low_extras = self.hierarchical_gat(low_feat, polarities, flat_attention_mask, coverage_mask)
        mid_gat, mid_extras = self.hierarchical_gat(mid_feat, polarities, flat_attention_mask, coverage_mask)
        high_gat, high_extras = self.hierarchical_gat(high_feat, polarities, flat_attention_mask, coverage_mask)

        # 4. 跨層級融合 → aspect pooling
        weights = F.softmax(self.hierarchical_weights, dim=0)
        low_pooled = self._aspect_pooling(low_gat, flat_attention_mask)
        mid_pooled = self._aspect_pooling(mid_gat, flat_attention_mask)
        high_pooled = self._aspect_pooling(high_gat, flat_attention_mask)

        if self.use_hierarchical_features:
            weighted = weights[0] * low_pooled + weights[1] * mid_pooled + weights[2] * high_pooled
            fused = self.level_fusion(torch.cat([low_pooled, mid_pooled, high_pooled], dim=-1))
            flat_aspect_features = weighted + fused
        else:
            flat_aspect_features = high_pooled

        aspect_features = flat_aspect_features.view(batch_size, max_aspects, -1)

        # 5. Inter-Aspect Attention + 情感感知隔離
        if self.use_inter_aspect:
            context_features, aspect_attn_weights = self.aspect_attention(
                query=aspect_features,
                key=aspect_features,
                value=aspect_features,
                key_padding_mask=~aspect_mask,
                need_weights=True,
                average_attn_weights=True
            )

            self_sentiment = F.softmax(self.sentiment_predictor(aspect_features), dim=-1)
            context_sentiment = F.softmax(self.sentiment_predictor(context_features), dim=-1)
            sentiment_consistency = (self_sentiment * context_sentiment).sum(dim=-1, keepdim=True)

            base_isolation_scores = self.isolation_gate(aspect_features)
            consistency_factor = 1 - self.consistency_strength * sentiment_consistency
            adjusted_isolation = base_isolation_scores * consistency_factor
            effective_isolation = adjusted_isolation * (1 - self.base_isolation) + self.base_isolation

            relation_gate_values = self.relation_gate(torch.cat([aspect_features, context_features], dim=-1))
            self_weight = effective_isolation + (1 - effective_isolation) * relation_gate_values
            context_weight = (1 - effective_isolation) * (1 - relation_gate_values)

            gated_features = self_weight * aspect_features + context_weight * context_features
            gate_values = relation_gate_values
            isolation_scores = base_isolation_scores
        else:
            gated_features = aspect_features
            aspect_attn_weights = torch.zeros(batch_size, max_aspects, max_aspects, device=device)
            gate_values = torch.ones(batch_size, max_aspects, 1, device=device)
            isolation_scores = torch.ones(batch_size, max_aspects, 1, device=device)
            effective_isolation = torch.ones(batch_size, max_aspects, 1, device=device)
            self_weight = torch.ones(batch_size, max_aspects, 1, device=device)
            context_weight = torch.zeros(batch_size, max_aspects, 1, device=device)
            sentiment_consistency = torch.ones(batch_size, max_aspects, 1, device=device)

        # 6. 分類
        logits = self.classifier(gated_features)
        logits = logits.masked_fill(~aspect_mask.unsqueeze(-1), 0.0)

        # 知識門控梯度（供 Gate Penalty 使用，保留梯度）
        knowledge_gate_grad = None
        for gat_ext in [low_extras, mid_extras, high_extras]:
            if 'gate_values_grad' in gat_ext:
                g = gat_ext['gate_values_grad']
                knowledge_gate_grad = g if knowledge_gate_grad is None else knowledge_gate_grad + g
        if knowledge_gate_grad is not None:
            knowledge_gate_grad = knowledge_gate_grad / 3.0

        extras = {
            'aspect_attention_weights': aspect_attn_weights.detach().cpu(),
            'gate_values': gate_values.squeeze(-1).detach().cpu(),
            'avg_gate': gate_values[aspect_mask.unsqueeze(-1).expand_as(gate_values)].mean().item() if aspect_mask.any() else 0.0,
            'hierarchical_weights': F.softmax(self.hierarchical_weights, dim=0).detach().cpu(),
            'gat_extras': {'low': low_extras, 'mid': mid_extras, 'high': high_extras},
            'mode': 'hkgan',
            'features': gated_features,
            'isolation_scores': isolation_scores.squeeze(-1).detach().cpu(),
            'effective_isolation': effective_isolation.squeeze(-1).detach().cpu(),
            'avg_isolation': effective_isolation[aspect_mask.unsqueeze(-1).expand_as(effective_isolation)].mean().item() if aspect_mask.any() else 0.0,
            'self_weight': self_weight.squeeze(-1).detach().cpu(),
            'context_weight': context_weight.squeeze(-1).detach().cpu(),
            'base_isolation': self.base_isolation.item(),
            'sentiment_consistency': sentiment_consistency.squeeze(-1).detach().cpu(),
            'avg_sentiment_consistency': sentiment_consistency[aspect_mask.unsqueeze(-1).expand_as(sentiment_consistency)].mean().item() if aspect_mask.any() else 0.0,
            'consistency_strength': self.consistency_strength.item()
        }

        if knowledge_gate_grad is not None:
            extras['gate_values_grad'] = knowledge_gate_grad

        if self.use_confidence_gate and 'avg_confidence_gate' in low_extras:
            extras['avg_confidence_gate'] = (
                low_extras.get('avg_confidence_gate', 0) +
                mid_extras.get('avg_confidence_gate', 0) +
                high_extras.get('avg_confidence_gate', 0)
            ) / 3.0
            extras['confidence_gate_enabled'] = True
        else:
            extras['confidence_gate_enabled'] = False

        extras['domain_filter_enabled'] = bool(self.domain)
        if self.domain:
            extras['domain'] = self.domain

        return logits, extras


def create_hkgan_model(args, num_classes: int = 3) -> HKGAN:
    """從 args 命名空間建立 HKGAN 實例（消融實驗入口）。"""
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
        use_inter_aspect=getattr(args, 'use_inter_aspect', True),
        use_hierarchical_features=getattr(args, 'use_hierarchical_features', True),
        domain=getattr(args, 'domain', None)
    )
