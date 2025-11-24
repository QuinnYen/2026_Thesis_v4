"""
Hierarchical Syntax Attention (HSA) for ABSA

Method 3: 階層式語法注意力網絡

核心創新：在依賴樹結構上進行階層式信息傳播

階層設計：
    Level 1 (Token): 詞級特徵 - 直接的 BERT embeddings
    Level 2 (Phrase): 短語級特徵 - 通過語法鄰居聚合
    Level 3 (Clause): 子句級特徵 - 通過多跳傳播聚合

與其他方法對比：
    - Hierarchical BERT: 從 BERT 不同層提取特徵（模型內部階層）
    - DepGCN: 純圖卷積，無明確階層概念
    - HSA: 在語法樹上進行階層式傳播（語言學階層）

理論依據：
    1. 語言學中，句子有自然的階層結構：詞 → 短語 → 子句
    2. 依賴樹反映了這種結構
    3. 不同粒度的特徵對情感分析有不同作用：
       - 詞級：情感詞識別（"excellent", "terrible"）
       - 短語級：修飾關係（"not good", "very bad"）
       - 子句級：整體語義（"although X, but Y"）

參考文獻：
    - Hierarchical Attention Networks (Yang et al., 2016)
    - ASGCN (Zhang et al., 2019)
    - 本文創新：結合階層概念與語法結構
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from models.bert_embedding import BERTForABSA
from models.base_model import BaseModel


class SyntaxAwareAttention(nn.Module):
    """
    語法感知注意力模組

    根據語法距離計算注意力權重
    距離 aspect 越近的詞獲得越高的注意力
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(SyntaxAwareAttention, self).__init__()

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.scale = hidden_dim ** 0.5
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        syntax_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch, 1, hidden] - aspect representation
            key: [batch, seq_len, hidden]
            value: [batch, seq_len, hidden]
            syntax_mask: [batch, seq_len] - 語法距離權重 (可選)
            attention_mask: [batch, seq_len] - padding mask

        Returns:
            output: [batch, hidden]
            attention_weights: [batch, seq_len]
        """
        Q = self.query_proj(query)  # [batch, 1, hidden]
        K = self.key_proj(key)      # [batch, seq_len, hidden]
        V = self.value_proj(value)  # [batch, seq_len, hidden]

        # Attention scores
        scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale  # [batch, 1, seq_len]

        # Apply syntax distance weighting (closer = higher weight)
        if syntax_mask is not None:
            scores = scores + syntax_mask.unsqueeze(1)  # Add syntax bias

        # Apply padding mask
        if attention_mask is not None:
            # 使用 -1e4 以兼容 float16 (AMP)
            # float16 的最小值約為 -65504
            scores = scores.masked_fill(
                ~attention_mask.unsqueeze(1).bool(),
                -1e4
            )

        # Softmax with numerical stability
        attention_weights = F.softmax(scores, dim=-1)  # [batch, 1, seq_len]

        # 處理可能的 NaN (當整行都被 mask 時)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)

        attention_weights = self.dropout(attention_weights)

        # Weighted sum
        output = torch.bmm(attention_weights, V)  # [batch, 1, hidden]
        output = output.squeeze(1)  # [batch, hidden]

        return output, attention_weights.squeeze(1)


class HierarchicalSyntaxLayer(nn.Module):
    """
    單層階層式語法傳播

    從 level k 到 level k+1 的信息聚合

    修復：使用標準的 key_padding_mask 而非 attn_mask 避免 NaN
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(HierarchicalSyntaxLayer, self).__init__()

        # 鄰居聚合 - 使用標準 self-attention
        self.neighbor_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # 語法感知的權重調整
        self.syntax_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 特徵轉換
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        adjacency_matrix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden]
            adjacency_matrix: [batch, seq_len, seq_len] - 語法鄰接矩陣 (用於加權)
            attention_mask: [batch, seq_len] - padding mask

        Returns:
            output: [batch, seq_len, hidden]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # 創建 key_padding_mask (True = 需要 mask 的位置)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()  # [batch, seq_len]

        # Multi-head self-attention (不使用 attn_mask 避免 NaN)
        residual = hidden_states
        attn_output, _ = self.neighbor_attention(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )

        # 使用語法鄰接矩陣進行加權聚合 (作為補充信息)
        # 語法鄰居的特徵聚合
        syntax_context = torch.bmm(adjacency_matrix, hidden_states)  # [batch, seq_len, hidden]

        # Gate: 決定多少來自 attention，多少來自 syntax
        gate_input = torch.cat([attn_output, syntax_context], dim=-1)
        gate = self.syntax_gate(gate_input)  # [batch, seq_len, hidden]

        # 融合
        fused_output = gate * attn_output + (1 - gate) * syntax_context

        hidden_states = self.layer_norm1(residual + fused_output)

        # FFN
        residual = hidden_states
        ffn_output = self.ffn(hidden_states)
        hidden_states = self.layer_norm2(residual + ffn_output)

        return hidden_states


class HierarchicalSyntaxAttention(BaseModel):
    """
    Method 3: Hierarchical Syntax Attention (HSA)

    架構：
        BERT → Token Features → Phrase Features → Clause Features
                    ↓                ↓                ↓
              Token Attn       Phrase Attn      Clause Attn
                    ↓                ↓                ↓
                    └────────── Hierarchical Fusion ──────────┘
                                      ↓
                                  Classifier

    三層階層結構：
        1. Token Level: 直接的 BERT embeddings + aspect attention
        2. Phrase Level: 1-hop 語法鄰居聚合 + aspect attention
        3. Clause Level: 2-hop 語法鄰居聚合 + aspect attention

    創新點：
        1. 語法結構引導的階層傳播
        2. 每個層級都有 aspect-aware attention
        3. 可學習的層級權重融合
        4. 保持「階層式」主題，同時利用語法信息

    預期效果：
        - 比純 Hierarchical BERT 更好：增加了語法結構
        - 比純 DepGCN 更好：有明確的階層概念
        - Laptops: 預期 78-80% accuracy
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.3,
        num_syntax_layers: int = 2  # 控制語法傳播的跳數
    ):
        super(HierarchicalSyntaxAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_syntax_layers = num_syntax_layers

        # BERT encoder
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert_absa.hidden_size

        # Project BERT to hidden_dim if needed
        if bert_hidden_size != hidden_dim:
            self.bert_projection = nn.Linear(bert_hidden_size, hidden_dim)
        else:
            self.bert_projection = nn.Identity()

        # === Level 1: Token-level (直接 BERT features) ===
        self.token_attention = SyntaxAwareAttention(hidden_dim, dropout)
        self.token_layer_norm = nn.LayerNorm(hidden_dim)

        # === Level 2: Phrase-level (1-hop 聚合) ===
        self.phrase_layer = HierarchicalSyntaxLayer(hidden_dim, dropout)
        self.phrase_attention = SyntaxAwareAttention(hidden_dim, dropout)
        self.phrase_layer_norm = nn.LayerNorm(hidden_dim)

        # === Level 3: Clause-level (2-hop 聚合) ===
        self.clause_layer = HierarchicalSyntaxLayer(hidden_dim, dropout)
        self.clause_attention = SyntaxAwareAttention(hidden_dim, dropout)
        self.clause_layer_norm = nn.LayerNorm(hidden_dim)

        # === Hierarchical Fusion ===
        # 可學習的層級權重 (類似 Layer-wise Attention)
        self.level_weights = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # 最終融合層
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # === Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        self.dropout = nn.Dropout(dropout)

    def _create_syntax_adjacency(
        self,
        attention_mask: torch.Tensor,
        seq_len: int,
        device: torch.device,
        hop: int = 1
    ) -> torch.Tensor:
        """
        創建基於語法距離的鄰接矩陣

        當沒有外部依賴解析時，使用滑動窗口近似：
        - hop=1: 直接鄰居 (window=3)
        - hop=2: 二跳鄰居 (window=5)

        Args:
            attention_mask: [batch, seq_len]
            seq_len: 序列長度
            device: 設備
            hop: 跳數

        Returns:
            adjacency: [batch, seq_len, seq_len]
        """
        batch_size = attention_mask.size(0)
        window_size = 1 + hop * 2  # hop=1 -> window=3, hop=2 -> window=5

        # 創建基礎鄰接矩陣
        adj = torch.zeros(seq_len, seq_len, device=device)
        for i in range(seq_len):
            start = max(0, i - window_size // 2)
            end = min(seq_len, i + window_size // 2 + 1)
            adj[i, start:end] = 1.0

        # 正規化
        degree = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        adj = adj / degree

        # 擴展到 batch
        adj = adj.unsqueeze(0).expand(batch_size, -1, -1).clone()

        # 應用 attention mask
        mask = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)
        adj = adj * mask.float()

        # 確保每行至少有自連接 (避免全零行導致 NaN)
        # 添加自連接
        eye = torch.eye(seq_len, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        adj = adj + eye * 0.1  # 小權重的自連接

        # 重新正規化
        row_sum = adj.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        adj = adj / row_sum

        return adj

    def forward(
        self,
        pair_input_ids: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        pair_token_type_ids: torch.Tensor,
        aspect_mask: torch.Tensor,
        dependency_matrices: Optional[torch.Tensor] = None
    ):
        """
        前向傳播

        Args:
            pair_input_ids: [batch, max_aspects, seq_len]
            pair_attention_mask: [batch, max_aspects, seq_len]
            pair_token_type_ids: [batch, max_aspects, seq_len]
            aspect_mask: [batch, max_aspects]
            dependency_matrices: [batch, max_aspects, seq_len, seq_len] (可選)

        Returns:
            logits: [batch, max_aspects, num_classes]
            extras: dict with attention weights and level weights
        """
        batch_size, max_aspects, seq_len = pair_input_ids.shape
        device = pair_input_ids.device

        logits_list = []
        all_level_outputs = {'token': [], 'phrase': [], 'clause': []}

        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                # 獲取當前 aspect 的輸入
                input_ids = pair_input_ids[:, i, :]
                attention_mask = pair_attention_mask[:, i, :]
                token_type_ids = pair_token_type_ids[:, i, :]

                # BERT encoding
                bert_output = self.bert_absa.bert_embedding(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                if isinstance(bert_output, tuple):
                    sequence_output = bert_output[0]
                else:
                    sequence_output = bert_output

                # Project to hidden_dim
                hidden_states = self.bert_projection(sequence_output)
                hidden_states = self.dropout(hidden_states)
                # hidden_states: [batch, seq_len, hidden_dim]

                # 獲取 aspect representation (使用 token_type_ids=1 的位置)
                aspect_token_mask = token_type_ids.bool()
                # 簡化：使用 [CLS] 作為 query
                aspect_repr = hidden_states[:, 0:1, :]  # [batch, 1, hidden]

                # 創建語法鄰接矩陣
                if dependency_matrices is not None:
                    adj_1hop = dependency_matrices[:, i, :, :]
                    # 計算 2-hop 鄰接
                    adj_2hop = torch.bmm(adj_1hop, adj_1hop)
                    adj_2hop = (adj_2hop > 0).float()
                else:
                    adj_1hop = self._create_syntax_adjacency(attention_mask, seq_len, device, hop=1)
                    adj_2hop = self._create_syntax_adjacency(attention_mask, seq_len, device, hop=2)

                # === Level 1: Token-level ===
                token_output, token_attn = self.token_attention(
                    query=aspect_repr,
                    key=hidden_states,
                    value=hidden_states,
                    attention_mask=attention_mask
                )
                token_output = self.token_layer_norm(token_output)
                # token_output: [batch, hidden]

                # === Level 2: Phrase-level (1-hop propagation) ===
                phrase_states = self.phrase_layer(hidden_states, adj_1hop, attention_mask)
                phrase_output, phrase_attn = self.phrase_attention(
                    query=aspect_repr,
                    key=phrase_states,
                    value=phrase_states,
                    attention_mask=attention_mask
                )
                phrase_output = self.phrase_layer_norm(phrase_output)
                # phrase_output: [batch, hidden]

                # === Level 3: Clause-level (2-hop propagation) ===
                clause_states = self.clause_layer(phrase_states, adj_2hop, attention_mask)
                clause_output, clause_attn = self.clause_attention(
                    query=aspect_repr,
                    key=clause_states,
                    value=clause_states,
                    attention_mask=attention_mask
                )
                clause_output = self.clause_layer_norm(clause_output)
                # clause_output: [batch, hidden]

                # === Hierarchical Fusion ===
                # 計算層級權重
                level_weights = F.softmax(self.level_weights, dim=0)

                # 加權融合
                weighted_token = level_weights[0] * token_output
                weighted_phrase = level_weights[1] * phrase_output
                weighted_clause = level_weights[2] * clause_output

                # 拼接所有層級
                concat_features = torch.cat([
                    weighted_token, weighted_phrase, weighted_clause
                ], dim=-1)  # [batch, hidden*3]

                # 融合
                fused_output = self.fusion(concat_features)  # [batch, hidden]

                # === Classification ===
                logit = self.classifier(fused_output)  # [batch, num_classes]
                logits_list.append(logit)

                # 記錄各層輸出用於分析
                all_level_outputs['token'].append(token_attn)
                all_level_outputs['phrase'].append(phrase_attn)
                all_level_outputs['clause'].append(clause_attn)
            else:
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=device)
                )
                all_level_outputs['token'].append(torch.zeros(batch_size, seq_len, device=device))
                all_level_outputs['phrase'].append(torch.zeros(batch_size, seq_len, device=device))
                all_level_outputs['clause'].append(torch.zeros(batch_size, seq_len, device=device))

        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]

        # 計算最終的層級權重
        final_level_weights = F.softmax(self.level_weights, dim=0)

        extras = {
            'level_weights': final_level_weights.detach().cpu().numpy(),
            'token_attention': torch.stack(all_level_outputs['token'], dim=1),
            'phrase_attention': torch.stack(all_level_outputs['phrase'], dim=1),
            'clause_attention': torch.stack(all_level_outputs['clause'], dim=1),
        }

        return logits, extras


def create_hsa_model(
    bert_model_name: str = 'bert-base-uncased',
    freeze_bert: bool = False,
    hidden_dim: int = 768,
    num_classes: int = 3,
    dropout: float = 0.3,
    num_syntax_layers: int = 2,
    **kwargs
) -> HierarchicalSyntaxAttention:
    """
    Factory function: 創建 HSA 模型
    """
    return HierarchicalSyntaxAttention(
        bert_model_name=bert_model_name,
        freeze_bert=freeze_bert,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
        num_syntax_layers=num_syntax_layers
    )


if __name__ == '__main__':
    print("Testing Hierarchical Syntax Attention model...")

    # Create model
    model = HierarchicalSyntaxAttention(
        bert_model_name='bert-base-uncased',
        hidden_dim=768,
        num_classes=3,
        dropout=0.3
    )

    # Mock inputs
    batch_size = 2
    max_aspects = 3
    seq_len = 64

    pair_input_ids = torch.randint(0, 1000, (batch_size, max_aspects, seq_len))
    pair_attention_mask = torch.ones(batch_size, max_aspects, seq_len).long()
    pair_token_type_ids = torch.zeros(batch_size, max_aspects, seq_len).long()
    aspect_mask = torch.tensor([[True, True, False], [True, False, False]])

    # Forward pass
    logits, extras = model(
        pair_input_ids,
        pair_attention_mask,
        pair_token_type_ids,
        aspect_mask
    )

    print(f"\nInput shapes:")
    print(f"  pair_input_ids: {pair_input_ids.shape}")
    print(f"  aspect_mask: {aspect_mask.shape}")

    print(f"\nOutput shapes:")
    print(f"  logits: {logits.shape}")

    print(f"\nHierarchical Level Weights:")
    print(f"  Token:  {extras['level_weights'][0]:.4f}")
    print(f"  Phrase: {extras['level_weights'][1]:.4f}")
    print(f"  Clause: {extras['level_weights'][2]:.4f}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")

    print("\nHSA model test complete!")
