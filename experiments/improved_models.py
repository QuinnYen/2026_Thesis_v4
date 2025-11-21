"""
Improved Models for ABSA

這個檔案包含我們提出的改進模型：
1. HierarchicalBERT: 階層式BERT特徵提取（方法 1）
   - 從 BERT 不同層提取 Low/Mid/High 層級特徵
   - 固定拼接方式組合

2. HierarchicalBERT_LayerAttn (HBL): Hierarchical BERT + Layer-wise Attention（方法 2，主要貢獻）
   - 基於 UDify (Kondratyuk & Straka, EMNLP 2019) 的 Layer-wise Attention
   - 動態學習層級權重，替代固定拼接

用途：
- 展示階層特徵提取的有效性
- 證明 Layer-wise Attention 的改進效果
- 與 BERT-CLS baseline 對比
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bert_embedding import BERTForABSA
from models.base_model import BaseModel


class HierarchicalBERT(BaseModel):
    """
    方法 1: Hierarchical BERT (階層式BERT)

    架構:
        BERT Layers → Extract Multi-Level Features → Hierarchical Fusion → Classifier

    階層設計:
        - Low-level (layers 1-2/1-4): Syntactic features (詞法特徵)
        - Mid-level (layers 3-4/5-8): Semantic features (語義特徵)
        - High-level (layers 5-6/9-12): Task-specific features (任務特徵)

    特點:
        - 利用BERT不同層的階層特性
        - 固定concatenation組合三個層級
        - 簡單、有效、可解釋

    與 Baseline 對比:
        - Baseline (BERT-CLS): 只用最後一層的 [CLS] token
        - 本方法: 利用多層級特徵，預期提升 2-3% F1

    與 HBL 對比:
        - 本方法: 固定拼接 (concatenation)
        - HBL: 動態學習權重 (Layer-wise Attention)
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super(HierarchicalBERT, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # BERT with hierarchical output
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        # Enable output of all hidden states
        self.bert_absa.bert_embedding.output_hidden_states = True
        self.bert_absa.bert_embedding.bert.config.output_hidden_states = True

        bert_hidden_size = self.bert_absa.hidden_size

        # Hierarchical fusion layers
        # DistilBERT has 6 layers, BERT has 12 layers
        is_distilbert = 'distilbert' in bert_model_name.lower()

        if is_distilbert:
            # DistilBERT: 6 layers
            # Low: 1-2, Mid: 3-4, High: 5-6
            self.low_layers = [1, 2]
            self.mid_layers = [3, 4]
            self.high_layers = [5, 6]
        else:
            # BERT/RoBERTa: 12 layers
            # Low: 1-4, Mid: 5-8, High: 9-12
            self.low_layers = [1, 2, 3, 4]
            self.mid_layers = [5, 6, 7, 8]
            self.high_layers = [9, 10, 11, 12]

        # Fusion layers for each level
        self.low_fusion = nn.Sequential(
            nn.Linear(bert_hidden_size * len(self.low_layers), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mid_fusion = nn.Sequential(
            nn.Linear(bert_hidden_size * len(self.mid_layers), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.high_fusion = nn.Sequential(
            nn.Linear(bert_hidden_size * len(self.high_layers), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classifier (concatenate all 3 levels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 levels concatenated
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        參數:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            aspect_input_ids: [batch, max_aspects, aspect_len]
            aspect_attention_mask: [batch, max_aspects, aspect_len]
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
            None: (無額外資訊)
        """
        batch_size, max_aspects, aspect_len = aspect_input_ids.shape

        logits_list = []

        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                # Concatenate text and aspect: [CLS] text [SEP] aspect [SEP]
                combined_ids = torch.cat([
                    text_input_ids,
                    aspect_input_ids[:, i, :]
                ], dim=1)

                combined_mask = torch.cat([
                    text_attention_mask,
                    aspect_attention_mask[:, i, :]
                ], dim=1)

                # Get hierarchical features from BERT
                outputs = self.bert_absa.bert_embedding.bert(
                    input_ids=combined_ids,
                    attention_mask=combined_mask,
                    return_dict=True
                )

                all_hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

                # Extract [CLS] token from different layers
                # all_hidden_states[0] is embedding layer, so layers start from index 1
                low_features = []
                for layer_idx in self.low_layers:
                    low_features.append(all_hidden_states[layer_idx][:, 0, :])  # CLS token
                low_features = torch.cat(low_features, dim=-1)  # [batch, hidden*num_low]

                mid_features = []
                for layer_idx in self.mid_layers:
                    mid_features.append(all_hidden_states[layer_idx][:, 0, :])
                mid_features = torch.cat(mid_features, dim=-1)

                high_features = []
                for layer_idx in self.high_layers:
                    high_features.append(all_hidden_states[layer_idx][:, 0, :])
                high_features = torch.cat(high_features, dim=-1)

                # Hierarchical fusion
                low_fused = self.low_fusion(low_features)   # [batch, hidden_dim]
                mid_fused = self.mid_fusion(mid_features)   # [batch, hidden_dim]
                high_fused = self.high_fusion(high_features) # [batch, hidden_dim]

                # Concatenate all levels (固定拼接)
                hierarchical_repr = torch.cat([low_fused, mid_fused, high_fused], dim=-1)  # [batch, hidden*3]

                # Classification
                logit = self.classifier(hierarchical_repr)  # [batch, num_classes]
                logits_list.append(logit)
            else:
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=text_input_ids.device)
                )

        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]

        return logits, None  # None: 無額外資訊


class HierarchicalBERT_LayerAttn(BaseModel):
    """
    Hierarchical BERT + Layer-wise Attention

    基於 UDify (Kondratyuk & Straka, EMNLP 2019) 的 Layer-wise Attention

    架構:
        BERT Layers → Extract Multi-Level Features → Hierarchical Fusion
        → Layer-wise Attention (可學習權重) → Classifier

    改進點:
        1. 為 Low/Mid/High 三個層級分別學習動態權重
        2. 替代原先的固定 concatenation (Baseline_HierarchicalBERT)
        3. 讓模型自動學習哪個層級最重要
        4. 減少分類器參數量 (2304 → 768 維)

    公式:
        α = [w_low, w_mid, w_high]  # 可學習參數
        β = softmax(α)               # 歸一化權重
        h = Σ(β_i * h_i)             # 加權組合

    預期提升:
        - MAMS: +2~3% F1
        - Restaurants: +2~3% F1
        - 改善過擬合問題
        - 提供可解釋性 (可視化權重分布)

    參考文獻:
        Kondratyuk, D., & Straka, M. (2019). 75 Languages, 1 Model: Parsing
        Universal Dependencies Universally. In Proceedings of EMNLP-IJCNLP
        2019 (pp. 2779-2795).
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super(HierarchicalBERT_LayerAttn, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # BERT with hierarchical output
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        # Enable output of all hidden states
        self.bert_absa.bert_embedding.output_hidden_states = True
        self.bert_absa.bert_embedding.bert.config.output_hidden_states = True

        bert_hidden_size = self.bert_absa.hidden_size

        # === Multi-level feature extractors ===
        # Low-level: 淺層特徵 (詞彙、句法)
        self.low_projection = nn.Linear(bert_hidden_size, hidden_dim)

        # Mid-level: 中層特徵 (短語、局部語義)
        self.mid_projection = nn.Linear(bert_hidden_size, hidden_dim)

        # High-level: 深層特徵 (全局語義、上下文)
        self.high_projection = nn.Linear(bert_hidden_size, hidden_dim)

        # === Aspect-aware fusion (每個層級) ===
        self.low_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.mid_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        self.high_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # ⭐ Layer-wise Attention Weights (可學習參數)
        # 初始化為均等權重 [1.0, 1.0, 1.0]
        self.layer_weights = nn.Parameter(torch.ones(3))

        # === Final classifier ===
        # 注意：輸入維度改為 hidden_dim (不再是 hidden_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        aspect_input_ids,
        aspect_attention_mask,
        aspect_positions=None,
        labels=None
    ):
        """
        前向傳播

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            aspect_input_ids: [batch, max_aspects, aspect_len]
            aspect_attention_mask: [batch, max_aspects, aspect_len]
            aspect_positions: [batch, max_aspects, 2] (start, end) 位置
            labels: [batch, max_aspects]

        Returns:
            logits: [batch, max_aspects, num_classes]
            extras: dict with 'layer_attention' weights
        """
        batch_size = input_ids.size(0)
        max_aspects = aspect_input_ids.size(1)

        # === 1. BERT encoding ===
        # BERTEmbedding.forward() returns (sequence_output, all_hidden_states) when output_hidden_states=True
        bert_forward_result = self.bert_absa.bert_embedding(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Unpack the result
        if isinstance(bert_forward_result, tuple):
            sequence_output, all_hidden_states = bert_forward_result
        else:
            raise ValueError("BERTEmbedding should return tuple when output_hidden_states=True")

        # all_hidden_states: tuple of [batch, seq_len, hidden_size]
        # DistilBERT: 7 層 (embedding + 6 transformer layers)

        num_layers = len(all_hidden_states)

        # === 2. Extract hierarchical features ===
        # Low-level: 前 1/3 層 (層 1-2)
        low_end = max(1, num_layers // 3)
        low_states = torch.mean(torch.stack(all_hidden_states[1:low_end+1]), dim=0)

        # Mid-level: 中 1/3 層 (層 3-4)
        mid_start = low_end + 1
        mid_end = max(mid_start, 2 * num_layers // 3)
        mid_states = torch.mean(torch.stack(all_hidden_states[mid_start:mid_end+1]), dim=0)

        # High-level: 後 1/3 層 (層 5-6)
        high_start = mid_end + 1
        high_states = torch.mean(torch.stack(all_hidden_states[high_start:]), dim=0)

        # Project to hidden_dim
        low_features = self.low_projection(low_states)    # [batch, seq_len, hidden_dim]
        mid_features = self.mid_projection(mid_states)    # [batch, seq_len, hidden_dim]
        high_features = self.high_projection(high_states) # [batch, seq_len, hidden_dim]

        # === 3. Process each aspect ===
        logits_list = []

        for i in range(max_aspects):
            # Aspect BERT encoding
            aspect_mask = aspect_attention_mask[:, i, :]  # [batch, aspect_len]

            # Skip invalid aspects (all padding)
            if aspect_mask.sum() == 0:
                logits_list.append(torch.zeros(batch_size, self.num_classes, device=input_ids.device))
                continue

            aspect_result = self.bert_absa.bert_embedding(
                input_ids=aspect_input_ids[:, i, :],
                attention_mask=aspect_mask
            )
            # Unpack if tuple (when output_hidden_states=True)
            if isinstance(aspect_result, tuple):
                aspect_output = aspect_result[0]  # sequence_output only
            else:
                aspect_output = aspect_result

            aspect_repr = aspect_output[:, 0, :]  # [CLS] token → [batch, hidden_size]
            aspect_repr = aspect_repr.unsqueeze(1)  # [batch, 1, hidden_size]

            # === 3.1 Aspect-aware fusion (每個層級) ===
            # Low-level fusion
            low_fused, _ = self.low_fusion(
                query=aspect_repr,
                key=low_features,
                value=low_features,
                key_padding_mask=~attention_mask.bool()
            )
            low_fused = low_fused.squeeze(1)  # [batch, hidden_dim]

            # Mid-level fusion
            mid_fused, _ = self.mid_fusion(
                query=aspect_repr,
                key=mid_features,
                value=mid_features,
                key_padding_mask=~attention_mask.bool()
            )
            mid_fused = mid_fused.squeeze(1)  # [batch, hidden_dim]

            # High-level fusion
            high_fused, _ = self.high_fusion(
                query=aspect_repr,
                key=high_features,
                value=high_features,
                key_padding_mask=~attention_mask.bool()
            )
            high_fused = high_fused.squeeze(1)  # [batch, hidden_dim]

            # ⭐ 3.2 Layer-wise Attention: 動態加權組合
            # 計算歸一化權重
            layer_attention = torch.softmax(self.layer_weights, dim=0)  # [3]

            # 加權求和
            hierarchical_repr = (
                layer_attention[0] * low_fused +
                layer_attention[1] * mid_fused +
                layer_attention[2] * high_fused
            )  # [batch, hidden_dim]

            # === 3.3 Classification ===
            logit = self.classifier(hierarchical_repr)  # [batch, num_classes]
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]

        # 返回額外資訊：學到的層級權重
        extras = {
            'layer_attention': layer_attention.detach().cpu().numpy()
        }

        return logits, extras


def create_improved_model(
    model_type: str,
    bert_model_name: str = 'distilbert-base-uncased',
    freeze_bert: bool = False,
    hidden_dim: int = 768,
    num_classes: int = 3,
    dropout: float = 0.1,
    **kwargs
):
    """
    工廠函數：創建改進模型

    Args:
        model_type: 模型類型
            - 'hierarchical': Hierarchical BERT (固定拼接)
            - 'hierarchical_layerattn': HBL (Layer-wise Attention)
        bert_model_name: BERT 模型名稱
        freeze_bert: 是否凍結 BERT
        hidden_dim: 隱藏層維度
        num_classes: 類別數量
        dropout: Dropout 比率
        **kwargs: 其他參數

    Returns:
        改進模型實例
    """
    if model_type == 'hierarchical':
        return HierarchicalBERT(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_type == 'hierarchical_layerattn':
        return HierarchicalBERT_LayerAttn(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown improved model type: {model_type}. "
                        f"Choose from: 'hierarchical', 'hierarchical_layerattn'")
