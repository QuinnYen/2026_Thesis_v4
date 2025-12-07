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
from models.hierarchical_syntax import HierarchicalSyntaxAttention, create_hsa_model


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
        - 支持 Supervised Contrastive Learning（可選）

    與 Baseline 對比:
        - Baseline (BERT-CLS): 只用最後一層的 [CLS] token
        - 本方法: 利用多層級特徵，預期提升 2-3% F1

    與 HBL 對比:
        - 本方法: 固定拼接 (concatenation)
        - HBL: 動態學習權重 (Layer-wise Attention)
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1,
        use_contrastive: bool = False  # 是否啟用對比學習
    ):
        super(HierarchicalBERT, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_contrastive = use_contrastive

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
        # BERT/DeBERTa: 12 layers
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

        # Projection head for contrastive learning (optional)
        # 將 hidden_dim*3 投影到較小的空間，更適合對比學習
        if use_contrastive:
            self.projection_head = nn.Sequential(
                nn.Linear(hidden_dim * 3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(
        self,
        pair_input_ids: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        pair_token_type_ids: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        使用正確的 BERT sentence-pair 格式：
            [CLS] text [SEP] aspect [SEP]
            token_type_ids: 0...0 1...1

        參數:
            pair_input_ids: [batch, max_aspects, seq_len] - 已編碼的 text-aspect pairs
            pair_attention_mask: [batch, max_aspects, seq_len]
            pair_token_type_ids: [batch, max_aspects, seq_len] - 區分 text(0) 和 aspect(1)
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
            None: (無額外資訊)
        """
        batch_size, max_aspects, seq_len = pair_input_ids.shape

        logits_list = []
        features_list = []  # 收集特徵表示（用於對比學習）

        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                # 獲取第 i 個 aspect 的 sentence-pair 編碼
                input_ids = pair_input_ids[:, i, :]        # [batch, seq_len]
                attention_mask = pair_attention_mask[:, i, :]  # [batch, seq_len]
                token_type_ids = pair_token_type_ids[:, i, :]  # [batch, seq_len]

                # Get hierarchical features from BERT (不使用 token_type_ids)
                outputs = self.bert_absa.bert_embedding.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
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

                # 收集特徵表示（用於對比學習）
                if self.use_contrastive:
                    # 使用 projection head 投影特徵
                    projected_repr = self.projection_head(hierarchical_repr)  # [batch, hidden_dim]
                    features_list.append(projected_repr)
                else:
                    features_list.append(hierarchical_repr)  # [batch, hidden*3]
            else:
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=pair_input_ids.device)
                )
                # 對於無效 aspect，填充零向量
                feature_dim = self.hidden_dim if self.use_contrastive else self.hidden_dim * 3
                features_list.append(
                    torch.zeros(batch_size, feature_dim, device=pair_input_ids.device)
                )

        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]
        features = torch.stack(features_list, dim=1)  # [batch, max_aspects, feature_dim]

        # extras 包含特徵表示（用於對比學習）
        extras = {'features': features}

        return logits, extras


class HierarchicalBERT_LayerAttn(BaseModel):
    """
    Hierarchical BERT + Layer-wise Attention (HBL)

    基於 UDify (Kondratyuk & Straka, EMNLP 2019) 的 Layer-wise Attention

    架構:
        BERT Layers → Extract Multi-Level Features → Layer-wise Attention → Classifier

    設計理念:
        1. 從 BERT 不同層提取 Low/Mid/High 層級特徵
        2. 使用可學習權重動態加權組合（取代固定拼接）
        3. 減少分類器參數量 (hidden_dim*3 → hidden_dim)
        4. 提供可解釋性（權重可視化）

    公式:
        α = [w_low, w_mid, w_high]  # 可學習參數
        β = softmax(α)               # 歸一化權重
        h = Σ(β_i * h_i)             # 加權組合

    與 HierarchicalBERT 的差異:
        - HierarchicalBERT: 固定拼接 concat([low, mid, high])
        - HBL: 動態加權 β₁×low + β₂×mid + β₃×high

    參考文獻:
        Kondratyuk, D., & Straka, M. (2019). 75 Languages, 1 Model: Parsing
        Universal Dependencies Universally. In Proceedings of EMNLP-IJCNLP
        2019 (pp. 2779-2795).
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
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

        # Hierarchical layer indices (BERT: 12 layers)
        self.low_layers = [1, 2, 3, 4]
        self.mid_layers = [5, 6, 7, 8]
        self.high_layers = [9, 10, 11, 12]

        # === Fusion layers for each level (與 HierarchicalBERT 一致的 concat 方式) ===
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

        # ⭐ Layer-wise Attention Weights (可學習參數)
        # 使用非均勻初始化打破對稱性
        # 根據 UDify 論文，高層特徵通常更重要，因此給予較高初始值
        self.layer_weights = nn.Parameter(torch.tensor([0.5, 1.0, 1.5]))

        # === Final classifier ===
        # 輸入維度為 hidden_dim (加權求和後)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        pair_input_ids: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        pair_token_type_ids: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        使用 BERT sentence-pair 格式：[CLS] text [SEP] aspect [SEP]

        參數:
            pair_input_ids: [batch, max_aspects, seq_len] - 已編碼的 text-aspect pairs
            pair_attention_mask: [batch, max_aspects, seq_len]
            pair_token_type_ids: [batch, max_aspects, seq_len] - 區分 text(0) 和 aspect(1)
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
            extras: dict with 'layer_attention' weights
        """
        batch_size, max_aspects, seq_len = pair_input_ids.shape

        logits_list = []

        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                # 獲取第 i 個 aspect 的 sentence-pair 編碼
                input_ids = pair_input_ids[:, i, :]
                attention_mask = pair_attention_mask[:, i, :]

                # Get hierarchical features from BERT
                outputs = self.bert_absa.bert_embedding.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                all_hidden_states = outputs.hidden_states

                # Extract [CLS] token from different layers (使用 concat)
                low_features = []
                for layer_idx in self.low_layers:
                    low_features.append(all_hidden_states[layer_idx][:, 0, :])
                low_features = torch.cat(low_features, dim=-1)  # [batch, hidden*4]

                mid_features = []
                for layer_idx in self.mid_layers:
                    mid_features.append(all_hidden_states[layer_idx][:, 0, :])
                mid_features = torch.cat(mid_features, dim=-1)

                high_features = []
                for layer_idx in self.high_layers:
                    high_features.append(all_hidden_states[layer_idx][:, 0, :])
                high_features = torch.cat(high_features, dim=-1)

                # Hierarchical fusion
                low_fused = self.low_fusion(low_features)    # [batch, hidden_dim]
                mid_fused = self.mid_fusion(mid_features)    # [batch, hidden_dim]
                high_fused = self.high_fusion(high_features)  # [batch, hidden_dim]

                # ⭐ Layer-wise Attention: 動態加權組合
                layer_attention = torch.softmax(self.layer_weights, dim=0)  # [3]

                # 加權求和 (取代固定拼接)
                hierarchical_repr = (
                    layer_attention[0] * low_fused +
                    layer_attention[1] * mid_fused +
                    layer_attention[2] * high_fused
                )  # [batch, hidden_dim]

                # Classification
                logit = self.classifier(hierarchical_repr)  # [batch, num_classes]
                logits_list.append(logit)
            else:
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=pair_input_ids.device)
                )

        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]

        # 計算當前的 layer attention weights
        layer_attention = torch.softmax(self.layer_weights, dim=0)

        # 返回額外資訊：學到的層級權重
        extras = {
            'layer_attention': layer_attention.detach().cpu().numpy()
        }

        return logits, extras


class HierarchicalBERT_AspectAware(BaseModel):
    """
    Hierarchical BERT + Aspect-aware Pooling

    核心改進：使用 Aspect-guided Attention 替代純 [CLS] token

    設計理念：
        1. 從 BERT 不同層提取 Low/Mid/High 層級的完整序列特徵
        2. 使用 aspect 表示作為 query，對每個層級的序列做 attention
        3. 捕捉 aspect 周圍的上下文信息（修飾語、否定詞等）
        4. 保留固定拼接，但使用 aspect-aware 的特徵

    與 HierarchicalBERT 的差異：
        - HierarchicalBERT: 只用 [CLS] token
        - AspectAware: 用 aspect 作為 query 對整個序列做 attention

    預期效果：
        - 更好地捕捉 "not good"、"very bad" 等修飾語
        - 對 Neutral 類別有提升（需要細微語義差異）
        - +1.0~2.0% Macro-F1
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1,
        num_attention_heads: int = 4
    ):
        super(HierarchicalBERT_AspectAware, self).__init__()

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

        # Hierarchical layer indices
        self.low_layers = [1, 2, 3, 4]
        self.mid_layers = [5, 6, 7, 8]
        self.high_layers = [9, 10, 11, 12]

        # Layer projection (將每層平均後投影)
        self.layer_projection = nn.Linear(bert_hidden_size, hidden_dim)

        # Aspect-guided Attention for each level
        self.low_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.mid_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        self.high_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization for each level
        self.low_norm = nn.LayerNorm(hidden_dim)
        self.mid_norm = nn.LayerNorm(hidden_dim)
        self.high_norm = nn.LayerNorm(hidden_dim)

        # Final classifier (concatenate all 3 levels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _get_aspect_mask_from_token_type_ids(
        self,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        從 token_type_ids 中提取 aspect 的位置
        token_type_ids: 0=text, 1=aspect
        返回: aspect token 的 mask
        """
        # aspect tokens: token_type_ids == 1 且 attention_mask == 1
        aspect_token_mask = (token_type_ids == 1) & (attention_mask == 1)
        return aspect_token_mask

    def forward(
        self,
        pair_input_ids: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        pair_token_type_ids: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        使用 BERT sentence-pair 格式：[CLS] text [SEP] aspect [SEP]

        參數:
            pair_input_ids: [batch, max_aspects, seq_len]
            pair_attention_mask: [batch, max_aspects, seq_len]
            pair_token_type_ids: [batch, max_aspects, seq_len] - 區分 text(0) 和 aspect(1)
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
            extras: dict with attention info
        """
        batch_size, max_aspects, seq_len = pair_input_ids.shape

        logits_list = []

        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                input_ids = pair_input_ids[:, i, :]
                attention_mask = pair_attention_mask[:, i, :]
                token_type_ids = pair_token_type_ids[:, i, :]

                # Get BERT outputs with all hidden states
                outputs = self.bert_absa.bert_embedding.bert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )

                all_hidden_states = outputs.hidden_states

                # 提取各層級的平均特徵 (對整個序列)
                # Low-level features
                low_states = torch.stack([all_hidden_states[idx] for idx in self.low_layers], dim=0)
                low_seq = low_states.mean(dim=0)  # [batch, seq_len, hidden]

                # Mid-level features
                mid_states = torch.stack([all_hidden_states[idx] for idx in self.mid_layers], dim=0)
                mid_seq = mid_states.mean(dim=0)

                # High-level features
                high_states = torch.stack([all_hidden_states[idx] for idx in self.high_layers], dim=0)
                high_seq = high_states.mean(dim=0)

                # Project to hidden_dim
                low_seq = self.layer_projection(low_seq)   # [batch, seq_len, hidden_dim]
                mid_seq = self.layer_projection(mid_seq)
                high_seq = self.layer_projection(high_seq)

                # 獲取 aspect token 的表示作為 query
                aspect_token_mask = self._get_aspect_mask_from_token_type_ids(token_type_ids, attention_mask)

                # 獲取 aspect tokens 的平均表示，使用最高層的表示作為 aspect query
                # 對每個樣本獨立處理，避免 batch 級別的全 0 檢查
                aspect_count = aspect_token_mask.sum(dim=1, keepdim=True).float()  # [batch, 1]
                has_aspect = (aspect_count > 0).unsqueeze(-1)  # [batch, 1, 1]

                # 計算 aspect 平均表示
                aspect_features = high_seq * aspect_token_mask.unsqueeze(-1).float()
                aspect_sum = aspect_features.sum(dim=1, keepdim=True)  # [batch, 1, hidden]
                aspect_avg = aspect_sum / aspect_count.clamp(min=1).unsqueeze(-1)  # [batch, 1, hidden_dim]

                # 對於沒有 aspect token 的樣本，使用 [CLS] token
                cls_query = high_seq[:, 0:1, :]  # [batch, 1, hidden_dim]
                aspect_query = torch.where(has_aspect, aspect_avg, cls_query)

                # Key padding mask for attention (True = ignore)
                key_padding_mask = ~attention_mask.bool()

                # 確保至少有一個位置不被 mask（避免 attention 全為 0 導致 NaN）
                # 如果全被 mask，則不使用 key_padding_mask
                all_masked = key_padding_mask.all(dim=1, keepdim=True)  # [batch, 1]
                if all_masked.any():
                    # 對於全 mask 的樣本，設置 key_padding_mask 為全 False
                    key_padding_mask = key_padding_mask & ~all_masked

                # Aspect-guided Attention for each level
                # Low-level: 詞彙/句法特徵
                low_attended, _ = self.low_attention(
                    query=aspect_query,
                    key=low_seq,
                    value=low_seq,
                    key_padding_mask=key_padding_mask
                )
                low_fused = self.low_norm(low_attended.squeeze(1))  # [batch, hidden_dim]

                # Mid-level: 語義特徵
                mid_attended, _ = self.mid_attention(
                    query=aspect_query,
                    key=mid_seq,
                    value=mid_seq,
                    key_padding_mask=key_padding_mask
                )
                mid_fused = self.mid_norm(mid_attended.squeeze(1))

                # High-level: 任務特徵
                high_attended, _ = self.high_attention(
                    query=aspect_query,
                    key=high_seq,
                    value=high_seq,
                    key_padding_mask=key_padding_mask
                )
                high_fused = self.high_norm(high_attended.squeeze(1))

                # 防止 NaN：將 NaN 替換為 0
                if torch.isnan(low_fused).any() or torch.isnan(mid_fused).any() or torch.isnan(high_fused).any():
                    low_fused = torch.nan_to_num(low_fused, nan=0.0)
                    mid_fused = torch.nan_to_num(mid_fused, nan=0.0)
                    high_fused = torch.nan_to_num(high_fused, nan=0.0)

                # Concatenate all levels (固定拼接)
                hierarchical_repr = torch.cat([low_fused, mid_fused, high_fused], dim=-1)

                # Classification
                logit = self.classifier(hierarchical_repr)
                logits_list.append(logit)
            else:
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=pair_input_ids.device)
                )

        logits = torch.stack(logits_list, dim=1)

        extras = {'model_type': 'aspect_aware'}

        return logits, extras


class InterAspectRelationNetwork(BaseModel):
    """
    方法 3: Inter-Aspect Relation Network (IARN)

    核心創新：顯式建模多個 aspects 之間的交互關係

    與 HPNet 的關鍵差異：
        - HPNet: 獨立處理每個 aspect
        - IARN: 顯式建模 aspects 之間的依賴關係

    架構：
        BERT → Hierarchical Features → Aspect-to-Aspect Attention
        → Relation-aware Gating → Classifier

    核心組件：
        1. Hierarchical Feature Extraction: 從 Low/Mid/High 層提取特徵
        2. Aspect-to-Aspect Attention: 讓每個 aspect 關注其他 aspects
        3. Relation-aware Gating: 動態調整自身特徵 vs 上下文特徵

    適用場景：
        - MAMS: 100% 多 aspect 句子（平均 3.2 aspects/sentence）
        - 對比關係："food is great but service is bad"
        - 因果關係："service is slow, making experience frustrating"

    理論貢獻：
        - Inter-aspect dependency modeling
        - Multi-aspect sentiment analysis專門優化
        - 超越 HPNet 的 task-specific layer selection

    預期提升：
        - MAMS: +3-5% F1 (相比 Method 1)
        - 更好的可解釋性（attention weights 可視化）
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1,
        num_attention_heads: int = 4
    ):
        super(InterAspectRelationNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads

        # BERT with hierarchical output
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        # Enable output of all hidden states
        self.bert_absa.bert_embedding.output_hidden_states = True
        self.bert_absa.bert_embedding.bert.config.output_hidden_states = True

        bert_hidden_size = self.bert_absa.hidden_size

        # Hierarchical fusion layers (same as HierarchicalBERT)
        # BERT/DeBERTa: 12 layers
        self.low_layers = [1, 2, 3, 4]
        self.mid_layers = [5, 6, 7, 8]
        self.high_layers = [9, 10, 11, 12]

        # === 1. Hierarchical Feature Projections ===
        self.proj_low = nn.Linear(bert_hidden_size, hidden_dim)
        self.proj_mid = nn.Linear(bert_hidden_size, hidden_dim)
        self.proj_high = nn.Linear(bert_hidden_size, hidden_dim)

        self.layer_norm_low = nn.LayerNorm(hidden_dim)
        self.layer_norm_mid = nn.LayerNorm(hidden_dim)
        self.layer_norm_high = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # === 2. Aspect-to-Aspect Attention ===
        # Multi-head attention over aspects (using concatenated hierarchical features)
        self.aspect_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 3,  # Concatenated Low+Mid+High
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # === 3. Relation-aware Gating ===
        # Gate to balance self features vs context features
        self.relation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 * 2, hidden_dim),  # [self; context]
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # === 4. Final Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        pair_input_ids: torch.Tensor,
        pair_attention_mask: torch.Tensor,
        pair_token_type_ids: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        使用正確的 BERT sentence-pair 格式：
            [CLS] text [SEP] aspect [SEP]
            token_type_ids: 0...0 1...1

        參數：
            pair_input_ids: [batch, max_aspects, seq_len] - 已編碼的 text-aspect pairs
            pair_attention_mask: [batch, max_aspects, seq_len]
            pair_token_type_ids: [batch, max_aspects, seq_len] - 區分 text(0) 和 aspect(1)
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回：
            logits: [batch, max_aspects, num_classes]
            extras: dict with attention weights and gate values
        """
        batch_size, max_aspects, seq_len = pair_input_ids.shape

        # === Step 1: Extract hierarchical features for all aspects ===
        all_self_features = []  # Will store [batch, max_aspects, hidden*3]

        for i in range(max_aspects):
            # 獲取第 i 個 aspect 的 sentence-pair 編碼
            input_ids = pair_input_ids[:, i, :]        # [batch, seq_len]
            attention_mask = pair_attention_mask[:, i, :]  # [batch, seq_len]
            token_type_ids = pair_token_type_ids[:, i, :]  # [batch, seq_len]

            # Get BERT outputs (不使用 token_type_ids，ABSA 任務中使用會降低效果)
            outputs = self.bert_absa.bert_embedding.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )

            all_hidden_states = outputs.hidden_states

            # Extract and average features from different levels
            # Low-level features
            low_feats = torch.stack([
                all_hidden_states[idx][:, 0, :]  # CLS token
                for idx in self.low_layers
            ], dim=0).mean(dim=0)  # [batch, hidden]

            # Mid-level features
            mid_feats = torch.stack([
                all_hidden_states[idx][:, 0, :]
                for idx in self.mid_layers
            ], dim=0).mean(dim=0)

            # High-level features
            high_feats = torch.stack([
                all_hidden_states[idx][:, 0, :]
                for idx in self.high_layers
            ], dim=0).mean(dim=0)

            # Project and normalize
            low_proj = self.dropout(F.relu(self.layer_norm_low(self.proj_low(low_feats))))
            mid_proj = self.dropout(F.relu(self.layer_norm_mid(self.proj_mid(mid_feats))))
            high_proj = self.dropout(F.relu(self.layer_norm_high(self.proj_high(high_feats))))

            # Concatenate hierarchical features
            aspect_features = torch.cat([low_proj, mid_proj, high_proj], dim=-1)  # [batch, hidden*3]
            all_self_features.append(aspect_features)

        # Stack all aspects: [batch, max_aspects, hidden*3]
        self_features = torch.stack(all_self_features, dim=1)

        # === Step 2: Aspect-to-Aspect Attention ===
        # Query: current aspect, Key/Value: all aspects
        # Mask out invalid aspects
        attn_mask = ~aspect_mask  # True for invalid aspects (to be masked)

        context_features, attn_weights = self.aspect_attention(
            query=self_features,
            key=self_features,
            value=self_features,
            key_padding_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        # context_features: [batch, max_aspects, hidden*3]
        # attn_weights: [batch, max_aspects, max_aspects]

        # === Step 3: Relation-aware Gating ===
        # Concatenate self and context features
        combined = torch.cat([self_features, context_features], dim=-1)  # [batch, max_aspects, hidden*6]

        # Compute gate values
        gate_values = self.relation_gate(combined)  # [batch, max_aspects, 1]

        # Gated fusion: dynamically balance self vs context
        fused_features = gate_values * self_features + (1 - gate_values) * context_features
        # [batch, max_aspects, hidden*3]

        # === Step 4: Classification ===
        logits = self.classifier(fused_features)  # [batch, max_aspects, num_classes]

        # Mask out invalid aspects
        logits = logits.masked_fill(~aspect_mask.unsqueeze(-1), 0.0)

        # Return extras for analysis
        extras = {
            'aspect_attention_weights': attn_weights.detach().cpu(),  # [batch, max_aspects, max_aspects]
            'gate_values': gate_values.squeeze(-1).detach().cpu(),     # [batch, max_aspects]
            'avg_gate': gate_values[aspect_mask].mean().item()         # Scalar: average gate value
        }

        return logits, extras


class VectorProjection(nn.Module):
    """
    向量投影模組

    核心公式: proj_v(u) = (u·v / ||v||²) * v

    作用：
        - 從聚合向量中提取與目標 aspect 相關的情感語義
        - 過濾其他 aspects 的干擾信息
        - 對單/多 aspect 場景都有效

    參考：VP-ACL (2025) - Vector Projection for Aspect-level Sentiment
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, vector: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """
        向量投影運算

        Args:
            vector: [batch, hidden] - 要投影的向量（多aspect聚合表示）
            direction: [batch, hidden] - 投影方向（目標aspect表示）

        Returns:
            projection: [batch, hidden] - 投影結果
        """
        # 計算內積: u·v
        dot_product = (vector * direction).sum(dim=-1, keepdim=True)
        # Shape: [batch, 1]

        # 計算方向向量的模平方: ||v||²
        norm_squared = (direction * direction).sum(dim=-1, keepdim=True) + 1e-8
        # Shape: [batch, 1]

        # 投影公式: (u·v / ||v||²) * v
        projection = (dot_product / norm_squared) * direction
        # Shape: [batch, hidden]

        return projection


class VP_IARN(BaseModel):
    """
    方法 4: VP-IARN (Vector Projection enhanced Inter-Aspect Relation Network)

    核心創新：結合向量投影與 Aspect-to-Aspect Attention，統一處理單/多 aspect 場景

    與 VP-ACL (2025) 的差異：
        - VP-ACL: 純向量投影，無 aspect 間顯式建模
        - VP-IARN: 向量投影 + Aspect-to-Aspect Attention + 自適應融合

    與原始 IARN 的差異：
        - IARN: 僅依賴 Aspect-to-Aspect Attention，在單 aspect 場景失效
        - VP-IARN: 向量投影為單 aspect 提供有效表示，自適應權重動態調整

    架構：
        1. Hierarchical Feature Extraction (Low/Mid/High BERT layers)
        2. Multi-Aspect Aggregation (聚合所有 aspects 的表示)
        3. Vector Projection (投影到每個目標 aspect 方向)
        4. Aspect-to-Aspect Attention (僅多 aspect 時啟用)
        5. Adaptive Fusion (自適應融合投影特徵與注意力特徵)
        6. Classification

    適用場景：
        - MAMS (100% 多 aspect): 向量投影 + Attention 雙重增強
        - Restaurants (20% 多 aspect): 向量投影為單 aspect 提供有效表示
        - 通用 ABSA 任務

    預期效果：
        - MAMS: F1 ≈ 0.8450 (+0.5% vs IARN)
        - Restaurants: F1 ≈ 0.7280 (+2.7% vs IARN, 超越 Baseline)
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1,
        num_attention_heads: int = 4
    ):
        super(VP_IARN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads

        # BERT with hierarchical output
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        # Enable output of all hidden states
        self.bert_absa.bert_embedding.output_hidden_states = True
        self.bert_absa.bert_embedding.bert.config.output_hidden_states = True

        bert_hidden_size = self.bert_absa.hidden_size

        # Hierarchical layer indices (BERT/DeBERTa: 12 layers)
        self.low_layers = [1, 2, 3, 4]
        self.mid_layers = [5, 6, 7, 8]
        self.high_layers = [9, 10, 11, 12]

        # === 1. Hierarchical Feature Projections ===
        self.proj_low = nn.Linear(bert_hidden_size, hidden_dim)
        self.proj_mid = nn.Linear(bert_hidden_size, hidden_dim)
        self.proj_high = nn.Linear(bert_hidden_size, hidden_dim)

        self.layer_norm_low = nn.LayerNorm(hidden_dim)
        self.layer_norm_mid = nn.LayerNorm(hidden_dim)
        self.layer_norm_high = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

        # === 2. Vector Projection Module (核心創新) ===
        self.vector_projection = VectorProjection(hidden_dim * 3)

        # === 3. Aspect-to-Aspect Attention ===
        self.aspect_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 3,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True
        )

        # === 4. Relation-aware Gating ===
        self.relation_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3 * 2, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # === 5. Adaptive Fusion Weight (可學習參數) ===
        # 控制 projection features vs attention features 的融合比例
        # 初始化為 0.5，讓模型自己學習最佳比例
        self.adaptive_alpha = nn.Parameter(torch.tensor(0.5))

        # === 6. Final Classifier ===
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def _extract_hierarchical_features(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        提取單個 aspect 的階層特徵

        Returns:
            features: [batch, hidden*3]
        """
        # Concatenate text and aspect
        combined_ids = torch.cat([text_input_ids, aspect_input_ids], dim=1)
        combined_mask = torch.cat([text_attention_mask, aspect_attention_mask], dim=1)

        # Get BERT outputs
        outputs = self.bert_absa.bert_embedding.bert(
            input_ids=combined_ids,
            attention_mask=combined_mask,
            return_dict=True
        )

        all_hidden_states = outputs.hidden_states

        # Extract features from different levels (CLS token)
        low_feats = torch.stack([
            all_hidden_states[idx][:, 0, :]
            for idx in self.low_layers
        ], dim=0).mean(dim=0)

        mid_feats = torch.stack([
            all_hidden_states[idx][:, 0, :]
            for idx in self.mid_layers
        ], dim=0).mean(dim=0)

        high_feats = torch.stack([
            all_hidden_states[idx][:, 0, :]
            for idx in self.high_layers
        ], dim=0).mean(dim=0)

        # Project and normalize
        low_proj = self.dropout(F.relu(self.layer_norm_low(self.proj_low(low_feats))))
        mid_proj = self.dropout(F.relu(self.layer_norm_mid(self.proj_mid(mid_feats))))
        high_proj = self.dropout(F.relu(self.layer_norm_high(self.proj_high(high_feats))))

        # Concatenate hierarchical features
        return torch.cat([low_proj, mid_proj, high_proj], dim=-1)  # [batch, hidden*3]

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播 - VP-IARN 統一處理單/多面向

        參數：
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            aspect_input_ids: [batch, max_aspects, aspect_len]
            aspect_attention_mask: [batch, max_aspects, aspect_len]
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回：
            logits: [batch, max_aspects, num_classes]
            extras: dict with attention weights, projection info, gate values
        """
        batch_size, max_aspects, aspect_len = aspect_input_ids.shape
        device = text_input_ids.device

        # Count valid aspects per sample
        num_valid_aspects = aspect_mask.sum(dim=1)  # [batch]

        # === Step 1: Extract hierarchical features for all aspects ===
        all_aspect_features = []

        for i in range(max_aspects):
            aspect_features = self._extract_hierarchical_features(
                text_input_ids,
                text_attention_mask,
                aspect_input_ids[:, i, :],
                aspect_attention_mask[:, i, :]
            )
            all_aspect_features.append(aspect_features)

        # Stack: [batch, max_aspects, hidden*3]
        aspect_features = torch.stack(all_aspect_features, dim=1)

        # === Step 2: Multi-Aspect Aggregation ===
        # 聚合所有有效 aspects 的表示，形成句子級多面向密集向量
        # 使用 aspect_mask 進行 masked mean
        masked_features = aspect_features * aspect_mask.unsqueeze(-1).float()
        # 避免除零
        valid_counts = num_valid_aspects.clamp(min=1).unsqueeze(-1)  # [batch, 1]
        multi_aspect_dense = masked_features.sum(dim=1) / valid_counts  # [batch, hidden*3]

        # === Step 3: Vector Projection ===
        # 將聚合向量投影到每個目標 aspect 方向
        projected_features = []

        for i in range(max_aspects):
            target_direction = aspect_features[:, i, :]  # [batch, hidden*3]
            projection = self.vector_projection(multi_aspect_dense, target_direction)
            projected_features.append(projection)

        projected_features = torch.stack(projected_features, dim=1)  # [batch, max_aspects, hidden*3]

        # === Step 4: Aspect-to-Aspect Attention ===
        # 對所有樣本計算 attention（向量投影已處理單 aspect 情況）
        attn_mask = ~aspect_mask  # True for invalid aspects
        context_features, attn_weights = self.aspect_attention(
            query=aspect_features,
            key=aspect_features,
            value=aspect_features,
            key_padding_mask=attn_mask,
            need_weights=True,
            average_attn_weights=True
        )
        # context_features: [batch, max_aspects, hidden*3]
        # attn_weights: [batch, max_aspects, max_aspects]

        # === Step 5: Relation-aware Gating ===
        # Gate to balance self features vs context features
        combined = torch.cat([aspect_features, context_features], dim=-1)
        gate_values = self.relation_gate(combined)  # [batch, max_aspects, 1]

        # Gated attention output
        gated_attn_features = gate_values * aspect_features + (1 - gate_values) * context_features
        # [batch, max_aspects, hidden*3]

        # === Step 6: Adaptive Fusion ===
        # 根據 aspect 數量動態調整 projection vs attention 的權重
        alpha = torch.sigmoid(self.adaptive_alpha)  # 可學習的融合係數

        # 多 aspect 樣本：更依賴 attention 特徵
        # 單 aspect 樣本：更依賴 projection 特徵
        # 使用 soft 版本：根據 aspect 數量連續調整
        aspect_ratio = (num_valid_aspects.float() / max_aspects).unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]

        # 動態權重：multi-aspect 時 alpha 更高（更多 attention），single-aspect 時 alpha 更低（更多 projection）
        dynamic_alpha = alpha * aspect_ratio + (1 - alpha) * (1 - aspect_ratio)

        # 融合投影特徵和注意力特徵
        final_features = dynamic_alpha * gated_attn_features + (1 - dynamic_alpha) * projected_features
        # [batch, max_aspects, hidden*3]

        # === Step 7: Classification ===
        logits = self.classifier(final_features)  # [batch, max_aspects, num_classes]

        # Mask out invalid aspects
        logits = logits.masked_fill(~aspect_mask.unsqueeze(-1), 0.0)

        # Statistics for analysis
        n_multi = (num_valid_aspects >= 2).sum().item()
        n_single = (num_valid_aspects == 1).sum().item()

        extras = {
            'aspect_attention_weights': attn_weights.detach().cpu(),
            'gate_values': gate_values.squeeze(-1).detach().cpu(),
            'avg_gate': gate_values[aspect_mask.unsqueeze(-1).expand_as(gate_values)].mean().item() if aspect_mask.any() else 0.0,
            # VP-IARN specific
            'adaptive_alpha': alpha.item(),
            'dynamic_alpha_mean': dynamic_alpha.mean().item(),
            'n_multi_aspect_samples': n_multi,
            'n_single_aspect_samples': n_single,
            'multi_aspect_ratio': n_multi / batch_size if batch_size > 0 else 0.0,
            'mode': 'vp_iarn'
        }

        return logits, extras


# ============================================================
# Factory Function
# ============================================================

def create_improved_model(model_type, args, num_classes=3):
    """
    創建改進模型的工廠函數

    支援的模型類型:
    - 'hierarchical': HierarchicalBERT (基礎階層模型)
    - 'hierarchical_layerattn': HierarchicalBERT_LayerAttn (層級注意力)
    - 'aspect_aware': HierarchicalBERT_AspectAware (面向感知)
    - 'iarn': InterAspectRelationNetwork (面向間關係網絡)
    - 'vp_iarn': VP_IARN (向量投影 + IARN)
    - 'hsa': HierarchicalBERT_AspectAware (別名)

    Args:
        model_type: 模型類型字串
        args: 命令行參數
        num_classes: 分類類別數

    Returns:
        對應的模型實例
    """
    bert_model = getattr(args, 'bert_model', 'bert-base-uncased')
    hidden_dim = getattr(args, 'hidden_dim', 256)
    dropout = getattr(args, 'dropout', 0.3)

    if model_type == 'hierarchical':
        return HierarchicalBERT(
            bert_model_name=bert_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=getattr(args, 'freeze_bert', False)
        )

    elif model_type == 'hierarchical_layerattn':
        return HierarchicalBERT_LayerAttn(
            bert_model_name=bert_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=getattr(args, 'freeze_bert', False)
        )

    elif model_type in ['aspect_aware', 'hsa']:
        return HierarchicalBERT_AspectAware(
            bert_model_name=bert_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=getattr(args, 'freeze_bert', False),
            num_heads=getattr(args, 'num_attention_heads', 4)
        )

    elif model_type == 'iarn':
        return InterAspectRelationNetwork(
            bert_model_name=bert_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=getattr(args, 'freeze_bert', False),
            num_heads=getattr(args, 'iarm_heads', 4),
            num_layers=getattr(args, 'iarm_layers', 2)
        )

    elif model_type == 'vp_iarn':
        return VP_IARN(
            bert_model_name=bert_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=getattr(args, 'freeze_bert', False),
            num_heads=getattr(args, 'iarm_heads', 4),
            num_layers=getattr(args, 'iarm_layers', 2)
        )

    elif model_type == 'hkgan':
        from models.hkgan import HKGAN
        return HKGAN(
            bert_model_name=bert_model,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            freeze_bert=getattr(args, 'freeze_bert', False),
            num_gat_heads=getattr(args, 'gat_heads', 4),
            num_gat_layers=getattr(args, 'gat_layers', 2),
            knowledge_weight=getattr(args, 'knowledge_weight', 0.1),
            use_senticnet=getattr(args, 'use_senticnet', True),
            # v2.0 新增：解決 Neutral 識別問題
            use_confidence_gate=getattr(args, 'use_confidence_gate', True),
            # v3.0 新增：動態知識門控（解決 MAMS 複雜句問題）
            use_dynamic_gate=getattr(args, 'use_dynamic_gate', True),
            domain=getattr(args, 'domain', None)
        )

    else:
        raise ValueError(f"不支援的模型類型: {model_type}。"
                        f"支援的類型: hierarchical, hierarchical_layerattn, "
                        f"aspect_aware, hsa, iarn, vp_iarn, hkgan")
