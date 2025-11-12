"""
HMACNet for Sentence-Level Sentiment Analysis

整合隱含 Aspect 發現模組，使 PMAC 和 IARM 創新可用於句子級別任務

核心架構:
    Text → BERT → Implicit Aspect Discovery → AAHA → PMAC → IARM → Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bert_embedding import BERTForABSA
from models.aaha_enhanced import AAHAEnhanced
from models.pmac_enhanced import PMACMultiAspect
from models.pmac_selective import SelectivePMACMultiAspect
from models.iarm_enhanced import IARMMultiAspect
from models.implicit_aspect_discovery import ImplicitAspectDiscovery, PREDEFINED_ASPECTS
from models.base_model import BaseModel


class HMACNetSentenceLevel(BaseModel):
    """
    句子級別 HMAC-Net

    創新點:
    - 使用 Implicit Aspect Discovery 自動發現語義面向
    - 保留原有的 PMAC 和 IARM 創新模組
    - 支援任何句子級別的情感分析任務
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1,
        # Implicit Aspect Discovery 參數
        num_implicit_aspects: int = 5,
        domain: str = 'generic',  # 領域類型
        # PMAC 參數
        use_pmac: bool = True,
        pmac_composition_mode: str = 'sequential',
        # IARM 參數
        use_iarm: bool = True,
        iarm_relation_mode: str = 'transformer',
        iarm_num_heads: int = 4,
        iarm_num_layers: int = 2,
        # 融合策略
        fusion_strategy: str = 'weighted_pooling'  # 'mean', 'max', 'weighted_pooling', 'attention'
    ):
        super(HMACNetSentenceLevel, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_implicit_aspects = num_implicit_aspects
        self.domain = domain
        self.use_pmac = use_pmac
        self.use_iarm = use_iarm
        self.fusion_strategy = fusion_strategy

        # 1. BERT 嵌入
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert_absa.hidden_size

        # 投影層（如果需要）
        if bert_hidden_size != hidden_dim:
            self.text_projection = nn.Linear(bert_hidden_size, hidden_dim)
        else:
            self.text_projection = nn.Identity()

        # 2. Implicit Aspect Discovery (隱含面向發現)
        aspect_names = PREDEFINED_ASPECTS.get(domain, PREDEFINED_ASPECTS['generic'])
        self.aspect_discovery = ImplicitAspectDiscovery(
            hidden_dim=hidden_dim,
            num_aspects=num_implicit_aspects,
            num_heads=8,
            dropout=dropout,
            predefined_aspect_names=aspect_names
        )

        # 3. AAHA (可選 - 為每個 aspect 精煉表示)
        # 在句子級別任務中，aspects 已經是從文本中提取的，
        # 但仍可使用 AAHA 進一步精煉
        self.use_aaha = False  # 可以設為 True 以啟用

        if self.use_aaha:
            self.aaha = AAHAEnhanced(
                hidden_dim=hidden_dim,
                aspect_dim=hidden_dim,
                word_attention_dims=[hidden_dim // 2],
                phrase_attention_dims=[hidden_dim // 2],
                sentence_attention_dims=[hidden_dim],
                attention_dropout=0.0,
                output_dropout=dropout
            )

        # 4. PMAC (多面向組合)
        if use_pmac:
            if pmac_composition_mode == 'selective':
                self.pmac = SelectivePMACMultiAspect(
                    input_dim=hidden_dim,
                    fusion_dim=hidden_dim,
                    num_composition_layers=2,
                    hidden_dim=256,
                    dropout=dropout,
                    use_layer_norm=True
                )
            else:
                self.pmac = PMACMultiAspect(
                    input_dim=hidden_dim,
                    fusion_dim=hidden_dim,
                    num_composition_layers=2,
                    hidden_dim=128,
                    dropout=dropout,
                    composition_mode=pmac_composition_mode
                )

        # 5. IARM (面向間關係建模)
        if use_iarm:
            self.iarm = IARMMultiAspect(
                input_dim=hidden_dim,
                relation_dim=hidden_dim,
                num_heads=iarm_num_heads,
                num_layers=iarm_num_layers,
                dropout=dropout,
                relation_mode=iarm_relation_mode
            )

        # 6. 融合層 (將多個 aspects 融合為句子級別表示)
        if fusion_strategy == 'weighted_pooling':
            # 學習每個 aspect 的權重
            self.aspect_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
                nn.Softmax(dim=1)
            )
        elif fusion_strategy == 'attention':
            # 注意力融合
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.fusion_query = nn.Parameter(torch.randn(1, hidden_dim))

        # 7. 分類器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        return_aspect_info: bool = False
    ):
        """
        前向傳播

        參數:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            return_aspect_info: 是否返回 aspect 相關資訊（用於可視化）

        返回:
            logits: [batch, num_classes]
            aspect_info: dict (如果 return_aspect_info=True)
        """
        batch_size = text_input_ids.size(0)

        # 1. BERT 編碼
        text_emb = self.bert_absa.bert_embedding(
            text_input_ids,
            attention_mask=text_attention_mask
        )  # [batch, seq_len, bert_dim]

        text_hidden = self.text_projection(text_emb)  # [batch, seq_len, hidden_dim]

        # 2. Implicit Aspect Discovery
        aspect_repr, aspect_importance, aspect_attn = self.aspect_discovery(
            text_hidden,
            text_attention_mask
        )
        # aspect_repr: [batch, num_aspects, hidden_dim]
        # aspect_importance: [batch, num_aspects]
        # aspect_attn: [batch, num_aspects, seq_len]

        # 3. AAHA 精煉（可選）
        if self.use_aaha:
            refined_aspects = []
            for i in range(self.num_implicit_aspects):
                ctx, _ = self.aaha(
                    text_hidden,
                    aspect_repr[:, i, :],
                    text_attention_mask.float()
                )
                refined_aspects.append(ctx)
            aspect_repr = torch.stack(refined_aspects, dim=1)

        # 4. PMAC 組合
        gate_stats = None
        if self.use_pmac:
            # 創建 aspect_mask (所有 aspects 都有效)
            aspect_mask = torch.ones(batch_size, self.num_implicit_aspects,
                                    dtype=torch.bool, device=text_input_ids.device)

            composed_features, pmac_outputs = self.pmac(aspect_repr, aspect_mask)

            # 如果是 Selective PMAC，提取 gate 統計
            if isinstance(self.pmac, SelectivePMACMultiAspect):
                gate_stats = pmac_outputs
        else:
            composed_features = aspect_repr

        # 5. IARM 關係建模
        if self.use_iarm:
            aspect_mask = torch.ones(batch_size, self.num_implicit_aspects,
                                    dtype=torch.bool, device=text_input_ids.device)
            enhanced_features, _ = self.iarm(composed_features, aspect_mask)
        else:
            enhanced_features = composed_features

        # 6. 融合為句子級別表示
        sentence_repr = self._fuse_aspects(enhanced_features, aspect_importance)
        # [batch, hidden_dim]

        # 7. 分類
        logits = self.classifier(sentence_repr)  # [batch, num_classes]

        if return_aspect_info:
            aspect_info = {
                'aspect_representations': aspect_repr,
                'aspect_importance': aspect_importance,
                'aspect_attention': aspect_attn,
                'aspect_names': self.aspect_discovery.get_aspect_names(),
                'gate_stats': gate_stats
            }
            return logits, aspect_info
        else:
            return logits, gate_stats

    def _fuse_aspects(
        self,
        aspect_features: torch.Tensor,
        aspect_importance: torch.Tensor
    ) -> torch.Tensor:
        """
        將多個 aspects 融合為句子級別表示

        參數:
            aspect_features: [batch, num_aspects, hidden_dim]
            aspect_importance: [batch, num_aspects]

        返回:
            sentence_repr: [batch, hidden_dim]
        """
        if self.fusion_strategy == 'mean':
            # 簡單平均
            return aspect_features.mean(dim=1)

        elif self.fusion_strategy == 'max':
            # Max pooling
            return aspect_features.max(dim=1)[0]

        elif self.fusion_strategy == 'weighted_pooling':
            # 加權平均（結合學習的權重和 importance）
            learned_weights = self.aspect_weights(aspect_features).squeeze(-1)
            # [batch, num_aspects]

            # 結合 implicit aspect discovery 的 importance
            combined_weights = learned_weights * aspect_importance.unsqueeze(-1)
            combined_weights = F.softmax(combined_weights, dim=1)

            # 加權求和
            weighted_features = aspect_features * combined_weights.unsqueeze(-1)
            return weighted_features.sum(dim=1)

        elif self.fusion_strategy == 'attention':
            # 注意力融合
            batch_size = aspect_features.size(0)
            query = self.fusion_query.unsqueeze(0).expand(batch_size, -1, -1)
            # [batch, 1, hidden_dim]

            fused, _ = self.fusion_attention(
                query=query,
                key=aspect_features,
                value=aspect_features
            )
            return fused.squeeze(1)  # [batch, hidden_dim]

        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")


def create_sentence_level_hmacnet(
    num_classes: int = 3,
    domain: str = 'generic',
    use_pmac: bool = True,
    use_iarm: bool = True,
    pmac_mode: str = 'selective',
    **kwargs
):
    """
    創建句子級別 HMAC-Net 的便捷函數

    參數:
        num_classes: 類別數（2=二分類, 3=三分類, 5=五分類等）
        domain: 領域類型（用於選擇預定義的 aspects）
        use_pmac: 是否使用 PMAC
        use_iarm: 是否使用 IARM
        pmac_mode: PMAC 模式
        **kwargs: 其他參數

    返回:
        HMACNetSentenceLevel 模型
    """
    return HMACNetSentenceLevel(
        num_classes=num_classes,
        domain=domain,
        use_pmac=use_pmac,
        use_iarm=use_iarm,
        pmac_composition_mode=pmac_mode,
        **kwargs
    )


if __name__ == '__main__':
    # 測試代碼
    print("測試句子級別 HMAC-Net...")

    batch_size = 4
    seq_len = 64
    num_classes = 2

    # 模擬輸入
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    # 創建模型
    model = HMACNetSentenceLevel(
        num_classes=num_classes,
        domain='movie_review',
        num_implicit_aspects=5,
        use_pmac=True,
        pmac_composition_mode='selective',
        use_iarm=True,
        fusion_strategy='weighted_pooling'
    )

    # Forward
    logits, aspect_info = model(input_ids, attention_mask, return_aspect_info=True)

    print(f"\n輸入形狀: {input_ids.shape}")
    print(f"輸出 logits: {logits.shape}")
    print(f"\nAspect 資訊:")
    print(f"  Aspect 表示: {aspect_info['aspect_representations'].shape}")
    print(f"  Aspect 重要性: {aspect_info['aspect_importance'].shape}")
    print(f"  Aspect 注意力: {aspect_info['aspect_attention'].shape}")
    print(f"  Aspect 名稱: {aspect_info['aspect_names']}")

    print("\n✓ 句子級別 HMAC-Net 測試通過！")
