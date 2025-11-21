"""
Baseline Models for ABSA

這個檔案包含標準 baseline 模型：
1. BERT_CLS: 標準 BERT baseline (使用 [CLS] token)

參考文獻：
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers
  for Language Understanding. NAACL 2019.

用途：
- 提供標準對照組
- 證明我們方法的有效性
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn

from models.bert_embedding import BERTForABSA
from models.base_model import BaseModel


class BERT_CLS_Baseline(BaseModel):
    """
    Baseline: BERT-CLS (標準 BERT baseline)

    架構:
        Text + Aspect → BERT → [CLS] → Classifier → Sentiment

    參考文獻:
        Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional
        Transformers for Language Understanding. NAACL 2019.

    特點:
        - ABSA 領域最標準的 baseline
        - 使用 BERT 的 [CLS] token 作為句子表示
        - 每個 aspect 獨立處理
        - 簡單、直接、可復現

    用途:
        - 作為標準對照組
        - 證明階層特徵提取和 Layer-wise Attention 的有效性
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super(BERT_CLS_Baseline, self).__init__()

        self.num_classes = num_classes

        # BERT 編碼器
        self.bert = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert.hidden_size

        # 簡單的分類器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_size, num_classes)
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
        """
        batch_size, max_aspects, aspect_len = aspect_input_ids.shape
        seq_len = text_input_ids.shape[1]

        logits_list = []

        for i in range(max_aspects):
            # 只處理有效的 aspects
            if aspect_mask[:, i].any():
                # 拼接 text 和 aspect: [CLS] text [SEP] aspect [SEP]
                # text_input_ids: [batch, seq_len]
                # aspect_input_ids[:, i, :]: [batch, aspect_len]

                combined_ids = torch.cat([
                    text_input_ids,
                    aspect_input_ids[:, i, :]
                ], dim=1)  # [batch, seq_len + aspect_len]

                combined_mask = torch.cat([
                    text_attention_mask,
                    aspect_attention_mask[:, i, :]
                ], dim=1)  # [batch, seq_len + aspect_len]

                # BERT encoding
                embeddings = self.bert.bert_embedding(
                    combined_ids,
                    attention_mask=combined_mask
                )  # [batch, seq_len + aspect_len, hidden_size]

                # 使用 [CLS] token (第一個 token)
                cls_token = embeddings[:, 0, :]  # [batch, hidden_size]

                # 分類
                logit = self.classifier(cls_token)  # [batch, num_classes]
                logits_list.append(logit)
            else:
                # 無效 aspect，填充零向量
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=text_input_ids.device)
                )

        # 堆疊所有 aspects 的 logits
        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]

        return logits, None  # None 是 gate_stats (baseline 沒有 gate)


def create_baseline(
    baseline_type: str,
    bert_model_name: str = 'distilbert-base-uncased',
    freeze_bert: bool = False,
    hidden_dim: int = 768,
    num_classes: int = 3,
    dropout: float = 0.1
):
    """
    創建 baseline 模型

    參數:
        baseline_type: 目前只支持 'bert_cls'
        其他參數與模型一致

    返回:
        baseline 模型實例
    """
    if baseline_type == 'bert_cls':
        return BERT_CLS_Baseline(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            num_classes=num_classes,
            dropout=dropout
        )

    # 保留向後兼容（bert_only 映射到 bert_cls）
    elif baseline_type == 'bert_only':
        return BERT_CLS_Baseline(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            num_classes=num_classes,
            dropout=dropout
        )

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}. Only 'bert_cls' is supported.")
