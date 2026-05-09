"""
BERT 嵌入層
提供動態 BERT 嵌入，替代靜態 GloVe
支持 BERT 和 DeBERTa
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import (
    BertModel, BertTokenizer,
    DebertaModel, DebertaTokenizer,
    DebertaV2Model, DebertaV2Tokenizer,
    AutoModel, AutoTokenizer
)


class BERTEmbedding(nn.Module):
    """
    BERT 嵌入層

    功能:
        - 使用預訓練 BERT/DeBERTa 模型
        - 支援微調或凍結
        - 動態上下文嵌入
        - 比靜態 GloVe 效果更好

    支援的模型:
        - BERT: bert-base-uncased, bert-large-uncased
        - DeBERTa: microsoft/deberta-base, microsoft/deberta-v3-base
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        freeze: bool = False,
        pooling: str = 'mean',
        output_hidden_states: bool = False
    ):
        """
        初始化 BERT 嵌入層

        參數:
            model_name: 模型名稱 (bert-base-uncased, microsoft/deberta-base, etc.)
            freeze: 是否凍結參數
            pooling: 池化方式 ('mean', 'max', 'cls')
            output_hidden_states: 是否輸出所有層的hidden states（用於階層模型）
        """
        super(BERTEmbedding, self).__init__()

        self.model_name = model_name
        self.pooling = pooling
        self.output_hidden_states = output_hidden_states

        model_name_lower = model_name.lower()
        self.is_deberta = 'deberta' in model_name_lower
        self.is_deberta_v2 = 'deberta-v2' in model_name_lower or 'deberta-v3' in model_name_lower

        print(f"載入模型: {model_name}")
        if self.is_deberta_v2:
            self.bert = DebertaV2Model.from_pretrained(model_name)
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
            print("  類型: DeBERTa-v2/v3 (增強版)")
        elif self.is_deberta:
            self.bert = DebertaModel.from_pretrained(model_name)
            self.tokenizer = DebertaTokenizer.from_pretrained(model_name)
            print("  類型: DeBERTa (解耦注意力)")
        else:
            self.bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            print("  類型: BERT")

        if output_hidden_states:
            self.bert.config.output_hidden_states = True
            print("  階層模式: 啟用多層特徵提取")

        self.hidden_size = self.bert.config.hidden_size
        print(f"  隱藏層維度: {self.hidden_size}")

        if freeze:
            num_layers = self.bert.config.num_hidden_layers
            last_layers = 2

            for name, param in self.bert.named_parameters():
                param.requires_grad = False

                for layer_idx in range(num_layers - last_layers, num_layers):
                    if f"layer.{layer_idx}" in name:
                        param.requires_grad = True

                if not self.is_deberta and not self.is_deberta_v2:
                    if "pooler" in name:
                        param.requires_grad = True

            trainable_params = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.bert.parameters())
            print(f"  部分凍結: 可訓練參數 {trainable_params:,} / 總參數 {total_params:,}")
            print(f"  可訓練比例: {trainable_params/total_params*100:.1f}%")
        else:
            total_params = sum(p.numel() for p in self.bert.parameters())
            print(f"  完全可訓練: {total_params:,} 參數（微調模式）")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            input_ids: 輸入 token IDs [batch, seq_len]
            attention_mask: 注意力掩碼 [batch, seq_len]
            token_type_ids: Token type IDs [batch, seq_len] (用於區分 text 和 aspect)

        返回:
            BERT 嵌入 [batch, seq_len, hidden_size]
        """
        # BERT 編碼 - 不使用 token_type_ids (ABSA 任務中不使用效果更好)
        # 研究表明：在 ABSA 中，aspect 不是獨立句子，使用 token_type_ids 反而降低效果
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        sequence_output = outputs.last_hidden_state

        if self.output_hidden_states:
            all_hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors
            return sequence_output, all_hidden_states

        return sequence_output

    def encode_text(
        self,
        texts: list,
        max_length: int = 128,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        編碼文本（用於數據預處理）

        參數:
            texts: 文本列表
            max_length: 最大長度
            device: 設備

        返回:
            (input_ids, attention_mask)
        """
        encoded = self.tokenizer.batch_encode_plus(
            texts,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].to(device)
        attention_mask = encoded['attention_mask'].to(device)

        return input_ids, attention_mask

    def get_pooled_embedding(
        self,
        sequence_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        池化序列輸出得到句子嵌入

        參數:
            sequence_output: 序列輸出 [batch, seq_len, hidden_size]
            attention_mask: 注意力掩碼 [batch, seq_len]

        返回:
            池化後的嵌入 [batch, hidden_size]
        """
        if self.pooling == 'cls':
            return sequence_output[:, 0, :]

        elif self.pooling == 'mean':
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(sequence_output, dim=1)

        elif self.pooling == 'max':
            return torch.max(sequence_output, dim=1)[0]

        else:
            raise ValueError(f"不支援的池化方式: {self.pooling}")


class BERTForABSA(nn.Module):
    """
    專為 ABSA 任務設計的 BERT 模型
    可以處理面向和句子的聯合編碼
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False
    ):
        """
        初始化 BERT for ABSA

        參數:
            model_name: BERT 模型名稱
            freeze_bert: 是否凍結 BERT
        """
        super(BERTForABSA, self).__init__()

        self.bert_embedding = BERTEmbedding(
            model_name=model_name,
            freeze=freeze_bert
        )

        self.hidden_size = self.bert_embedding.hidden_size

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        聯合編碼句子和面向

        參數:
            text_input_ids: 句子 token IDs [batch, text_len]
            text_attention_mask: 句子注意力掩碼 [batch, text_len]
            aspect_input_ids: 面向 token IDs [batch, aspect_len]
            aspect_attention_mask: 面向注意力掩碼 [batch, aspect_len]

        返回:
            (句子嵌入 [batch, text_len, hidden_size],
             面向嵌入 [batch, aspect_len, hidden_size])
        """
        text_emb = self.bert_embedding(text_input_ids, text_attention_mask)
        aspect_emb = self.bert_embedding(aspect_input_ids, aspect_attention_mask)
        aspect_pooled = self.bert_embedding.get_pooled_embedding(
            aspect_emb, aspect_attention_mask
        )

        return text_emb, aspect_pooled


