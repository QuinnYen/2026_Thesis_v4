"""
BERT 嵌入層
提供動態 BERT 嵌入，替代靜態 GloVe
支持 BERT 和 DistilBERT
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import BertModel, BertTokenizer, DistilBertModel, DistilBertTokenizer, AutoModel, AutoTokenizer


class BERTEmbedding(nn.Module):
    """
    BERT 嵌入層

    功能:
        - 使用預訓練 BERT 模型
        - 支援微調或凍結
        - 動態上下文嵌入
        - 比靜態 GloVe 效果更好
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        freeze: bool = False,
        pooling: str = 'mean'
    ):
        """
        初始化 BERT 嵌入層

        參數:
            model_name: BERT 模型名稱
            freeze: 是否凍結 BERT 參數
            pooling: 池化方式 ('mean', 'max', 'cls')
        """
        super(BERTEmbedding, self).__init__()

        self.model_name = model_name
        self.pooling = pooling

        # 檢測模型類型
        self.is_distilbert = 'distilbert' in model_name.lower()

        # 載入模型和分詞器（使用 AutoModel 自動識別）
        print(f"載入模型: {model_name}")
        if self.is_distilbert:
            self.bert = DistilBertModel.from_pretrained(model_name)
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            print("  類型: DistilBERT (輕量級)")
        else:
            self.bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            print("  類型: BERT")

        # 獲取隱藏層大小
        self.hidden_size = self.bert.config.hidden_size
        print(f"  隱藏層維度: {self.hidden_size}")

        # 凍結參數策略
        if freeze:
            # 獲取總層數
            if self.is_distilbert:
                num_layers = self.bert.config.n_layers  # DistilBERT 有 6 層
                last_layers = 2  # 訓練最後 2 層
            else:
                num_layers = self.bert.config.num_hidden_layers  # BERT 有 12 層
                last_layers = 2  # 訓練最後 2 層

            # 凍結除了最後幾層的所有層
            for name, param in self.bert.named_parameters():
                param.requires_grad = False

                # 訓練最後幾層
                for layer_idx in range(num_layers - last_layers, num_layers):
                    if f"layer.{layer_idx}" in name or f"transformer.layer.{layer_idx}" in name:
                        param.requires_grad = True

                # BERT 的 pooler（DistilBERT 沒有）
                if not self.is_distilbert and "pooler" in name:
                    param.requires_grad = True

            # 計算可訓練參數數量
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
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            input_ids: 輸入 token IDs [batch, seq_len]
            attention_mask: 注意力掩碼 [batch, seq_len]

        返回:
            BERT 嵌入 [batch, seq_len, hidden_size]
        """
        # BERT 編碼
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 獲取序列輸出 [batch, seq_len, hidden_size]
        sequence_output = outputs.last_hidden_state

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
        # 分詞
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
            # 使用 [CLS] token
            return sequence_output[:, 0, :]

        elif self.pooling == 'mean':
            # 平均池化（考慮掩碼）
            if attention_mask is not None:
                # 擴展掩碼維度
                mask_expanded = attention_mask.unsqueeze(-1).expand(sequence_output.size()).float()
                sum_embeddings = torch.sum(sequence_output * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(sequence_output, dim=1)

        elif self.pooling == 'max':
            # 最大池化
            return torch.max(sequence_output, dim=1)[0]

        else:
            raise ValueError(f"不支援的池化方式: {self.pooling}")


class HybridEmbedding(nn.Module):
    """
    混合嵌入層
    結合 BERT 和傳統詞嵌入的優勢
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        vocab_size: int = 10000,
        static_embedding_dim: int = 300,
        freeze_bert: bool = False,
        use_static: bool = True
    ):
        """
        初始化混合嵌入層

        參數:
            bert_model_name: BERT 模型名稱
            vocab_size: 靜態詞彙表大小
            static_embedding_dim: 靜態嵌入維度
            freeze_bert: 是否凍結 BERT
            use_static: 是否使用靜態嵌入
        """
        super(HybridEmbedding, self).__init__()

        # BERT 嵌入
        self.bert_embedding = BERTEmbedding(
            model_name=bert_model_name,
            freeze=freeze_bert
        )

        self.use_static = use_static

        # 靜態嵌入（可選）
        if use_static:
            self.static_embedding = nn.Embedding(
                vocab_size,
                static_embedding_dim,
                padding_idx=0
            )

            # 投影層（將靜態嵌入投影到 BERT 維度）
            self.static_projection = nn.Linear(
                static_embedding_dim,
                self.bert_embedding.hidden_size
            )

        # 融合層
        if use_static:
            self.fusion = nn.Linear(
                self.bert_embedding.hidden_size * 2,
                self.bert_embedding.hidden_size
            )

        self.hidden_size = self.bert_embedding.hidden_size

    def forward(
        self,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
        static_input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向傳播

        參數:
            bert_input_ids: BERT token IDs [batch, seq_len]
            bert_attention_mask: BERT 注意力掩碼 [batch, seq_len]
            static_input_ids: 靜態詞嵌入索引 [batch, seq_len]

        返回:
            融合的嵌入 [batch, seq_len, hidden_size]
        """
        # BERT 嵌入
        bert_emb = self.bert_embedding(bert_input_ids, bert_attention_mask)

        if not self.use_static or static_input_ids is None:
            return bert_emb

        # 靜態嵌入
        static_emb = self.static_embedding(static_input_ids)
        static_emb_proj = self.static_projection(static_emb)

        # 融合
        combined = torch.cat([bert_emb, static_emb_proj], dim=-1)
        fused_emb = self.fusion(combined)

        return fused_emb


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
        # 編碼句子
        text_emb = self.bert_embedding(text_input_ids, text_attention_mask)

        # 編碼面向
        aspect_emb = self.bert_embedding(aspect_input_ids, aspect_attention_mask)

        # 池化面向嵌入得到面向表示
        aspect_pooled = self.bert_embedding.get_pooled_embedding(
            aspect_emb, aspect_attention_mask
        )

        return text_emb, aspect_pooled


if __name__ == "__main__":
    # 測試 BERT 嵌入
    print("測試 BERT 嵌入層...")

    # 創建 BERT 嵌入層
    bert_emb = BERTEmbedding(
        model_name='bert-base-uncased',
        freeze=False
    )

    print(f"\nBERT 隱藏層大小: {bert_emb.hidden_size}")

    # 測試文本編碼
    texts = [
        "The food was great but the service was terrible.",
        "I love this restaurant!"
    ]

    input_ids, attention_mask = bert_emb.encode_text(texts, max_length=20)

    print(f"\nInput IDs 形狀: {input_ids.shape}")
    print(f"Attention Mask 形狀: {attention_mask.shape}")

    # 前向傳播
    embeddings = bert_emb(input_ids, attention_mask)
    print(f"BERT 嵌入形狀: {embeddings.shape}")

    # 池化
    pooled = bert_emb.get_pooled_embedding(embeddings, attention_mask)
    print(f"池化後形狀: {pooled.shape}")

    # 測試 BERT for ABSA
    print("\n\n測試 BERT for ABSA...")
    bert_absa = BERTForABSA(freeze_bert=False)

    aspect_texts = ["food", "service"]
    aspect_ids, aspect_mask = bert_emb.encode_text(aspect_texts, max_length=10)

    text_emb, aspect_emb = bert_absa(
        input_ids, attention_mask,
        aspect_ids, aspect_mask
    )

    print(f"句子嵌入形狀: {text_emb.shape}")
    print(f"面向嵌入形狀: {aspect_emb.shape}")

    print("\nBERT 嵌入層測試完成！")
