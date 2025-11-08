"""
HMAC-Net: Hierarchical Multi-Aspect Composition Network
面向級情感分析的階層式多面向組合網路

整合 AAHA + PMAC + IARM 三個核心模組
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
sys.path.append('..')

from models.base_model import BaseModel, EmbeddingLayer, MLP
from models.aaha import AAHA
from models.pmac import PMAC
from models.iarm import IARM


class HMACNet(BaseModel):
    """
    HMAC-Net 完整模型

    架構:
        1. 嵌入層（GloVe/BERT）
        2. 雙向 LSTM 編碼器
        3. AAHA 模組（階層式注意力）
        4. PMAC 模組（多面向組合）
        5. IARM 模組（面向間關係）- 可選
        6. 分類器
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_classes: int = 3,
        num_layers: int = 2,
        dropout: float = 0.5,
        pretrained_embeddings: Optional[torch.Tensor] = None,
        freeze_embeddings: bool = False,
        # AAHA 參數
        word_attention_dim: int = 128,
        phrase_attention_dim: int = 128,
        sentence_attention_dim: int = 128,
        # PMAC 參數
        fusion_dim: int = 256,
        num_composition_layers: int = 2,
        fusion_method: str = "gated",
        # IARM 參數
        use_iarm: bool = True,
        relation_dim: int = 128,
        relation_type: str = "transformer",
        num_heads: int = 4,
        # 分類器參數
        classifier_hidden_dims: list = [128, 64],
        use_batch_norm: bool = True
    ):
        """
        初始化 HMAC-Net

        參數:
            vocab_size: 詞彙表大小
            embedding_dim: 嵌入維度
            hidden_dim: LSTM 隱藏維度
            num_classes: 分類類別數
            num_layers: LSTM 層數
            dropout: Dropout 比率
            pretrained_embeddings: 預訓練嵌入
            freeze_embeddings: 是否凍結嵌入層
            ... （其他模組參數見上方）
        """
        super(HMACNet, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_iarm = use_iarm

        # 1. 嵌入層
        self.embedding = EmbeddingLayer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            pretrained_embeddings=pretrained_embeddings.numpy() if pretrained_embeddings is not None else None,
            freeze=freeze_embeddings
        )

        # 2. 雙向 LSTM 編碼器
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 3. AAHA 模組（階層式注意力）
        self.aaha = AAHA(
            hidden_dim=hidden_dim,
            aspect_dim=hidden_dim,  # 面向也用 LSTM 編碼
            word_attention_dim=word_attention_dim,
            phrase_attention_dim=phrase_attention_dim,
            sentence_attention_dim=sentence_attention_dim,
            dropout=dropout
        )

        # 4. PMAC 模組（多面向組合）
        self.pmac = PMAC(
            input_dim=hidden_dim,
            fusion_dim=fusion_dim,
            num_composition_layers=num_composition_layers,
            fusion_method=fusion_method,
            dropout=dropout
        )

        # 5. IARM 模組（面向間關係建模） - 可選
        if use_iarm:
            self.iarm = IARM(
                input_dim=fusion_dim,
                relation_dim=relation_dim,
                relation_type=relation_type,
                num_heads=num_heads,
                num_layers=2,
                dropout=dropout
            )

        # 6. 分類器
        self.classifier = MLP(
            input_dim=fusion_dim,
            hidden_dims=classifier_hidden_dims,
            output_dim=num_classes,
            dropout=dropout,
            activation='relu',
            use_batch_norm=use_batch_norm
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化權重
        self.init_weights()

    def encode_sequence(
        self,
        text_indices: torch.Tensor,
        text_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        編碼文本序列

        參數:
            text_indices: 文本索引 [batch, seq_len]
            text_len: 實際長度 [batch]

        返回:
            編碼後的隱藏狀態 [batch, seq_len, hidden_dim]
        """
        # 嵌入 [batch, seq_len, embedding_dim]
        embedded = self.embedding(text_indices)
        embedded = self.dropout(embedded)

        # LSTM 編碼
        if text_len is not None:
            # 打包序列（處理變長序列）
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, text_len.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)

        return lstm_out

    def encode_aspect(
        self,
        aspect_indices: torch.Tensor,
        aspect_len: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        編碼面向

        參數:
            aspect_indices: 面向索引 [batch, aspect_len]
            aspect_len: 實際長度 [batch]

        返回:
            面向嵌入 [batch, hidden_dim]
        """
        # 嵌入 [batch, aspect_len, embedding_dim]
        embedded = self.embedding(aspect_indices)

        # 平均池化得到面向表示
        if aspect_len is not None:
            # 創建掩碼
            mask = torch.arange(aspect_indices.size(1), device=aspect_indices.device).unsqueeze(0)
            mask = mask < aspect_len.unsqueeze(1)
            mask = mask.unsqueeze(-1).float()

            # 加權平均
            aspect_embedding = (embedded * mask).sum(dim=1) / aspect_len.unsqueeze(1).float()
        else:
            # 簡單平均
            aspect_embedding = embedded.mean(dim=1)

        return aspect_embedding

    def forward(
        self,
        text_indices: torch.Tensor,
        aspect_indices: torch.Tensor,
        text_len: Optional[torch.Tensor] = None,
        aspect_len: Optional[torch.Tensor] = None,
        aspect_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        前向傳播

        參數:
            text_indices: 文本索引 [batch, seq_len]
            aspect_indices: 面向索引 [batch, aspect_len]
            text_len: 文本實際長度 [batch]
            aspect_len: 面向實際長度 [batch]
            aspect_mask: 面向掩碼 [batch, seq_len]
            return_attention: 是否返回注意力權重

        返回:
            (分類 logits [batch, num_classes], 注意力權重字典)
        """
        batch_size = text_indices.size(0)

        # 1. 編碼文本
        text_hidden = self.encode_sequence(text_indices, text_len)  # [batch, seq_len, hidden_dim]

        # 2. 編碼面向
        aspect_embedding = self.encode_aspect(aspect_indices, aspect_len)  # [batch, hidden_dim]

        # 3. AAHA 模組（階層式注意力）
        # 創建文本掩碼
        if text_len is not None:
            seq_len = text_indices.size(1)
            text_mask = torch.arange(seq_len, device=text_indices.device).unsqueeze(0)
            text_mask = text_mask < text_len.unsqueeze(1)
            text_mask = text_mask.float()
        else:
            text_mask = None

        context_vector, aaha_attention = self.aaha(
            text_hidden, aspect_embedding, text_mask
        )  # [batch, hidden_dim]

        # 4. PMAC 模組（多面向組合）
        # 這裡假設單個面向，如果有多個面向需要相應調整
        composed_repr = self.pmac(context_vector)  # [batch, fusion_dim]

        # 5. IARM 模組（面向間關係）- 可選
        iarm_attention = None
        if self.use_iarm:
            # 將單個面向擴展為 [batch, 1, fusion_dim]
            aspect_repr = composed_repr.unsqueeze(1)
            enhanced_repr, iarm_attention = self.iarm(aspect_repr)
            # 取回 [batch, fusion_dim]
            final_repr = enhanced_repr.squeeze(1)
        else:
            final_repr = composed_repr

        # 6. 分類器
        logits = self.classifier(final_repr)  # [batch, num_classes]

        # 收集注意力權重
        if return_attention:
            attention_weights = {
                'aaha': aaha_attention,
                'iarm': iarm_attention
            }
            return logits, attention_weights
        else:
            return logits, None


class HMACNetMultiAspect(BaseModel):
    """
    HMAC-Net 多面向版本
    處理一個句子包含多個面向的情況
    """

    def __init__(self, *args, **kwargs):
        """使用與 HMACNet 相同的參數"""
        super(HMACNetMultiAspect, self).__init__()

        # 內部使用單面向版本
        self.single_aspect_model = HMACNet(*args, **kwargs)

        # 提取參數
        self.use_iarm = self.single_aspect_model.use_iarm

    def forward(
        self,
        text_indices: torch.Tensor,
        aspect_indices: torch.Tensor,
        text_len: Optional[torch.Tensor] = None,
        aspect_len: Optional[torch.Tensor] = None,
        num_aspects: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        前向傳播（多面向）

        參數:
            text_indices: 文本索引 [batch, seq_len]
            aspect_indices: 面向索引 [batch, max_num_aspects, aspect_len]
            text_len: 文本長度 [batch]
            aspect_len: 面向長度 [batch, max_num_aspects]
            num_aspects: 每個樣本的實際面向數 [batch]
            return_attention: 是否返回注意力權重

        返回:
            (分類 logits [batch, max_num_aspects, num_classes], 注意力權重)
        """
        batch_size, max_num_aspects, aspect_seq_len = aspect_indices.size()

        # 收集所有面向的 logits
        all_logits = []
        all_attentions = []

        for i in range(max_num_aspects):
            # 獲取當前面向
            current_aspect = aspect_indices[:, i, :]  # [batch, aspect_len]
            current_aspect_len = aspect_len[:, i] if aspect_len is not None else None

            # 前向傳播
            logits, attention = self.single_aspect_model(
                text_indices=text_indices,
                aspect_indices=current_aspect,
                text_len=text_len,
                aspect_len=current_aspect_len,
                return_attention=return_attention
            )

            all_logits.append(logits)
            if return_attention:
                all_attentions.append(attention)

        # 堆疊 [batch, max_num_aspects, num_classes]
        logits_stacked = torch.stack(all_logits, dim=1)

        if return_attention:
            return logits_stacked, all_attentions
        else:
            return logits_stacked, None


if __name__ == "__main__":
    # 測試 HMAC-Net
    print("測試 HMAC-Net 模型...")

    vocab_size = 5000
    batch_size = 4
    seq_len = 50
    aspect_len = 5

    # 創建模型
    model = HMACNet(
        vocab_size=vocab_size,
        embedding_dim=300,
        hidden_dim=256,
        num_classes=3,
        num_layers=2,
        dropout=0.5,
        use_iarm=True,
        relation_type="transformer"
    )

    print(f"\n模型參數統計:")
    model.print_model_summary()

    # 模擬輸入
    text_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    aspect_indices = torch.randint(0, vocab_size, (batch_size, aspect_len))
    text_len = torch.tensor([50, 45, 48, 50])
    aspect_len = torch.tensor([5, 3, 4, 5])

    # 前向傳播
    logits, attention = model(
        text_indices=text_indices,
        aspect_indices=aspect_indices,
        text_len=text_len,
        aspect_len=aspect_len,
        return_attention=True
    )

    print(f"\n輸出:")
    print(f"  Logits 形狀: {logits.shape}")  # [batch, num_classes]
    print(f"  預測類別: {torch.argmax(logits, dim=1)}")

    print(f"\n注意力權重:")
    if attention['aaha'] is not None:
        print(f"  AAHA 詞級注意力: {attention['aaha']['word'].shape}")
        print(f"  AAHA 層級權重: {attention['aaha']['layer_weights'].shape}")

    print("\nHMAC-Net 測試完成！")
