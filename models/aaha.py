"""
AAHA 模組: Aspect-Aware Hierarchical Attention
面向感知階層式注意力

功能:
    - 三層階層式注意力機制（詞級、片語級、句子級）
    - 每層都融合面向資訊
    - 動態調整不同層級的注意力權重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AspectAwareAttention(nn.Module):
    """
    面向感知注意力
    基礎注意力模組，考慮面向資訊
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dim: int,
        dropout: float = 0.3
    ):
        """
        初始化面向感知注意力

        參數:
            hidden_dim: 隱藏狀態維度
            aspect_dim: 面向嵌入維度
            attention_dim: 注意力維度
            dropout: Dropout 比率
        """
        super(AspectAwareAttention, self).__init__()

        # 將隱藏狀態投影到注意力空間
        self.hidden_projection = nn.Linear(hidden_dim, attention_dim)

        # 將面向嵌入投影到注意力空間
        self.aspect_projection = nn.Linear(aspect_dim, attention_dim)

        # 注意力權重計算
        self.attention_weight = nn.Linear(attention_dim, 1, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            hidden_states: 隱藏狀態 [batch, seq_len, hidden_dim]
            aspect_embedding: 面向嵌入 [batch, aspect_dim]
            mask: 注意力掩碼 [batch, seq_len]

        返回:
            (上下文向量 [batch, hidden_dim], 注意力權重 [batch, seq_len])
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()

        # 投影隱藏狀態 [batch, seq_len, attention_dim]
        hidden_proj = self.hidden_projection(hidden_states)

        # 投影面向嵌入 [batch, attention_dim]
        aspect_proj = self.aspect_projection(aspect_embedding)

        # 擴展面向嵌入到序列長度 [batch, seq_len, attention_dim]
        aspect_proj_expanded = aspect_proj.unsqueeze(1).expand(-1, seq_len, -1)

        # 融合隱藏狀態和面向資訊 [batch, seq_len, attention_dim]
        combined = self.tanh(hidden_proj + aspect_proj_expanded)

        # 計算注意力分數 [batch, seq_len, 1]
        attention_scores = self.attention_weight(combined)

        # 應用掩碼（如果有）
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # 計算注意力權重 [batch, seq_len, 1]
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)

        # 加權求和得到上下文向量 [batch, hidden_dim]
        context_vector = torch.sum(hidden_states * attention_weights, dim=1)

        return context_vector, attention_weights.squeeze(-1)


class WordLevelAttention(nn.Module):
    """
    詞級注意力
    關注單個詞與面向的關聯
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dim: int,
        dropout: float = 0.3
    ):
        """初始化詞級注意力"""
        super(WordLevelAttention, self).__init__()

        self.attention = AspectAwareAttention(
            hidden_dim, aspect_dim, attention_dim, dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            hidden_states: 隱藏狀態 [batch, seq_len, hidden_dim]
            aspect_embedding: 面向嵌入 [batch, aspect_dim]
            mask: 掩碼 [batch, seq_len]

        返回:
            (詞級上下文向量, 詞級注意力權重)
        """
        return self.attention(hidden_states, aspect_embedding, mask)


class PhraseLevelAttention(nn.Module):
    """
    片語級注意力
    使用 CNN 提取局部片語特徵，再應用注意力
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dim: int,
        kernel_sizes: list = [3, 5],
        dropout: float = 0.3
    ):
        """
        初始化片語級注意力

        參數:
            hidden_dim: 隱藏維度
            aspect_dim: 面向維度
            attention_dim: 注意力維度
            kernel_sizes: CNN 卷積核大小列表
            dropout: Dropout 比率
        """
        super(PhraseLevelAttention, self).__init__()

        self.kernel_sizes = kernel_sizes

        # 多尺度卷積
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=k,
                padding=k // 2
            )
            for k in kernel_sizes
        ])

        # 融合多尺度特徵
        self.fusion = nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim)

        # 注意力
        self.attention = AspectAwareAttention(
            hidden_dim, aspect_dim, attention_dim, dropout
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            hidden_states: 隱藏狀態 [batch, seq_len, hidden_dim]
            aspect_embedding: 面向嵌入 [batch, aspect_dim]
            mask: 掩碼 [batch, seq_len]

        返回:
            (片語級上下文向量, 片語級注意力權重)
        """
        # 轉置以適配 Conv1d [batch, hidden_dim, seq_len]
        hidden_transposed = hidden_states.transpose(1, 2)

        # 多尺度卷積
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(hidden_transposed))  # [batch, hidden_dim, seq_len]
            conv_outputs.append(conv_out)

        # 拼接多尺度特徵 [batch, hidden_dim * num_kernels, seq_len]
        concat_features = torch.cat(conv_outputs, dim=1)

        # 轉回 [batch, seq_len, hidden_dim * num_kernels]
        concat_features = concat_features.transpose(1, 2)

        # 融合多尺度特徵 [batch, seq_len, hidden_dim]
        phrase_features = self.fusion(concat_features)
        phrase_features = self.dropout(phrase_features)

        # 應用注意力
        context_vector, attention_weights = self.attention(
            phrase_features, aspect_embedding, mask
        )

        return context_vector, attention_weights


class SentenceLevelAttention(nn.Module):
    """
    句子級注意力
    使用雙向 LSTM 捕捉全局資訊，再應用注意力
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dim: int,
        dropout: float = 0.3
    ):
        """
        初始化句子級注意力

        參數:
            hidden_dim: 隱藏維度
            aspect_dim: 面向維度
            attention_dim: 注意力維度
            dropout: Dropout 比率
        """
        super(SentenceLevelAttention, self).__init__()

        # 雙向 LSTM 用於捕捉全局資訊
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0  # 單層 LSTM 不需要 dropout
        )

        # 注意力
        self.attention = AspectAwareAttention(
            hidden_dim, aspect_dim, attention_dim, dropout
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            hidden_states: 隱藏狀態 [batch, seq_len, hidden_dim]
            aspect_embedding: 面向嵌入 [batch, aspect_dim]
            mask: 掩碼 [batch, seq_len]

        返回:
            (句子級上下文向量, 句子級注意力權重)
        """
        # 通過 LSTM [batch, seq_len, hidden_dim]
        lstm_output, _ = self.lstm(hidden_states)
        lstm_output = self.dropout(lstm_output)

        # 應用注意力
        context_vector, attention_weights = self.attention(
            lstm_output, aspect_embedding, mask
        )

        return context_vector, attention_weights


class AAHA(nn.Module):
    """
    AAHA 模組: Aspect-Aware Hierarchical Attention
    面向感知階層式注意力

    整合詞級、片語級、句子級三層注意力
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        word_attention_dim: int = 128,
        phrase_attention_dim: int = 128,
        sentence_attention_dim: int = 128,
        dropout: float = 0.3
    ):
        """
        初始化 AAHA 模組

        參數:
            hidden_dim: 隱藏維度
            aspect_dim: 面向維度
            word_attention_dim: 詞級注意力維度
            phrase_attention_dim: 片語級注意力維度
            sentence_attention_dim: 句子級注意力維度
            dropout: Dropout 比率
        """
        super(AAHA, self).__init__()

        # 三層注意力
        self.word_attention = WordLevelAttention(
            hidden_dim, aspect_dim, word_attention_dim, dropout
        )

        self.phrase_attention = PhraseLevelAttention(
            hidden_dim, aspect_dim, phrase_attention_dim, dropout=dropout
        )

        self.sentence_attention = SentenceLevelAttention(
            hidden_dim, aspect_dim, sentence_attention_dim, dropout
        )

        # 層級權重學習（動態組合三層特徵）
        self.layer_weight = nn.Linear(hidden_dim * 3, 3)

        # 最終融合
        self.fusion = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向傳播

        參數:
            hidden_states: 隱藏狀態 [batch, seq_len, hidden_dim]
            aspect_embedding: 面向嵌入 [batch, aspect_dim]
            mask: 掩碼 [batch, seq_len]

        返回:
            (融合的上下文向量 [batch, hidden_dim], 注意力權重字典)
        """
        # 詞級注意力
        word_context, word_weights = self.word_attention(
            hidden_states, aspect_embedding, mask
        )

        # 片語級注意力
        phrase_context, phrase_weights = self.phrase_attention(
            hidden_states, aspect_embedding, mask
        )

        # 句子級注意力
        sentence_context, sentence_weights = self.sentence_attention(
            hidden_states, aspect_embedding, mask
        )

        # 拼接三層特徵 [batch, hidden_dim * 3]
        concat_features = torch.cat([word_context, phrase_context, sentence_context], dim=1)

        # 計算層級權重 [batch, 3]
        layer_weights = F.softmax(self.layer_weight(concat_features), dim=1)

        # 加權融合三層特徵
        weighted_word = word_context * layer_weights[:, 0:1]
        weighted_phrase = phrase_context * layer_weights[:, 1:2]
        weighted_sentence = sentence_context * layer_weights[:, 2:3]

        # 最終融合 [batch, hidden_dim]
        final_context = self.fusion(torch.cat([
            weighted_word, weighted_phrase, weighted_sentence
        ], dim=1))

        final_context = self.dropout(final_context)

        # 收集所有注意力權重（用於視覺化）
        attention_weights = {
            'word': word_weights,
            'phrase': phrase_weights,
            'sentence': sentence_weights,
            'layer_weights': layer_weights
        }

        return final_context, attention_weights


if __name__ == "__main__":
    # 測試 AAHA 模組
    print("測試 AAHA 模組...")

    batch_size = 4
    seq_len = 20
    hidden_dim = 256
    aspect_dim = 256

    # 創建 AAHA 模組
    aaha = AAHA(
        hidden_dim=hidden_dim,
        aspect_dim=aspect_dim,
        word_attention_dim=128,
        phrase_attention_dim=128,
        sentence_attention_dim=128,
        dropout=0.3
    )

    # 模擬輸入
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    aspect_embedding = torch.randn(batch_size, aspect_dim)
    mask = torch.ones(batch_size, seq_len)  # 全為有效

    # 前向傳播
    context, attention_weights = aaha(hidden_states, aspect_embedding, mask)

    print(f"\n輸入形狀:")
    print(f"  hidden_states: {hidden_states.shape}")
    print(f"  aspect_embedding: {aspect_embedding.shape}")

    print(f"\n輸出形狀:")
    print(f"  context: {context.shape}")

    print(f"\n注意力權重形狀:")
    print(f"  word_weights: {attention_weights['word'].shape}")
    print(f"  phrase_weights: {attention_weights['phrase'].shape}")
    print(f"  sentence_weights: {attention_weights['sentence'].shape}")
    print(f"  layer_weights: {attention_weights['layer_weights'].shape}")

    print(f"\n層級權重範例:")
    print(attention_weights['layer_weights'][0])

    print("\nAAHA 模組測試完成！")
