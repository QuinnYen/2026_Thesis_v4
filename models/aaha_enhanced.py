"""
增強版 AAHA 模組: Aspect-Aware Hierarchical Attention (Enhanced)
面向感知階層式注意力（增強版）

新增功能:
    1. Residual Connection - 跳過連接防止梯度消失
    2. Multi-scale Attention - 不同粒度的注意力頭
    3. Attention Dropout - 專門針對注意力權重的 dropout
    4. Layer Normalization - 穩定訓練
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


class MultiScaleAttention(nn.Module):
    """
    多尺度注意力機制

    使用多個注意力頭，每個頭關注不同粒度的資訊：
    - 細粒度頭：關注局部細節（小的 attention_dim）
    - 粗粒度頭：關注全局模式（大的 attention_dim）
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dims: List[int] = [64, 128, 256],  # 多尺度
        attention_dropout: float = 0.1,  # 注意力權重專用 dropout
        output_dropout: float = 0.3
    ):
        """
        初始化多尺度注意力

        參數:
            hidden_dim: 隱藏維度
            aspect_dim: 面向維度
            attention_dims: 多個注意力維度（不同尺度）
            attention_dropout: 注意力權重 dropout
            output_dropout: 輸出 dropout
        """
        super(MultiScaleAttention, self).__init__()

        self.num_heads = len(attention_dims)
        self.attention_dims = attention_dims

        # 為每個尺度創建投影層
        self.hidden_projections = nn.ModuleList([
            nn.Linear(hidden_dim, attn_dim) for attn_dim in attention_dims
        ])

        self.aspect_projections = nn.ModuleList([
            nn.Linear(aspect_dim, attn_dim) for attn_dim in attention_dims
        ])

        # 為每個尺度創建注意力權重層
        self.attention_weights = nn.ModuleList([
            nn.Linear(attn_dim, 1, bias=False) for attn_dim in attention_dims
        ])

        # 融合多尺度輸出
        self.fusion = nn.Linear(hidden_dim * self.num_heads, hidden_dim)

        # Attention dropout（針對注意力權重）
        self.attention_dropout = nn.Dropout(attention_dropout)

        # 輸出 dropout
        self.output_dropout = nn.Dropout(output_dropout)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

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
            hidden_states: [batch, seq_len, hidden_dim]
            aspect_embedding: [batch, aspect_dim]
            mask: [batch, seq_len]

        返回:
            (上下文向量 [batch, hidden_dim], 注意力權重 [batch, num_heads, seq_len])
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()

        contexts = []
        all_attention_weights = []

        # 對每個尺度計算注意力
        for i in range(self.num_heads):
            # 投影隱藏狀態
            hidden_proj = self.hidden_projections[i](hidden_states)  # [batch, seq_len, attn_dim]

            # 投影面向嵌入
            aspect_proj = self.aspect_projections[i](aspect_embedding)  # [batch, attn_dim]

            # 擴展面向嵌入
            aspect_proj_expanded = aspect_proj.unsqueeze(1).expand(-1, seq_len, -1)

            # 融合
            combined = self.tanh(hidden_proj + aspect_proj_expanded)

            # 計算注意力分數
            attention_scores = self.attention_weights[i](combined)  # [batch, seq_len, 1]

            # 應用掩碼
            if mask is not None:
                attention_scores = attention_scores.masked_fill(
                    mask.unsqueeze(-1) == 0, -1e9
                )

            # 計算注意力權重
            attention_weights = F.softmax(attention_scores, dim=1)  # [batch, seq_len, 1]

            # *** 關鍵改進：Attention Dropout ***
            # 在注意力權重上應用 dropout，而不是在輸出上
            attention_weights = self.attention_dropout(attention_weights)

            # 加權求和
            context = torch.sum(hidden_states * attention_weights, dim=1)  # [batch, hidden_dim]

            contexts.append(context)
            all_attention_weights.append(attention_weights.squeeze(-1))

        # 拼接多尺度上下文
        multi_scale_context = torch.cat(contexts, dim=1)  # [batch, hidden_dim * num_heads]

        # 融合
        fused_context = self.fusion(multi_scale_context)  # [batch, hidden_dim]

        # Layer Normalization
        fused_context = self.layer_norm(fused_context)

        # 輸出 dropout
        fused_context = self.output_dropout(fused_context)

        # 堆疊注意力權重
        stacked_weights = torch.stack(all_attention_weights, dim=1)  # [batch, num_heads, seq_len]

        return fused_context, stacked_weights


class ResidualAttentionBlock(nn.Module):
    """
    帶殘差連接的注意力塊

    結構: Input -> Attention -> Residual + LayerNorm -> FFN -> Residual + LayerNorm
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dims: List[int] = [64, 128, 256],
        attention_dropout: float = 0.1,
        output_dropout: float = 0.3,
        ffn_hidden_dim: int = 512
    ):
        """
        初始化殘差注意力塊

        參數:
            hidden_dim: 隱藏維度
            aspect_dim: 面向維度
            attention_dims: 多尺度注意力維度
            attention_dropout: 注意力權重 dropout
            output_dropout: 輸出 dropout
            ffn_hidden_dim: Feed-Forward 網絡隱藏維度
        """
        super(ResidualAttentionBlock, self).__init__()

        # 多尺度注意力
        self.attention = MultiScaleAttention(
            hidden_dim, aspect_dim, attention_dims,
            attention_dropout, output_dropout
        )

        # Layer Normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Dropout(output_dropout),
            nn.Linear(ffn_hidden_dim, hidden_dim),
            nn.Dropout(output_dropout)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播（帶殘差連接）

        參數:
            hidden_states: [batch, seq_len, hidden_dim]
            aspect_embedding: [batch, aspect_dim]
            mask: [batch, seq_len]

        返回:
            (上下文向量, 注意力權重)
        """
        # 計算平均池化作為原始表示（用於殘差連接）
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            count = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
            residual = sum_hidden / count  # [batch, hidden_dim]
        else:
            residual = torch.mean(hidden_states, dim=1)  # [batch, hidden_dim]

        # *** 關鍵改進 1: 注意力 + 殘差連接 ***
        attn_output, attn_weights = self.attention(hidden_states, aspect_embedding, mask)
        attn_output = self.ln1(attn_output + residual)  # Residual connection

        # *** 關鍵改進 2: FFN + 殘差連接 ***
        ffn_output = self.ffn(attn_output)
        final_output = self.ln2(ffn_output + attn_output)  # Residual connection

        return final_output, attn_weights


class WordLevelAttentionEnhanced(nn.Module):
    """增強版詞級注意力（帶殘差連接）"""

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dims: List[int] = [64, 128],
        attention_dropout: float = 0.1,
        output_dropout: float = 0.3
    ):
        super(WordLevelAttentionEnhanced, self).__init__()

        self.attention_block = ResidualAttentionBlock(
            hidden_dim, aspect_dim, attention_dims,
            attention_dropout, output_dropout
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.attention_block(hidden_states, aspect_embedding, mask)


class PhraseLevelAttentionEnhanced(nn.Module):
    """增強版片語級注意力（多尺度 CNN + 殘差連接）"""

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dims: List[int] = [64, 128, 256],
        kernel_sizes: list = [3, 5, 7],  # 多尺度卷積
        attention_dropout: float = 0.1,
        output_dropout: float = 0.3
    ):
        super(PhraseLevelAttentionEnhanced, self).__init__()

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

        # 融合多尺度 CNN 特徵
        self.cnn_fusion = nn.Linear(hidden_dim * len(kernel_sizes), hidden_dim)
        self.cnn_ln = nn.LayerNorm(hidden_dim)

        # 殘差注意力塊
        self.attention_block = ResidualAttentionBlock(
            hidden_dim, aspect_dim, attention_dims,
            attention_dropout, output_dropout
        )

        self.relu = nn.GELU()
        self.dropout = nn.Dropout(output_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # *** 殘差連接的起點 ***
        residual = hidden_states

        # 轉置以適配 Conv1d
        hidden_transposed = hidden_states.transpose(1, 2)  # [batch, hidden_dim, seq_len]

        # 多尺度卷積
        conv_outputs = []
        for conv in self.convs:
            conv_out = self.relu(conv(hidden_transposed))
            conv_outputs.append(conv_out)

        # 拼接
        concat_features = torch.cat(conv_outputs, dim=1)
        concat_features = concat_features.transpose(1, 2)  # [batch, seq_len, hidden_dim * K]

        # 融合
        phrase_features = self.cnn_fusion(concat_features)  # [batch, seq_len, hidden_dim]

        # *** 殘差連接 ***
        phrase_features = self.cnn_ln(phrase_features + residual)
        phrase_features = self.dropout(phrase_features)

        # 應用注意力
        context_vector, attention_weights = self.attention_block(
            phrase_features, aspect_embedding, mask
        )

        return context_vector, attention_weights


class SentenceLevelAttentionEnhanced(nn.Module):
    """增強版句子級注意力（LSTM + 殘差連接）"""

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        attention_dims: List[int] = [64, 128, 256],
        attention_dropout: float = 0.1,
        output_dropout: float = 0.3
    ):
        super(SentenceLevelAttentionEnhanced, self).__init__()

        # 雙向 LSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        # Layer Normalization
        self.lstm_ln = nn.LayerNorm(hidden_dim)

        # 殘差注意力塊
        self.attention_block = ResidualAttentionBlock(
            hidden_dim, aspect_dim, attention_dims,
            attention_dropout, output_dropout
        )

        self.dropout = nn.Dropout(output_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # *** 殘差連接的起點 ***
        residual = hidden_states

        # LSTM
        lstm_output, _ = self.lstm(hidden_states)

        # *** 殘差連接 ***
        lstm_output = self.lstm_ln(lstm_output + residual)
        lstm_output = self.dropout(lstm_output)

        # 應用注意力
        context_vector, attention_weights = self.attention_block(
            lstm_output, aspect_embedding, mask
        )

        return context_vector, attention_weights


class AAHAEnhanced(nn.Module):
    """
    增強版 AAHA 模組

    改進:
        1. ✅ Residual Connection - 每層都有跳過連接
        2. ✅ Multi-scale Attention - 不同粒度的注意力頭 (64, 128, 256)
        3. ✅ Attention Dropout - 專門針對注意力權重的 dropout
        4. ✅ Layer Normalization - 穩定訓練
        5. ✅ GELU 激活函數 - 比 ReLU 更好的性能
    """

    def __init__(
        self,
        hidden_dim: int,
        aspect_dim: int,
        word_attention_dims: List[int] = [64, 128],
        phrase_attention_dims: List[int] = [64, 128, 256],
        sentence_attention_dims: List[int] = [64, 128, 256],
        attention_dropout: float = 0.1,  # 注意力權重 dropout
        output_dropout: float = 0.3       # 輸出 dropout
    ):
        """
        初始化增強版 AAHA 模組

        參數:
            hidden_dim: 隱藏維度
            aspect_dim: 面向維度
            word_attention_dims: 詞級多尺度注意力維度
            phrase_attention_dims: 片語級多尺度注意力維度
            sentence_attention_dims: 句子級多尺度注意力維度
            attention_dropout: 注意力權重 dropout
            output_dropout: 輸出 dropout
        """
        super(AAHAEnhanced, self).__init__()

        # 三層增強注意力
        self.word_attention = WordLevelAttentionEnhanced(
            hidden_dim, aspect_dim, word_attention_dims,
            attention_dropout, output_dropout
        )

        self.phrase_attention = PhraseLevelAttentionEnhanced(
            hidden_dim, aspect_dim, phrase_attention_dims,
            attention_dropout=attention_dropout,
            output_dropout=output_dropout
        )

        self.sentence_attention = SentenceLevelAttentionEnhanced(
            hidden_dim, aspect_dim, sentence_attention_dims,
            attention_dropout, output_dropout
        )

        # 層級權重學習（動態組合三層特徵）
        self.layer_weight = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(output_dropout),
            nn.Linear(hidden_dim, 3)
        )

        # 最終融合（帶殘差連接）
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(output_dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Layer Normalization
        self.final_ln = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(output_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        aspect_embedding: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        前向傳播

        參數:
            hidden_states: [batch, seq_len, hidden_dim]
            aspect_embedding: [batch, aspect_dim]
            mask: [batch, seq_len]

        返回:
            (融合的上下文向量, 注意力權重字典)
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

        # 拼接三層特徵
        concat_features = torch.cat([word_context, phrase_context, sentence_context], dim=1)

        # 計算層級權重
        layer_weights = F.softmax(self.layer_weight(concat_features), dim=1)  # [batch, 3]

        # 加權融合三層特徵
        weighted_word = word_context * layer_weights[:, 0:1]
        weighted_phrase = phrase_context * layer_weights[:, 1:2]
        weighted_sentence = sentence_context * layer_weights[:, 2:3]

        weighted_concat = torch.cat([
            weighted_word, weighted_phrase, weighted_sentence
        ], dim=1)

        # *** 關鍵改進：最終融合 + 殘差連接 ***
        # 計算平均池化作為殘差
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            sum_hidden = torch.sum(hidden_states * mask_expanded, dim=1)
            count = torch.sum(mask_expanded, dim=1).clamp(min=1e-9)
            residual = sum_hidden / count
        else:
            residual = torch.mean(hidden_states, dim=1)

        # 融合
        final_context = self.fusion(weighted_concat)

        # 殘差連接 + Layer Normalization
        final_context = self.final_ln(final_context + residual)
        final_context = self.dropout(final_context)

        # 收集所有注意力權重
        attention_weights = {
            'word': word_weights,
            'phrase': phrase_weights,
            'sentence': sentence_weights,
            'layer_weights': layer_weights
        }

        return final_context, attention_weights


if __name__ == "__main__":
    print("測試增強版 AAHA 模組...")

    batch_size = 4
    seq_len = 20
    hidden_dim = 256
    aspect_dim = 256

    # 創建增強版 AAHA
    aaha_enhanced = AAHAEnhanced(
        hidden_dim=hidden_dim,
        aspect_dim=aspect_dim,
        word_attention_dims=[64, 128],
        phrase_attention_dims=[64, 128, 256],
        sentence_attention_dims=[64, 128, 256],
        attention_dropout=0.1,
        output_dropout=0.3
    )

    # 模擬輸入
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    aspect_embedding = torch.randn(batch_size, aspect_dim)
    mask = torch.ones(batch_size, seq_len)

    # 前向傳播
    context, attention_weights = aaha_enhanced(hidden_states, aspect_embedding, mask)

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

    # 計算參數量
    total_params = sum(p.numel() for p in aaha_enhanced.parameters())
    print(f"\n總參數量: {total_params:,}")

    print("\n✅ 增強版 AAHA 測試完成!")
    print("\n改進總結:")
    print("  ✅ 1. Residual Connection - 每層都有跳過連接")
    print("  ✅ 2. Multi-scale Attention - 3個不同粒度的注意力頭")
    print("  ✅ 3. Attention Dropout - 注意力權重專用 dropout (0.1)")
    print("  ✅ 4. Layer Normalization - 穩定訓練")
    print("  ✅ 5. GELU 激活函數 - 更好的非線性")
