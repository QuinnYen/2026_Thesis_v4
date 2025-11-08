"""
基礎模型抽象類
定義所有模型的通用介面和基礎功能
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple
import numpy as np


class BaseModel(ABC, nn.Module):
    """
    基礎模型抽象類

    功能:
        - 定義模型的基本介面
        - 提供通用的模型方法（參數初始化、保存/載入等）
        - 強制子類實作 forward 方法
    """

    def __init__(self):
        """初始化基礎模型"""
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        前向傳播（必須由子類實作）

        返回:
            模型輸出
        """
        raise NotImplementedError("子類必須實作 forward 方法")

    def init_weights(self):
        """
        初始化模型權重
        使用 Xavier 均勻初始化
        """
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def count_parameters(self) -> int:
        """
        計算模型總參數量

        返回:
            可訓練參數總數
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameters_info(self) -> Dict[str, int]:
        """
        獲取模型參數資訊

        返回:
            參數統計字典
        """
        total_params = 0
        trainable_params = 0

        for param in self.parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return {
            'total': total_params,
            'trainable': trainable_params,
            'non_trainable': total_params - trainable_params
        }

    def save_model(self, save_path: str, **kwargs):
        """
        保存模型

        參數:
            save_path: 保存路徑
            **kwargs: 額外要保存的資訊（如 epoch, optimizer state 等）
        """
        save_dict = {
            'model_state_dict': self.state_dict(),
            'model_class': self.__class__.__name__,
            **kwargs
        }

        torch.save(save_dict, save_path)
        print(f"模型已保存到: {save_path}")

    def load_model(self, load_path: str, strict: bool = True) -> Dict:
        """
        載入模型

        參數:
            load_path: 載入路徑
            strict: 是否嚴格匹配參數名稱

        返回:
            載入的字典（可能包含額外資訊）
        """
        checkpoint = torch.load(load_path, map_location='cpu')

        if 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            print(f"模型已從 {load_path} 載入")
        else:
            # 如果直接是 state_dict
            self.load_state_dict(checkpoint, strict=strict)
            print(f"模型權重已從 {load_path} 載入")

        return checkpoint

    def freeze_embeddings(self):
        """凍結嵌入層"""
        if hasattr(self, 'embedding'):
            for param in self.embedding.parameters():
                param.requires_grad = False
            print("嵌入層已凍結")

    def unfreeze_embeddings(self):
        """解凍嵌入層"""
        if hasattr(self, 'embedding'):
            for param in self.embedding.parameters():
                param.requires_grad = True
            print("嵌入層已解凍")

    def set_dropout(self, dropout_rate: float):
        """
        設定 dropout 比率

        參數:
            dropout_rate: 新的 dropout 比率
        """
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate

    def get_device(self) -> torch.device:
        """
        獲取模型所在設備

        返回:
            設備 (CPU 或 CUDA)
        """
        return next(self.parameters()).device

    def print_model_summary(self):
        """打印模型摘要"""
        print("\n" + "=" * 60)
        print(f"模型: {self.__class__.__name__}")
        print("=" * 60)

        # 參數資訊
        params_info = self.get_parameters_info()
        print(f"總參數量:     {params_info['total']:,}")
        print(f"可訓練參數:   {params_info['trainable']:,}")
        print(f"凍結參數:     {params_info['non_trainable']:,}")

        # 設備資訊
        device = self.get_device()
        print(f"當前設備:     {device}")

        print("=" * 60 + "\n")


class EmbeddingLayer(nn.Module):
    """
    詞嵌入層

    功能:
        - 支援預訓練嵌入（GloVe, BERT 等）
        - 支援隨機初始化
        - 支援嵌入微調控制
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        pretrained_embeddings: Optional[np.ndarray] = None,
        freeze: bool = False,
        padding_idx: int = 0
    ):
        """
        初始化嵌入層

        參數:
            vocab_size: 詞彙表大小
            embedding_dim: 嵌入維度
            pretrained_embeddings: 預訓練嵌入矩陣 [vocab_size, embedding_dim]
            freeze: 是否凍結嵌入層
            padding_idx: 填充索引
        """
        super(EmbeddingLayer, self).__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )

        # 載入預訓練嵌入
        if pretrained_embeddings is not None:
            self._load_pretrained_embeddings(pretrained_embeddings)

        # 凍結嵌入層
        if freeze:
            self.embedding.weight.requires_grad = False

    def _load_pretrained_embeddings(self, embeddings: np.ndarray):
        """
        載入預訓練嵌入

        參數:
            embeddings: 嵌入矩陣
        """
        embeddings_tensor = torch.from_numpy(embeddings).float()
        self.embedding.weight.data.copy_(embeddings_tensor)
        print(f"已載入預訓練嵌入: {embeddings.shape}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入索引 [batch, seq_len]

        返回:
            嵌入向量 [batch, seq_len, embedding_dim]
        """
        return self.embedding(x)


class AttentionPooling(nn.Module):
    """
    注意力池化層
    用於將變長序列池化為固定維度向量
    """

    def __init__(self, hidden_dim: int):
        """
        初始化注意力池化

        參數:
            hidden_dim: 隱藏層維度
        """
        super(AttentionPooling, self).__init__()

        self.attention = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向傳播

        參數:
            hidden_states: 隱藏狀態 [batch, seq_len, hidden_dim]
            mask: 掩碼 [batch, seq_len] (1 表示有效，0 表示填充)

        返回:
            (池化向量 [batch, hidden_dim], 注意力權重 [batch, seq_len])
        """
        # 計算注意力分數 [batch, seq_len, 1]
        attention_scores = self.attention(hidden_states)

        # 應用掩碼
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)

        # 計算注意力權重 [batch, seq_len, 1]
        attention_weights = torch.softmax(attention_scores, dim=1)

        # 加權求和 [batch, hidden_dim]
        pooled_output = torch.sum(hidden_states * attention_weights, dim=1)

        return pooled_output, attention_weights.squeeze(-1)


class MLP(nn.Module):
    """
    多層感知機（MLP）
    用於分類或特徵轉換
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.5,
        activation: str = 'relu',
        use_batch_norm: bool = False
    ):
        """
        初始化 MLP

        參數:
            input_dim: 輸入維度
            hidden_dims: 隱藏層維度列表
            output_dim: 輸出維度
            dropout: Dropout 比率
            activation: 激活函數 ('relu', 'tanh', 'gelu')
            use_batch_norm: 是否使用批次正規化
        """
        super(MLP, self).__init__()

        # 選擇激活函數
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支援的激活函數: {activation}")

        # 構建層
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # 線性層
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # 最後一層不加激活函數和 dropout
            if i < len(dims) - 2:
                # 批次正規化
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))

                # 激活函數
                layers.append(self.activation)

                # Dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        參數:
            x: 輸入張量 [batch, input_dim]

        返回:
            輸出張量 [batch, output_dim]
        """
        return self.mlp(x)


if __name__ == "__main__":
    # 測試基礎模型組件
    print("測試基礎模型組件...")

    # 測試嵌入層
    print("\n1. 測試嵌入層")
    embedding_layer = EmbeddingLayer(
        vocab_size=1000,
        embedding_dim=300,
        freeze=False
    )

    test_input = torch.randint(0, 1000, (4, 10))  # [batch=4, seq_len=10]
    embeddings = embedding_layer(test_input)
    print(f"嵌入輸出形狀: {embeddings.shape}")  # 應為 [4, 10, 300]

    # 測試注意力池化
    print("\n2. 測試注意力池化")
    pooling = AttentionPooling(hidden_dim=256)

    hidden_states = torch.randn(4, 10, 256)  # [batch, seq_len, hidden_dim]
    pooled, weights = pooling(hidden_states)
    print(f"池化輸出形狀: {pooled.shape}")  # 應為 [4, 256]
    print(f"注意力權重形狀: {weights.shape}")  # 應為 [4, 10]

    # 測試 MLP
    print("\n3. 測試 MLP")
    mlp = MLP(
        input_dim=256,
        hidden_dims=[128, 64],
        output_dim=3,
        dropout=0.5,
        use_batch_norm=True
    )

    output = mlp(pooled)
    print(f"MLP 輸出形狀: {output.shape}")  # 應為 [4, 3]

    print("\n基礎模型組件測試完成！")
