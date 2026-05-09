"""
基礎模型抽象類
定義所有模型的通用介面和基礎功能
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict


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

        params_info = self.get_parameters_info()
        print(f"總參數量:     {params_info['total']:,}")
        print(f"可訓練參數:   {params_info['trainable']:,}")
        print(f"凍結參數:     {params_info['non_trainable']:,}")

        device = self.get_device()
        print(f"當前設備:     {device}")

        print("=" * 60 + "\n")


