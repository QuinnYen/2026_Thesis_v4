"""
Progressive Gate Training for PMAC

策略：
1. 階段1 (0-10 epochs): Gate 偏置 = -2.0 (sigmoid ≈ 0.12)
2. 階段2 (10-20 epochs): Gate 偏置 = -1.0 (sigmoid ≈ 0.27)
3. 階段3 (20-30 epochs): Gate 偏置 = -0.5 (sigmoid ≈ 0.38)

目的：讓模型逐步學習 aspect 交互，避免一開始就被稀疏性困住
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from experiments.train_multiaspect import main as train_main
import argparse

def progressive_gate_training():
    """
    漸進式 Gate 訓練
    """
    print("="*80)
    print("Progressive Gate Training Strategy")
    print("="*80)
    print()
    print("階段1 (Epochs 1-10):  gate_bias_init = -2.0 (sigmoid ≈ 0.12)")
    print("階段2 (Epochs 11-20): gate_bias_init = -1.0 (sigmoid ≈ 0.27)")
    print("階段3 (Epochs 21-30): gate_bias_init = -0.5 (sigmoid ≈ 0.38)")
    print()
    print("="*80)

    # TODO: 實現漸進式訓練邏輯
    # 當前只是示範框架
    print("\n⚠️  此功能需要進一步實現")
    print("建議：直接使用 --gate_bias_init -1.0 重新訓練")

if __name__ == "__main__":
    progressive_gate_training()
