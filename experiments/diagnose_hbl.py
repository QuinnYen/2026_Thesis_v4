"""
診斷 HBL 模型的 Layer-wise Attention 權重

從已訓練的模型檢查點中提取權重，分析是否存在學習失效問題
"""

import torch
import argparse
from pathlib import Path
import numpy as np


def load_and_analyze_weights(checkpoint_path):
    """從檢查點加載並分析 layer_weights"""
    print(f"\n正在加載檢查點: {checkpoint_path}")

    try:
        state_dict = torch.load(checkpoint_path, map_location='cpu')

        # 查找 layer_weights
        if 'layer_weights' in state_dict:
            raw_weights = state_dict['layer_weights'].numpy()

            # 計算 softmax 後的權重
            softmax_weights = torch.softmax(torch.from_numpy(raw_weights), dim=0).numpy()

            print(f"\n[OK] 找到 layer_weights")
            print(f"\n原始權重 (softmax 前):")
            print(f"  Low-level:  {raw_weights[0]:.6f}")
            print(f"  Mid-level:  {raw_weights[1]:.6f}")
            print(f"  High-level: {raw_weights[2]:.6f}")

            print(f"\nSoftmax 歸一化後:")
            print(f"  Low-level:  {softmax_weights[0]:.6f} ({softmax_weights[0]*100:.2f}%)")
            print(f"  Mid-level:  {softmax_weights[1]:.6f} ({softmax_weights[1]*100:.2f}%)")
            print(f"  High-level: {softmax_weights[2]:.6f} ({softmax_weights[2]*100:.2f}%)")
            print(f"  總和:       {softmax_weights.sum():.6f}")

            # 診斷問題
            print(f"\n診斷分析:")

            # 1. 檢查是否過度集中
            max_weight = softmax_weights.max()
            if max_weight > 0.90:
                print(f"  [WARNING] 權重過度集中 (最大值 {max_weight:.4f} > 0.90)")
                print(f"     -> Layer-wise Attention 失去意義，退化為單層特徵")

            # 2. 檢查是否過於平均
            weight_std = softmax_weights.std()
            if weight_std < 0.05:
                print(f"  [WARNING] 權重過於平均 (std {weight_std:.4f} < 0.05)")
                print(f"     -> 與固定平均權重 [0.33, 0.33, 0.33] 幾乎相同")

            # 3. 檢查初始化問題
            if np.allclose(raw_weights, 1.0, atol=0.01):
                print(f"  [WARNING] 權重接近初始值 [1.0, 1.0, 1.0]")
                print(f"     -> 權重可能沒有充分訓練")

            # 4. 理想情況
            if 0.15 <= softmax_weights.min() <= 0.25 and weight_std >= 0.05:
                print(f"  [OK] 權重分布合理 (std={weight_std:.4f})")
                print(f"     -> 各層級都有適當的貢獻")

            return softmax_weights

        else:
            print(f"\n[ERROR] 檢查點中沒有找到 layer_weights")
            print(f"   可用的參數: {list(state_dict.keys())[:10]}...")
            return None

    except Exception as e:
        print(f"\n[ERROR] 加載失敗: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='診斷 HBL Layer-wise Attention 權重')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型檢查點路徑 (.pt 檔案)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        print(f"錯誤: 檢查點不存在: {checkpoint_path}")
        return

    print("="*80)
    print("HBL Layer-wise Attention 權重診斷")
    print("="*80)

    weights = load_and_analyze_weights(checkpoint_path)

    if weights is not None:
        print("\n" + "="*80)
        print("建議:")
        print("="*80)

        max_idx = weights.argmax()
        layer_names = ['Low-level (詞法)', 'Mid-level (語義)', 'High-level (任務)']

        print(f"\n當前模型最依賴: {layer_names[max_idx]} ({weights[max_idx]*100:.1f}%)")

        if weights.max() > 0.90:
            print(f"\n建議方案:")
            print(f"1. 降低學習率: 當前可能過大，導致權重快速收斂到極端值")
            print(f"2. 增加正則化: Weight Decay 或 Dropout")
            print(f"3. 考慮使用溫度係數: layer_weights / temperature")

        if weights.std() < 0.05:
            print(f"\n建議方案:")
            print(f"1. 權重初始化: 使用非均勻初始化，如 [0.5, 1.0, 1.5]")
            print(f"2. 增加訓練 epochs: 權重可能需要更長時間才能分化")
            print(f"3. 檢查梯度: 確保 layer_weights 有接收到梯度更新")


if __name__ == "__main__":
    main()
