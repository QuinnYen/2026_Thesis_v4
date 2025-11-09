# HMAC-Net 快速開始指南（碩士論文實驗）

## 🎯 概述

本指南將協助您快速開始使用 **HMAC-Net with BERT** 進行面向級情感分析實驗。

---

## 📋 前置準備

### 1. 安裝依賴

```bash
# 安裝所有必要套件
pip install -r requirements.txt

# 如果使用 GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 驗證數據

確認您的數據已正確放置：

```bash
# 檢查數據檔案
ls data/raw/semeval2014/

# 應該看到：
# Restaurants_Train_v2.xml
# Restaurants_Test_Data_phaseB.xml
# Laptop_Train_v2.xml
# Laptops_Test_Data_phaseB.xml
```

---

## 🚀 開始訓練

### 方案 A：使用 BERT（推薦，效果更好）

```bash
cd experiments

# 訓練餐廳領域（Restaurant）
python train_bert.py --domain restaurant --epochs 20 --batch_size 16

# 訓練筆電領域（Laptop）
python train_bert.py --domain laptop --epochs 20 --batch_size 16

# 如果記憶體不足，可以凍結 BERT
python train_bert.py --domain restaurant --freeze_bert --batch_size 32
```

### 方案 B：使用 GloVe（傳統方法）

```bash
# 需要先下載 GloVe 嵌入
# 然後執行：
python train.py
```

---

## 📊 訓練參數說明

### 重要參數

| 參數 | 說明 | 預設值 | 建議值 |
|------|------|--------|--------|
| `--domain` | 數據集領域 | restaurant | restaurant 或 laptop |
| `--bert_model` | BERT 模型 | bert-base-uncased | bert-base-uncased |
| `--freeze_bert` | 凍結 BERT | False | GPU 記憶體小時使用 |
| `--batch_size` | 批次大小 | 16 | 16-32 |
| `--epochs` | 訓練輪數 | 20 | 20-30 |
| `--lr` | 學習率 | 2e-5 | 1e-5 到 3e-5 |

### 完整命令範例

```bash
# 完整配置訓練
python train_bert.py \
  --domain restaurant \
  --bert_model bert-base-uncased \
  --batch_size 16 \
  --epochs 25 \
  --lr 2e-5
```

---

## 📈 訓練過程監控

### 訓練輸出

訓練過程中您會看到：

```
訓練集: 2345 樣本
驗證集: 413 樣本
測試集: 800 樣本

Epoch 1/20 - 訓練損失: 0.8456, F1: 0.6234 | 驗證損失: 0.7234, F1: 0.6789
當前學習率: 0.000020
✓ 保存最佳模型: results/checkpoints/hmac_bert_best_f1_0.6789.pt

Epoch 2/20 - 訓練損失: 0.6234, F1: 0.7123 | 驗證損失: 0.6123, F1: 0.7456
當前學習率: 0.000020
✓ 保存最佳模型: results/checkpoints/hmac_bert_best_f1_0.7456.pt
```

### 查看結果

訓練完成後，結果保存在：

```
results/
├── checkpoints/              # 最佳模型
│   └── hmac_bert_best_f1_0.XXXX.pt
├── logs/                     # 訓練日誌
│   └── HMAC-BERT-Training_YYYYMMDD_HHMMSS.log
└── visualizations/           # 視覺化圖表
    ├── hmac_bert_training_curves.png    # 訓練曲線
    └── confusion_matrix_epoch_XX.png    # 混淆矩陣
```

---

## 🔍 常見問題

### Q1: CUDA Out of Memory 錯誤

**解決方法：**

```bash
# 方案 1：減少批次大小
python train_bert.py --batch_size 8

# 方案 2：凍結 BERT
python train_bert.py --freeze_bert --batch_size 32

# 方案 3：使用 CPU（較慢）
CUDA_VISIBLE_DEVICES="" python train_bert.py
```

### Q2: 訓練速度太慢

**建議：**

1. 使用 GPU（速度提升 10-20 倍）
2. 凍結 BERT 參數（`--freeze_bert`）
3. 減少訓練輪數（`--epochs 10`）
4. 增加批次大小（`--batch_size 32`）

### Q3: 驗證 F1 分數很低

**可能原因：**

1. 學習率太大或太小 → 調整 `--lr`
2. Dropout 太高 → 修改配置檔案
3. 訓練輪數不足 → 增加 `--epochs`

### Q4: 如何查看數據統計？

訓練開始時會自動打印數據統計：

```
============================================================
數據統計資訊
============================================================
樣本數量: 2345
唯一句子數: 2100
唯一面向數: 456
詞彙表大小: 5234
平均句子長度: 18.45 詞
平均面向長度: 1.82 詞

標籤分布:
  positive: 1245 (53.09%)
  neutral: 456 (19.44%)
  negative: 644 (27.46%)
============================================================
```

---

## 📊 期望結果

### SemEval-2014 Restaurant

根據論文中的結果，您應該期望：

- **Accuracy**: 0.82-0.85
- **Macro F1**: 0.75-0.78

### SemEval-2014 Laptop

- **Accuracy**: 0.76-0.79
- **Macro F1**: 0.71-0.74

---

## 🎓 下一步

### 1. 運行消融實驗

```bash
# 測試不同模組的貢獻
python ablation_study.py  # (待實作)
```

### 2. 比較 Baseline 模型

```bash
# 與其他模型比較
python compare_baselines.py  # (待實作)
```

### 3. 視覺化注意力權重

```python
# 在 Jupyter Notebook 中運行
from utils import AttentionVisualizer
# ... 視覺化程式碼
```

---

## 💡 實驗建議

### 最佳配置（基於經驗）

**餐廳領域：**
```bash
python train_bert.py \
  --domain restaurant \
  --batch_size 16 \
  --epochs 25 \
  --lr 2e-5
```

**筆電領域：**
```bash
python train_bert.py \
  --domain laptop \
  --batch_size 16 \
  --epochs 30 \
  --lr 2e-5
```

### 記憶體受限環境

```bash
python train_bert.py \
  --domain restaurant \
  --freeze_bert \
  --batch_size 32 \
  --epochs 20
```

---

## 📞 支援

如果遇到問題：

1. 檢查 `results/logs/` 中的日誌檔案
2. 確認數據檔案路徑正確
3. 驗證 CUDA/PyTorch 安裝

---

**祝實驗順利！🚀**
