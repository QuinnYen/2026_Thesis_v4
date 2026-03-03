# HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-22c55e)
![Status](https://img.shields.io/badge/Status-論文收尾中-8b5cf6)

面向級情感分析（Aspect-Level Sentiment Analysis）深度學習框架，整合階層式 BERT 特徵提取與 SenticNet 情感知識庫。

---

## 核心創新

### 1. 階層式 BERT 特徵提取
- **Low / Mid / High 層級**：從 BERT 不同層提取互補語義特徵
- **動態層級融合**：自動學習最佳層級組合權重

### 2. SenticNet 知識增強
- **情感極性注入**：將外部情感知識融入注意力計算
- **動態知識門控**：根據上下文自動決定是否信任外部知識
  - 簡單句：Gate 高，充分利用 SenticNet
  - 複雜句（含轉折詞）：Gate 低，依賴 BERT 上下文

### 3. 純 PyTorch 圖注意力網路
- **多頭 GAT**：不依賴 torch_geometric 的純 PyTorch 實現
- **跨面向關係建模（IARN）**：捕捉同句多面向間的情感依賴

### 4. 情感隔離機制
- **情感感知隔離**：防止強烈情感透過 IARN 流向中性面向
- **解決蹺蹺板效應**：動態調整隔離程度

---

## 實驗結果（5 seeds 平均 ± 標準差）

| 資料集 | Accuracy | Macro-F1 | vs. Baseline |
|--------|----------|----------|-------------|
| REST14 | 83.16% ± 0.63% | 75.66% ± 1.18% | +3.44% F1 |
| LAP14  | 75.80% ± 0.78% | 71.59% ± 1.09% | +1.96% F1 |
| REST16 | 89.03% ± 0.25% | 77.81% ± 0.74% | +7.95% F1 |
| LAP16  | 83.31% ± 0.65% | 68.48% ± 0.78% | +1.81% F1 |
| MAMS   | 84.57% ± 0.38% | 83.96% ± 0.43% | +0.91% F1 |

---

## 專案結構

```
2026_Thesis_v4/
├── configs/                        # 配置檔案
│   ├── unified_hkgan.yaml          # HKGAN 主模型配置
│   ├── unified_baseline.yaml       # Baseline 配置
│   └── ablation/                   # 消融實驗配置（5 個變體）
│
├── data/                           # 數據目錄
│   ├── raw/                        # 原始數據（SemEval-2014/2016, MAMS）
│   ├── dapt/                       # Domain-Adaptive Pre-Training 模型
│   └── SenticNet_5.0/              # SenticNet 情感知識庫
│
├── models/                         # 模型定義
│   ├── hkgan.py                    # HKGAN 主模型
│   ├── bert_embedding.py           # BERT 特徵提取器
│   ├── hierarchical_syntax.py      # 階層式語法模組
│   └── base_model.py               # 基礎模型類
│
├── experiments/                    # 實驗腳本
│   ├── train_from_config.py        # 統一訓練入口
│   ├── train_multiaspect.py        # 多面向訓練
│   ├── improved_models.py          # 改進模型實現
│   ├── baselines.py                # Baseline 模型
│   ├── generate_hkgan_report.py    # HKGAN 報告生成
│   ├── generate_ablation_report.py # 消融實驗報告
│   └── plot_thesis_figures.py      # 論文圖表繪製
│
├── utils/                          # 工具模組
│   ├── focal_loss.py               # Focal Loss 實現
│   ├── senticnet_loader.py         # SenticNet 載入器
│   ├── dataset_analyzer.py         # 數據集分析
│   ├── model_selector.py           # 模型選擇器
│   └── checkpoint_cleaner.py       # Checkpoint 自動清理
│
├── tests/                          # 工具腳本
│   └── cleanup_checkpoints.py      # 手動清理 checkpoint（命令列入口）
│
├── results/                        # 實驗結果
│   ├── improved/                   # HKGAN 實驗結果
│   ├── baseline/                   # Baseline 結果
│   ├── ablation/                   # 消融實驗結果
│   └── figures/                    # 生成的圖表
│
├── docs/                           # 文件
├── run_experiments.py              # 批次實驗執行腳本
├── run_ablation.py                 # 消融實驗腳本
└── requirements.txt                # 依賴套件
```

---

## 安裝

### 環境要求
- Python 3.13+
- PyTorch 2.0+
- CUDA 11.0+（GPU 加速）

### 安裝步驟

```bash
# 進入專案目錄
cd 2026_Thesis_v4

# 建立虛擬環境
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt
```

---

## 快速開始

### 單一資料集訓練

```bash
# HKGAN 模式
python run_experiments.py --dataset restaurants --hkgan

# Baseline 模式（BERT-CLS）
python run_experiments.py --dataset restaurants --baseline
```

### 全資料集批次執行

```bash
# 對所有資料集執行 HKGAN
python run_experiments.py --hkgan --full-run

# 對所有資料集執行 Baseline
python run_experiments.py --full-baseline
```

### 多種子實驗（統計驗證）

```bash
# 多種子 HKGAN（seeds: 42, 123, 2023, 999, 0）
python run_experiments.py --hkgan --full-run --multi-seed

# 實驗結束後自動清理多餘 checkpoint
python run_experiments.py --hkgan --full-run --multi-seed --auto-cleanup
```

### 統計顯著性檢驗

```bash
# Paired t-test 比較 HKGAN vs Baseline
python run_experiments.py --significance-test
```

### 生成報告與圖表

```bash
python run_experiments.py --report-only
```

---

## 消融實驗

```bash
# 完整消融研究（5 變體 × 5 資料集 × 5 seeds）
python run_ablation.py --full-study --multi-seed

# 生成消融報告
python run_ablation.py --report-only
```

| 消融變體 | 說明 |
|----------|------|
| `bert_only` | 移除所有 HKGAN 組件，建立下限 |
| `no_all_knowledge` | 移除整個知識增強模組（SenticNet + Gates）|
| `no_inter_aspect` | 移除跨面向建模（IARN）|
| `no_all_loss_eng` | 移除損失函數工程（Focal + Contrastive + Logit Adjust）|
| `no_knowledge_gating` | 移除知識門控（Confidence + Dynamic Gate）|

---

## 配置說明

### 模型配置（`configs/unified_hkgan.yaml`）

```yaml
model:
  improved: "hkgan"
  bert_model:
    laptops: "data/dapt/laptop_dapt/final"
    restaurants: "data/dapt/restaurant_dapt/final"
    default: "bert-base-uncased"
  hidden_dim: 768
  dropout: 0.3
  gat_heads: 4
  gat_layers: 2
  use_senticnet: true
  knowledge_weight: 0.1
  use_dynamic_gate: true

training:
  batch_size: 16
  epochs: 30
  lr: 3.0e-5
  loss_type: "focal"
  focal_gamma: 2.0
```

### 主要訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 16 | 批次大小 |
| `accumulation_steps` | 2 | 梯度累積（等效 batch=32）|
| `epochs` | 30 | 訓練輪數 |
| `lr` | 3e-5 | 學習率 |
| `patience` | 10 | Early stopping 容忍度 |
| `loss_type` | focal | 損失函數（focal / ce）|

---

## Checkpoint 清理

```bash
# Dry-run（只列出，不刪除）
python tests/cleanup_checkpoints.py

# 實際執行刪除
python tests/cleanup_checkpoints.py --execute
```

---

## 常見問題

**Q: GPU 記憶體不足？**
1. 減少 `batch_size`（建議 8 或 4）
2. 增加 `accumulation_steps` 補償
3. 減少 `max_text_len`

**Q: 訓練不穩定？**
1. 降低學習率（`lr: 2e-5`）
2. 增加 warmup（`warmup_ratio: 0.15`）
3. 使用梯度裁剪（`grad_clip: 1.0`）

**Q: Neutral F1 過低？**
1. 調整 `neutral_boost` 參數
2. 增加 `class_weights[1]`（Neutral 權重）
3. 確認 `use_dynamic_gate: true`

---

## 授權

本專案採用 MIT 授權。

## 引用

```bibtex
@article{hkgan2026,
  title={HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network
         for Aspect-Level Sentiment Analysis},
  author={Your Name},
  year={2026}
}
```
