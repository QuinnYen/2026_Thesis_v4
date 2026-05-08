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

## 實驗結果

### HKGAN vs Baseline（Macro-F1，5-seed avg ± std）

| 資料集 | Baseline | HKGAN       | Ensemble    | vs Baseline |
|--------|----------|-------------|-------------|------------|
| REST14 | 72.32%   | 75.34±0.90% | **76.11%**  | +3.79%     |
| LAP14  | 68.91%   | 70.85±1.35% | **71.39%**  | +2.48%     |
| MAMS   | 82.58%   | 84.29±0.63% | **84.97%**  | +2.39%     |
| REST16 | 69.00%   | 73.67±2.37% | **74.42%**  | +5.42%     |
| LAP16  | 66.36%   | 66.74±0.93% | **67.93%**  | +1.57%     |

Ensemble 報告存於 `results/HKGAN_Ensemble_{dataset}.txt`，由 `utils/ensemble_runner.py` 自動生成。

### 各類別 F1（5-seed 平均）

| 資料集 | Neg F1  | Neu F1  | Pos F1  |
|--------|---------|---------|---------|
| REST14 | 77.64%  | 57.51%  | 90.86%  |
| LAP14  | 70.64%  | 55.07%  | 86.83%  |
| MAMS   | 81.45%  | 87.23%  | 84.19%  |
| REST16 | 81.39%  | 47.51%  | 92.11%  |
| LAP16  | 82.53%  | 28.42%  | 89.28%  |

> **LAP16 Neutral F1 僅 28.42%**：train 188 Neutral（6.5%），test 46（5.7%），屬資料固有稀少性，非 bug。

---

## 消融分析

> 消融基準 `ablation_full` 使用統一超參（epochs=40，patience=12，無 per-dataset routing），確保各變體在相同訓練條件下比較。括號內為 Δ = full - 移除後，正數代表組件有效。

| 資料集 | full（基準）    | no_knowledge            | no_inter_aspect         | no_loss_eng             | no_gate                 |
|--------|----------------|-------------------------|-------------------------|-------------------------|-------------------------|
| REST14 | 76.34 ± 0.88   | 74.60 ± 0.82 **(−1.74)** | 75.00 ± 0.71 **(−1.34)** | 75.80 ± 1.51 **(−0.54)** | 75.53 ± 0.95 **(−0.81)** |
| LAP14  | 70.38 ± 1.57   | 69.15 ± 1.79 **(−1.23)** | 71.39 ± 0.45 (+1.01)    | 69.69 ± 1.19 **(−0.69)** | 69.74 ± 1.71 **(−0.64)** |
| MAMS   | 84.26 ± 0.19   | 83.96 ± 0.73 **(−0.30)** | 82.92 ± 0.23 **(−1.34)** | 84.42 ± 0.91 (+0.16)    | 83.60 ± 0.35 **(−0.66)** |
| REST16 | 74.04 ± 2.51   | 75.19 ± 0.52 (+1.15)    | 74.55 ± 1.79 (+0.51)    | 74.28 ± 2.86 (+0.24)    | 74.60 ± 1.07 (+0.56)    |
| LAP16  | 66.59 ± 1.01   | 68.18 ± 0.41 (+1.59)    | 68.01 ± 1.65 (+1.42)    | 66.91 ± 1.93 (+0.32)    | 67.01 ± 1.30 (+0.42)    |

> **REST16/LAP16 組件負貢獻說明**：Neutral 類別極度稀少（REST16 Neutral F1 47.51%，LAP16 僅 28.42%），macro-F1 波動主要受 Neutral 驅動，知識注入在稀少 Neutral 類別上帶來雜訊。不影響 REST14/LAP14/MAMS 三個主力資料集的結論。

### 消融變體說明

| 變體 | 說明 |
|------|------|
| `full` | HKGAN Full（統一超參，作為消融 delta 基準線）|
| `bert_only` | 移除所有 HKGAN 組件，建立下限 |
| `no_knowledge` | 移除整個知識增強模組（SenticNet + Gates）|
| `no_inter_aspect` | 移除跨面向建模（IARN）|
| `no_loss_eng` | 移除損失函數工程（Focal Loss + Logit Adjust）|
| `no_gate` | 移除知識門控（Confidence + Dynamic Gate）|

---

## 專案結構

```
2026_Thesis_v4/
├── configs/                              # 配置檔案
│   ├── unified_hkgan.yaml                # HKGAN 主模型配置
│   ├── unified_baseline.yaml             # Baseline 配置
│   └── ablation/                         # 消融實驗配置（6 個變體）
│       ├── ablation_full.yaml            # HKGAN Full（統一超參，消融基準線）
│       ├── tier1_bert_only.yaml          # BERT-only
│       ├── tier1_no_all_knowledge.yaml   # 移除知識增強模組
│       ├── tier1_no_inter_aspect.yaml    # 移除 Inter-Aspect 模組
│       ├── tier2_no_all_loss_engineering.yaml  # 移除損失函數工程
│       └── tier2_no_knowledge_gating.yaml      # 移除知識門控
│
├── data/                                 # 數據目錄
│   ├── raw/                              # 原始數據（SemEval-2014/2016, MAMS）
│   ├── dapt/                             # Domain-Adaptive Pre-Training 模型
│   └── SenticNet_5.0/                    # SenticNet 情感知識庫
│
├── datasets/                             # 資料集載入器
│   ├── loader_semeval14.py               # SemEval-2014 載入
│   ├── loader_semeval16.py               # SemEval-2016 載入
│   ├── loader_mams.py                    # MAMS 載入
│   ├── loader_knowledge.py               # SenticNet 知識載入
│   └── multiaspect_dataset.py            # 多面向資料集封裝
│
├── models/                               # 模型定義
│   ├── hkgan.py                          # HKGAN 主模型
│   ├── bert_embedding.py                 # BERT 特徵提取器
│   ├── hierarchical_syntax.py            # 階層式語法模組
│   └── base_model.py                     # 基礎模型類
│
├── experiments/                          # 實驗腳本
│   ├── train_from_config.py              # 統一訓練入口（YAML 驅動）
│   ├── train_multiaspect.py              # 多面向訓練核心
│   ├── improved_models.py                # 改進模型選擇器
│   ├── baselines.py                      # Baseline 模型
│   ├── generate_hkgan_report.py          # HKGAN 報告生成
│   ├── generate_ablation_report.py       # 消融實驗報告
│   ├── generate_comprehensive_report.py  # 綜合比較報告
│   └── plot_thesis_figures.py            # 論文圖表繪製
│
├── utils/                                # 工具模組
│   ├── focal_loss.py                     # Focal Loss 實現
│   ├── ensemble_runner.py                # Ensemble 推理模組（可 import）
│   ├── dataset_analyzer.py              # 數據集統計分析
│   ├── model_selector.py                 # 模型選擇器
│   └── checkpoint_cleaner.py             # Checkpoint 自動清理
│
├── results/                              # 實驗結果（.gitignore）
│   ├── improved/                         # HKGAN 實驗結果
│   ├── baseline/                         # Baseline 結果
│   ├── ablation/                         # 消融實驗結果
│   ├── HKGAN_Ensemble_{dataset}.txt      # Ensemble 報告
│   └── figures/                          # 生成的圖表
│
├── docs/                                 # 文件
├── run_experiments.py                    # 批次實驗執行腳本
├── run_ablation.py                       # 消融實驗腳本
└── requirements.txt                      # 依賴套件
```

---

## 安裝

### 環境要求
- Python 3.13+
- PyTorch 2.0+
- CUDA 11.0+（GPU 加速）

### 安裝步驟

```bash
cd 2026_Thesis_v4

python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

---

## 執行指令

### 全流程訓練

```bash
# 多種子 HKGAN（seeds: 42, 123, 2023, 999, 0）
python run_experiments.py --hkgan --full-run --multi-seed
```

### 全消融實驗

```bash
# 完整消融研究（6 變體 × 5 資料集 × 5 seeds）
python run_ablation.py --full-study --multi-seed --auto-cleanup
```

### 報表輸出

```bash
# HKGAN 實驗報表
python run_experiments.py --report-only

# 消融實驗報表
python run_ablation.py --report-only
```

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
  knowledge_weight:        # per-dataset（噪音資料集降至 0.02）
    restaurants: 0.02
    laptops: 0.02
    mams: 0.02
    rest16: 0.1
    lap16: 0.05
    default: 0.1
  use_dynamic_gate: true
  domain: null             # 停用手工 Domain Filter，由 Dynamic Gate 自主學習

training:
  batch_size: 16
  accumulation_steps: 2   # 等效 batch=32
  epochs: 40
  patience: 12
  lr:                      # per-dataset
    restaurants: 3.0e-5
    laptops: 3.0e-5
    mams: 3.0e-5
    rest16: 3.0e-5
    lap16: 2.0e-5
    default: 3.0e-5
  loss_type: "focal"
  focal_gamma:             # per-dataset
    restaurants: 2.5
    laptops: 2.5
    mams: 2.0
    rest16: 2.0
    lap16: 2.5
    default: 2.0
  use_stratified_sampler: true
```

### 主要訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 16 | 批次大小 |
| `accumulation_steps` | 2 | 梯度累積（等效 batch=32）|
| `epochs` | 40 | 訓練輪數 |
| `lr` | 3e-5（per-dataset）| 學習率 |
| `patience` | 12 | Early stopping 容忍度 |
| `loss_type` | focal | 損失函數（focal / ce）|

---

## 授權

本專案採用 MIT 授權。

## 引用

```bibtex
@article{hkgan2026,
  title={HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network
         for Aspect-Level Sentiment Analysis},
  author={Kuan Yen},
  year={2026}
}
```
