# HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network

面向級情感分析 (Aspect-Level Sentiment Analysis) 深度學習框架，整合階層式 BERT 特徵提取與 SenticNet 情感知識庫。

## 核心創新

### 1. 階層式 BERT 特徵提取
- **Low/Mid/High 層級**: 從 BERT 不同層提取互補語義特徵
- **動態層級融合**: 自動學習最佳層級組合權重

### 2. SenticNet 知識增強
- **情感極性注入**: 將外部情感知識融入注意力計算
- **動態知識門控 (v3.0)**: 根據上下文自動決定是否信任外部知識
  - 簡單句: Gate 高，充分利用 SenticNet
  - 複雜句 (含轉折詞): Gate 低，依賴 BERT 上下文

### 3. 純 PyTorch 圖注意力網路
- **多頭 GAT**: 不依賴 torch_geometric 的純 PyTorch 實現
- **跨面向關係建模 (IARN)**: 捕捉同句多面向間的情感依賴

### 4. 情感隔離機制 (v2.2+)
- **情感感知隔離**: 防止強烈情感透過 IARN 流向中性面向
- **解決蹺蹺板效應**: 動態調整隔離程度

## 專案結構

```
HKGAN/
├── configs/                    # 配置檔案
│   ├── unified_hkgan.yaml     # HKGAN 模型配置
│   └── unified_baseline.yaml  # Baseline 配置
│
├── data/                       # 數據目錄
│   ├── raw/                   # 原始數據 (SemEval-2014/2016, MAMS)
│   ├── dapt/                  # Domain-Adaptive Pre-Training 模型
│   └── SenticNet_5.0/         # SenticNet 情感知識庫
│
├── models/                     # 模型定義
│   ├── hkgan.py               # HKGAN 主模型
│   ├── bert_embedding.py      # BERT 特徵提取器
│   ├── hierarchical_syntax.py # 階層式語法模組
│   └── base_model.py          # 基礎模型類
│
├── experiments/                # 實驗腳本
│   ├── train_from_config.py   # 統一訓練入口
│   ├── train_multiaspect.py   # 多面向訓練
│   ├── improved_models.py     # 改進模型實現
│   ├── baselines.py           # Baseline 模型
│   ├── generate_hkgan_report.py      # HKGAN 報告生成
│   ├── generate_ablation_report.py   # 消融實驗報告
│   └── plot_thesis_figures.py        # 論文圖表繪製
│
├── utils/                      # 工具模組
│   ├── focal_loss.py          # Focal Loss 實現
│   ├── senticnet_loader.py    # SenticNet 載入器
│   ├── dataset_analyzer.py    # 數據集分析
│   └── model_selector.py      # 模型選擇器
│
├── results/                    # 實驗結果
│   ├── improved/              # HKGAN 實驗結果
│   ├── baseline/              # Baseline 結果
│   ├── ablation/              # 消融實驗結果
│   └── figures/               # 生成的圖表
│
├── run_experiments.py          # 批次實驗執行腳本
├── run_ablation.py            # 消融實驗腳本
└── requirements.txt            # 依賴套件
```

## 安裝

### 環境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU 加速)

### 安裝步驟

```bash
# 進入專案目錄
cd HKGAN

# 建立虛擬環境
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# 安裝依賴
pip install -r requirements.txt
```

## 數據集

支援以下面向級情感分析數據集：

| 數據集 | 代碼 | 領域 | 說明 |
|--------|------|------|------|
| SemEval-2014 Restaurant | `restaurants` / `REST14` | 餐廳 | 標準評測集 |
| SemEval-2014 Laptop | `laptops` / `LAP14` | 筆電 | 標準評測集 |
| SemEval-2016 Restaurant | `rest16` / `REST16` | 餐廳 | 擴展評測集 |
| SemEval-2016 Laptop | `lap16` / `LAP16` | 筆電 | 擴展評測集 |
| MAMS | `mams` | 餐廳 | 多面向挑戰集 |

## 快速開始

### 單一數據集訓練

```bash
# HKGAN 模式 (預設)
python run_experiments.py --dataset restaurants --hkgan

# Baseline 模式 (BERT-CLS)
python run_experiments.py --dataset restaurants --baseline
```

### 全數據集批次執行

```bash
# 對所有數據集執行 HKGAN
python run_experiments.py --hkgan --full-run

# 對所有數據集執行 Baseline
python run_experiments.py --full-baseline
```

### 多種子實驗 (統計驗證)

```bash
# 多種子 HKGAN (seeds: 42, 123, 2023, 999, 0)
python run_experiments.py --hkgan --full-run --multi-seed

# 多種子 Baseline
python run_experiments.py --full-baseline --multi-seed
```

### 統計顯著性檢驗

```bash
# Paired t-test 比較 HKGAN vs Baseline
python run_experiments.py --significance-test
```

### 生成報告與圖表

```bash
# 生成所有報告 (含 ROC 曲線)
python run_experiments.py --report-only
```

## 配置說明

### 模型配置 (`configs/unified_hkgan.yaml`)

```yaml
model:
  improved: "hkgan"
  bert_model:
    laptops: "data/dapt/laptop_dapt/final"
    restaurants: "data/dapt/restaurant_dapt/final"
    default: "bert-base-uncased"
  hidden_dim: 768
  dropout: 0.3

  # GAT 參數
  gat_heads: 4
  gat_layers: 2

  # SenticNet 知識增強
  use_senticnet: true
  knowledge_weight: 0.1
  use_dynamic_gate: true    # v3.0 動態知識門控

training:
  batch_size: 16
  epochs: 30
  lr: 3.0e-5
  loss_type: "focal"
  focal_gamma: 2.0
```

### 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `batch_size` | 16 | 批次大小 |
| `accumulation_steps` | 2 | 梯度累積 (等效 batch=32) |
| `epochs` | 30 | 訓練輪數 |
| `lr` | 3e-5 | 學習率 |
| `patience` | 10 | Early stopping 容忍度 |
| `loss_type` | focal | 損失函數 (focal/ce) |

## 實驗功能

### 1. 消融實驗

```bash
python run_ablation.py
```

測試各模組貢獻：
- **Full HKGAN**: 完整模型
- **w/o SenticNet**: 移除知識增強
- **w/o Dynamic Gate**: 移除動態門控
- **w/o IARN**: 移除跨面向建模

### 2. 報告生成

實驗完成後自動生成：
- `results/HKGAN報告_{dataset}.txt` - 單次實驗報告
- `results/HKGAN_MultiSeed_{dataset}.txt` - 多種子統計報告
- `results/Statistical_Significance_Report.txt` - 顯著性檢驗報告
- `results/figures/roc_curves.png` - ROC 曲線圖

### 3. 論文圖表

```bash
python experiments/plot_thesis_figures.py --figure all --output results/figures/
```

## 預期性能

| 數據集 | Accuracy | Macro-F1 | 說明 |
|--------|----------|----------|------|
| REST14 | 85-87% | 78-81% | 標準餐廳評測 |
| LAP14 | 80-82% | 72-75% | 標準筆電評測 |
| REST16 | 88-90% | 72-75% | 類別不均衡挑戰 |
| LAP16 | 82-84% | 70-73% | 類別不均衡挑戰 |
| MAMS | 83-85% | 84-86% | 多面向複雜句 |

## 技術特點

### Focal Loss
處理類別不均衡問題，特別針對 Neutral 類別識別：
```yaml
loss_type: "focal"
focal_gamma: 2.0
class_weights: [0.8, 1.8, 0.8]  # [Neg, Neu, Pos]
```

### 非對稱 Logit 調整
推理時動態調整各類別 logits：
```yaml
neutral_boost:
  laptops: 0.8
  restaurants: 0.6
neg_suppress:
  laptops: 0.6
```

### LLRD (Layer-wise Learning Rate Decay)
BERT 各層使用不同學習率：
```yaml
use_llrd: true
llrd_decay: 0.95
```

## 常見問題

### Q: GPU 記憶體不足？
1. 減少 `batch_size` (建議 8 或 4)
2. 增加 `accumulation_steps` 補償
3. 減少 `max_text_len`

### Q: 訓練不穩定？
1. 降低學習率 (`lr: 2e-5`)
2. 增加 warmup (`warmup_ratio: 0.15`)
3. 使用梯度裁剪 (`grad_clip: 1.0`)

### Q: Neutral F1 過低？
1. 調整 `neutral_boost` 參數
2. 增加 `class_weights[1]` (Neutral 權重)
3. 確認 `use_dynamic_gate: true`

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
