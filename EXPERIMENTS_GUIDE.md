# 實驗執行指南

## 數據增強（EDA）🔥

訓練前先生成增強數據：

```bash
# Restaurants 數據集（+83.6%）
python data/augment_multiaspect.py \
  --dataset restaurants --num_aug 2 --alpha 0.15 \
  --output_dir data/augmented_restaurants

# Laptops 數據集（預計 +~80%）
python data/augment_multiaspect.py \
  --dataset laptops --num_aug 2 --alpha 0.15 \
  --output_dir data/augmented_laptops
```

**參數說明**：
- `--dataset`: 數據集選擇 (restaurants 或 laptops)
- `--num_aug`: 每樣本生成數量（2 = 每樣本增強 2 次）
- `--alpha`: EDA 強度（0.15 = 15% 單詞會被修改）
- `--output_dir`: 輸出目錄

## 快速開始

**所有命令都必須指定 `--dataset <restaurants|laptops>` 參數**

### 方法 1: 使用配置文件（推薦）✨

```bash
# 執行完整模型（EDA 增強數據）🔥 推薦
python experiments/train_from_config.py --config configs/full_model_augmented.yaml --dataset <DATASET>

# 執行完整模型（原始數據，不增強）
python experiments/train_from_config.py --config configs/full_model_optimized.yaml --dataset <DATASET>

# 執行 BERT Only baseline
python experiments/train_from_config.py --config configs/baseline_bert_only.yaml --dataset <DATASET>

# 執行 PMAC Only（消融實驗）
python experiments/train_from_config.py --config configs/pmac_only.yaml --dataset <DATASET>
```

### 方法 2: 批次執行多個實驗

```bash
# 執行所有 baseline（BERT Only, BERT+AAHA, BERT+Mean）並自動生成報告
python run_experiments.py --baselines

# 執行所有實驗（含自動生成報告）
python run_experiments.py --all

# 只執行完整模型
python run_experiments.py --full

# 執行消融實驗
python run_experiments.py --ablation

# 只生成 baseline 比較報告（不執行訓練）
python run_experiments.py --report
```

### 方法 3: 使用命令行參數（傳統方式）

```bash
python experiments/train_multiaspect.py --dataset <DATASET> --use_pmac --use_iarm --gate_bias_init -0.5 --epochs 30 --batch_size 32 --lr 2e-5 --dropout 0.3 --loss_type focal --focal_gamma 2.5 --class_weights 1.0 5.0 1.0
```

## 配置文件說明

### 已創建的配置

1. **`configs/full_model_augmented.yaml`** 🔥 - 完整模型（使用 EDA 增強數據）
2. **`configs/baseline_bert_only.yaml`** - BERT Only baseline
3. **`configs/baseline_bert_aaha.yaml`** - BERT + AAHA baseline
4. **`configs/baseline_bert_mean.yaml`** - BERT + Mean Pooling baseline
5. **`configs/full_model_optimized.yaml`** - 完整模型（原始數據，gate_bias_init=-1.0）
6. **`configs/pmac_only.yaml`** - 只使用 PMAC（不用 IARM）

### 配置文件結構

```yaml
experiment_name: "my_experiment"

model:
  baseline: null  # 或 "bert_only", "bert_aaha", "bert_mean"
  bert_model: "distilbert-base-uncased"
  dropout: 0.3

  # PMAC 配置
  use_pmac: true
  gate_bias_init: -0.5
  gate_weight_gain: 1.0

  # IARM 配置
  use_iarm: true
  iarm_heads: 4
  iarm_layers: 2

data:
  max_text_len: 128
  max_aspect_len: 10

training:
  batch_size: 32
  epochs: 30
  lr: 2.0e-5
  patience: 15

  # 損失函數
  loss_type: "focal"
  focal_gamma: 2.5
  class_weights: [1.0, 5.0, 1.0]
```

## 覆蓋配置參數

可以在使用配置文件時覆蓋特定參數：

```bash
python experiments/train_from_config.py --config configs/full_model_optimized.yaml --dataset <DATASET> --override --epochs 50 --lr 3e-5 --batch_size 16
```

## 實驗結果位置

- **完整模型**: `results/experiments/YYYYMMDD_HHMMSS_pmac_iarm_*/`
- **Baseline**: `results/baseline/YYYYMMDD_HHMMSS_baseline_*/`

每個實驗目錄包含:
- `checkpoints/` - 模型檢查點
- `visualizations/` - 訓練曲線和 gate 分析
- `reports/` - 實驗報告和配置

## 常用實驗配置

### 實驗 A: Gate 初始化測試

```yaml
# configs/gate_test_conservative.yaml
model:
  gate_bias_init: -1.0  # 保守 (sigmoid ≈ 0.27)

# configs/gate_test_moderate.yaml
model:
  gate_bias_init: -0.5  # 適中 (sigmoid ≈ 0.38)

# configs/gate_test_aggressive.yaml
model:
  gate_bias_init: 0.0   # 積極 (sigmoid ≈ 0.50)
```

### 實驗 B: Class Weights 調整

```yaml
# 當前配置
training:
  class_weights: [1.0, 5.0, 1.0]  # Neutral 權重 5倍

# 測試更高權重
training:
  class_weights: [1.0, 8.0, 1.0]  # Neutral 權重 8倍
```

### 實驗 C: 消融實驗

```bash
# 不使用 PMAC
python experiments/train_from_config.py --config configs/baseline_bert_only.yaml --dataset <DATASET>

# 只使用 PMAC
python experiments/train_from_config.py --config configs/pmac_only.yaml --dataset <DATASET>

# 完整模型
python experiments/train_from_config.py --config configs/full_model_optimized.yaml --dataset <DATASET>
```

## 統一配置參數

所有 baseline 使用相同訓練配置（確保公平比較）：

| 參數 | 值 | 說明 |
|------|----|----|
| `epochs` | 30 | 訓練輪數 |
| `lr` | 2e-5 | 學習率 |
| `dropout` | 0.3 | Dropout 比率 |
| `loss_type` | focal | Focal Loss |
| `focal_gamma` | 2.5 | Focal Loss gamma 參數 |
| `class_weights` | [1.0, 5.0, 1.0] | 類別權重 [Neg, Neu, Pos] |
| `patience` | 10 | Early stopping patience |

各 baseline 的差異僅在 `batch_size`：BERT Only/Mean 用 32，BERT+AAHA 用 24

## 生成 Baseline 報告

**注意**：使用 `python run_experiments.py --baselines` 會自動生成報告，無需手動執行。

如果只需要重新生成報告（不執行訓練）:

```bash
# 方法 1: 使用批次腳本（推薦）
python run_experiments.py --report

# 方法 2: 直接調用報告生成腳本
python experiments/generate_baseline_report.py
```

報告位置: `results/baseline_comparison/baseline_comparison_*.md`

報告包含指標：Test Acc, Test F1, Negative F1, Neutral F1, Positive F1, Best Epoch

## 故障排除

### 問題: YAML 解析錯誤

確保 YAML 語法正確，特別是縮排（使用空格，不用 Tab）

### 問題: 配置文件找不到

使用相對路徑或絕對路徑:
```bash
python experiments/train_from_config.py --config configs/my_config.yaml
```

### 問題: 記憶體不足

調整 batch_size 或 accumulation_steps:
```yaml
training:
  batch_size: 16          # 減少
  accumulation_steps: 2   # 增加（效果相當於 batch_size 32）
```
