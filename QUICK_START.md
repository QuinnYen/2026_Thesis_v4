# Multi-Aspect HMAC-Net - 快速啟動指南

**最後更新**: 2025-01-12

---

## 🚀 單行命令（Windows 終端機直接複製執行）

### 1. 完整模型 - Selective PMAC + IARM + Focal Loss（推薦）⭐

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --dropout 0.3 --accumulation_steps 2 --use_pmac --pmac_mode selective --use_iarm --iarm_mode transformer --loss_type focal --focal_gamma 2.0 --class_weights 1.0 3.0 1.0
```

### 2. Selective PMAC + IARM（標準 CE Loss）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --dropout 0.3 --accumulation_steps 2 --use_pmac --pmac_mode selective --use_iarm --iarm_mode transformer
```

### 3. Sequential PMAC + IARM + Focal Loss

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --dropout 0.1 --use_pmac --pmac_mode sequential --use_iarm --iarm_mode transformer --loss_type focal --focal_gamma 2.0 --class_weights 1.0 2.0 1.0
```

### 4. Pairwise PMAC + IARM

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --pmac_mode pairwise --use_iarm --iarm_mode transformer
```

### 5. Attention-based PMAC + IARM

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --pmac_mode attention --use_iarm --iarm_mode transformer
```

---

## 🔬 消融實驗命令

### 6. 無 PMAC（僅 IARM）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --dropout 0.3 --use_iarm --iarm_mode transformer --loss_type focal --focal_gamma 2.0 --class_weights 1.0 3.0 1.0
```

### 7. 無 IARM（僅 Selective PMAC）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --dropout 0.3 --use_pmac --pmac_mode selective --loss_type focal --focal_gamma 2.0 --class_weights 1.0 3.0 1.0
```

### 8. Baseline（無 PMAC、無 IARM）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --dropout 0.3 --loss_type focal --focal_gamma 2.0 --class_weights 1.0 3.0 1.0
```

### 9. BERT Baseline（標準配置）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5
```

---

## 🧪 不同 Loss Function 實驗

### 10. Adaptive Loss（自動調整類別權重）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --pmac_mode selective --use_iarm --loss_type adaptive --focal_gamma 2.0
```

### 11. Focal Loss（無類別權重）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --pmac_mode selective --use_iarm --loss_type focal --focal_gamma 2.0
```

### 12. 不同 Focal Gamma 值測試

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --use_iarm --loss_type focal --focal_gamma 1.0 --class_weights 1.0 3.0 1.0
```

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --use_iarm --loss_type focal --focal_gamma 3.0 --class_weights 1.0 3.0 1.0
```

---

## 🎯 不同 IARM 模式實驗

### 13. GAT-based IARM

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --pmac_mode selective --use_iarm --iarm_mode gat
```

### 14. Bilinear IARM

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --pmac_mode selective --use_iarm --iarm_mode bilinear
```

---

## ⚡ 快速測試命令

### 15. 快速驗證（2 epochs）

```powershell
python experiments/train_multiaspect.py --epochs 2 --batch_size 16 --use_pmac --pmac_mode selective --use_iarm
```

### 16. 快速測試 Selective PMAC Gate 統計

```powershell
python experiments/train_multiaspect.py --epochs 5 --batch_size 16 --use_pmac --pmac_mode selective --use_iarm
```

---

## 🌐 句子級別任務命令（IMDB、SST-2 等）

### 17. IMDB 電影評論（完整模型）

```powershell
python experiments/train_sentence_level.py --dataset imdb --epochs 20 --batch_size 16 --lr 2e-5 --num_implicit_aspects 5 --use_pmac --pmac_mode selective --use_iarm --iarm_mode transformer --fusion_strategy weighted_pooling
```

### 18. IMDB Baseline（無 PMAC、無 IARM）

```powershell
python experiments/train_sentence_level.py --dataset imdb --epochs 20 --batch_size 16 --lr 2e-5 --num_implicit_aspects 5
```

### 19. SST-2 情感分析

```powershell
python experiments/train_sentence_level.py --dataset sst2 --epochs 20 --batch_size 16 --lr 2e-5 --num_implicit_aspects 5 --use_pmac --use_iarm
```

### 20. 句子級別快速測試（限制樣本數）

```powershell
python experiments/train_sentence_level.py --dataset imdb --epochs 3 --batch_size 16 --use_pmac --use_iarm --limit 500
```

---

## 🔧 資料集管理命令

### 21. 列出所有可用資料集

```powershell
python data/dataset_manager.py list
```

### 22. 查看特定資料集資訊

```powershell
python data/dataset_manager.py info --dataset imdb
```

### 23. 測試資料集載入

```powershell
python data/dataset_manager.py test --dataset semeval_rest --limit 10
```

---

## 📊 參數說明

### 資料集參數
| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--min_aspects` | 最小 aspect 數量（過濾用） | 2 | `--min_aspects 2` |
| `--max_aspects` | 最大 aspect 數量（截斷用） | 8 | `--max_aspects 8` |
| `--include_single_aspect` | 包含單 aspect 樣本 | True | 自動啟用 |
| `--virtual_aspect_mode` | 虛擬 aspect 模式 | overall | `--virtual_aspect_mode overall` |
| `--max_text_len` | 最大文本長度 | 128 | `--max_text_len 256` |
| `--max_aspect_len` | 最大 aspect 長度 | 10 | `--max_aspect_len 15` |

### 模型參數
| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--bert_model` | BERT 模型名稱 | distilbert-base-uncased | `--bert_model bert-base-uncased` |
| `--freeze_bert` | 凍結 BERT 參數 | False | `--freeze_bert` |
| `--hidden_dim` | 隱藏層維度 | 768 | `--hidden_dim 512` |
| `--dropout` | Dropout 比率 | 0.1 | `--dropout 0.3` |

### PMAC 參數
| 參數 | 說明 | 預設值 | 可選值 |
|------|------|--------|--------|
| `--use_pmac` | 啟用 PMAC | False | flag |
| `--pmac_mode` | PMAC 組合模式 | sequential | sequential, pairwise, attention, selective |

**PMAC 模式說明：**
- `sequential`: 順序組合各 aspects
- `pairwise`: 成對組合
- `attention`: 注意力機制組合
- `selective`: **可學習的 gate（推薦）** - 自動決定是否組合

### IARM 參數
| 參數 | 說明 | 預設值 | 可選值 |
|------|------|--------|--------|
| `--use_iarm` | 啟用 IARM | False | flag |
| `--iarm_mode` | IARM 關係模式 | transformer | transformer, gat, bilinear |
| `--iarm_heads` | 注意力頭數 | 4 | 2, 4, 8 |
| `--iarm_layers` | IARM 層數 | 2 | 1, 2, 3 |

**IARM 模式說明：**
- `transformer`: Transformer-based 關係建模
- `gat`: Graph Attention Network
- `bilinear`: Bilinear 交互

### 訓練參數
| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--batch_size` | Batch size | 16 | `--batch_size 32` |
| `--epochs` | 訓練輪數 | 30 | `--epochs 50` |
| `--lr` | 學習率 | 2e-5 | `--lr 3e-5` |
| `--weight_decay` | 權重衰減 | 0.01 | `--weight_decay 0.05` |
| `--grad_clip` | 梯度裁剪 | 1.0 | `--grad_clip 5.0` |
| `--patience` | Early stopping 耐心值 | 10 | `--patience 5` |
| `--virtual_weight` | 虛擬 aspect 損失權重 | 0.5 | `--virtual_weight 0.3` |
| `--accumulation_steps` | 梯度累積步數 | 2 | `--accumulation_steps 4` |
| `--use_scheduler` | 使用學習率調度器 | True | 自動啟用 |
| `--warmup_ratio` | Warmup 比例 | 0.1 | `--warmup_ratio 0.15` |

### Loss Function 參數
| 參數 | 說明 | 預設值 | 可選值 |
|------|------|--------|--------|
| `--loss_type` | 損失函數類型 | ce | ce, focal, adaptive |
| `--focal_gamma` | Focal Loss gamma 參數 | 2.0 | `--focal_gamma 3.0` |
| `--class_weights` | 類別權重 [neg, neu, pos] | None | `--class_weights 1.0 3.0 1.0` |

**Loss 類型說明：**
- `ce`: 標準 Cross-Entropy Loss
- `focal`: Focal Loss（處理類別不平衡）
- `adaptive`: 自適應加權 Loss

**類別權重建議：**
- 平衡資料：`1.0 1.0 1.0`
- 增強 Neutral：`1.0 2.0 1.0` 或 `1.0 3.0 1.0`
- 增強 Negative/Positive：`2.0 1.0 2.0`

### 句子級別專用參數
| 參數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `--dataset` | 資料集代號 | 必填 | `--dataset imdb` |
| `--num_implicit_aspects` | 隱含 aspects 數量 | 5 | `--num_implicit_aspects 7` |
| `--fusion_strategy` | Aspect 融合策略 | weighted_pooling | mean, max, weighted_pooling, attention |
| `--limit` | 限制樣本數（測試用） | None | `--limit 1000` |

---

## 📁 結果檔案位置

### Aspect-Based 任務（SemEval）

訓練完成後自動儲存至：
```
results/experiments/<timestamp>_<exp_name>/
├── checkpoints/
│   └── best_model_epoch<N>_f1_<score>.pt
├── visualizations/
│   ├── comprehensive_training_metrics.png
│   ├── per_class_f1_curves.png
│   └── (gate 分析圖表，如果使用 selective PMAC)
└── reports/
    ├── experiment_results.json
    ├── experiment_config.json
    ├── experiment_summary.txt
    └── training_report.txt
```

### 句子級別任務（IMDB 等）

訓練完成後自動儲存至：
```
results/sentence_level/<timestamp>_<dataset>_<exp_name>/
├── checkpoints/
│   └── best_model_epoch<N>_f1_<score>.pt
├── visualizations/
│   └── (可視化圖表)
└── reports/
    ├── experiment_results.json
    ├── experiment_config.json
    └── experiment_summary.txt
```

---

## 🔍 查看幫助資訊

### Aspect-Based 訓練腳本幫助

```powershell
python experiments/train_multiaspect.py --help
```

### 句子級別訓練腳本幫助

```powershell
python experiments/train_sentence_level.py --help
```

### 資料集管理器幫助

```powershell
python data/dataset_manager.py --help
```

---

## 🎯 推薦的實驗流程

### Step 1: 快速驗證系統正常運作（~5 分鐘）

```powershell
python experiments/train_multiaspect.py --epochs 2 --batch_size 16 --use_pmac --pmac_mode selective --use_iarm
```

### Step 2: 完整訓練最佳配置（~1.5 小時）

```powershell
python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --dropout 0.3 --accumulation_steps 2 --use_pmac --pmac_mode selective --use_iarm --iarm_mode transformer --loss_type focal --focal_gamma 2.0 --class_weights 1.0 3.0 1.0
```

### Step 3: 消融實驗（~4-6 小時）

依序執行命令 6、7、8、9 進行對比

### Step 4: 句子級別任務測試（~30 分鐘）

```powershell
python experiments/train_sentence_level.py --dataset imdb --epochs 3 --batch_size 16 --use_pmac --use_iarm --limit 500
```

---

**狀態**: ✅ 系統就緒，所有命令可直接複製執行
