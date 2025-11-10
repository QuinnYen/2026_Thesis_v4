# Multi-Aspect HMAC-Net - Quick Start Guide

**最後更新**: 2025-11-10

---

## 🚀 Windows 終端機快速啟動命令

### 推薦實驗 (Focal Loss + Neutral權重增強) ⭐

```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --use_iarm --pmac_mode sequential --iarm_mode transformer --hidden_dim 768 --dropout 0.1 --loss_type focal --focal_gamma 2.0 --class_weights 1.0 2.0 1.0
```

**預期**: Test Acc 82-83%, Neutral F1 提升至 0.55-0.60

---

## 📋 所有可用命令 (單行，Windows終端機)

### 1. 完整模型 (PMAC + IARM) + Focal Loss
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --loss_type focal --focal_gamma 2.0 --class_weights 1.0 2.0 1.0 --use_pmac --use_iarm
```

### 2. 基礎 Focal Loss (無權重增強)
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --loss_type focal --focal_gamma 2.0 --use_pmac --use_iarm
```

### 3. Adaptive Loss (自動調整權重)
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --loss_type adaptive --focal_gamma 2.0 --use_pmac --use_iarm
```

### 4. 標準 CE Loss (Baseline)
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --use_pmac --use_iarm
```

### 5. 消融實驗 - 無 PMAC
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --use_iarm --loss_type focal --class_weights 1.0 2.0 1.0
```

### 6. 消融實驗 - 無 IARM
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --use_pmac --loss_type focal --class_weights 1.0 2.0 1.0
```

### 7. 消融實驗 - 無 PMAC & IARM (只有BERT+AAHA)
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --loss_type focal --class_weights 1.0 2.0 1.0
```

### 8. 快速測試 (2 epochs, 驗證系統)
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 2 --loss_type focal --class_weights 1.0 2.0 1.0 --use_pmac --use_iarm
```

---

## 🎯 參數說明

### 必要參數
- `--epochs`: 訓練輪數 (推薦 30)
- `--batch_size`: 批次大小 (預設 16)
- `--lr`: 學習率 (預設 2e-5)

### 模組開關
- `--use_pmac`: 啟用 PMAC (漸進式組合)
- `--use_iarm`: 啟用 IARM (關係建模)
- `--pmac_mode`: PMAC模式 (sequential/pairwise/attention)
- `--iarm_mode`: IARM模式 (transformer/gat/bilinear)

### 損失函數
- `--loss_type`: 損失類型 (ce/focal/adaptive)
- `--focal_gamma`: Focal Loss gamma參數 (預設 2.0)
- `--class_weights`: 類別權重 [neg neu pos] (例: 1.0 2.0 1.0)

### 其他
- `--hidden_dim`: 隱藏層維度 (預設 768)
- `--dropout`: Dropout率 (預設 0.1)
- `--patience`: Early stopping耐心值 (預設 10)

---

## 📊 當前性能基線

```
Dataset: SemEval-2014 Restaurant
Model: DistilBERT + AAHA + PMAC + IARM
Loss: Cross-Entropy

Test Accuracy:  79.84%
Test F1 (Macro): 0.6780

Per-Class F1:
├─ Negative: 0.713 ✅
├─ Neutral:  0.430 ⚠️  ← 主要瓶頸
└─ Positive: 0.891 ✅
```

---

## 🎯 預期改進 (使用 Focal Loss)

```
推薦配置: Focal Loss + class_weights=[1.0, 2.0, 1.0]

Test Accuracy:  82-83% (+2-3%)
Test F1 (Macro): 0.72-0.75 (+0.04-0.07)

Per-Class F1:
├─ Negative: 0.73-0.76 (+0.02-0.05)
├─ Neutral:  0.55-0.60 (+0.12-0.17) ← 主要提升
└─ Positive: 0.88-0.90 (-0.01~+0.01)
```

---

## 📁 結果位置

訓練完成後，結果保存在:

```
results/
├─ checkpoints/
│  └─ hmac_multiaspect_best_f1_*.pt  # 最佳模型
├─ reports/
│  └─ multiaspect_results.json       # 測試結果
└─ visualizations/
   ├─ class_performance.png          # 類別性能圖
   └─ performance_report.md           # 性能報告
```

---

## 🔬 查看結果

### 自動生成可視化
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python utils/visualize_results.py
```

### 打開結果資料夾
```powershell
explorer D:\Quinn_SmallHouse\2026_Thesis_v4\results
```

---

## ⏱️ 預計時間

- **完整訓練 (30 epochs)**: 1-1.5 小時
- **快速測試 (2 epochs)**: 5 分鐘
- **消融實驗 (4個配置)**: 4-6 小時

---

## 🐛 故障排除

### Q: Import 錯誤 "No module named 'utils.focal_loss'"
**A**: 確認已執行過 `patch_focal_loss.py`
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python patch_focal_loss.py
```

### Q: CUDA out of memory
**A**: 降低 batch_size
```powershell
--batch_size 8  # 或 4
```

### Q: 訓練太慢
**A**: 使用 DistilBERT (已是預設)，或降低 max_aspects
```powershell
--max_aspects 6  # 預設 8
```

### Q: Neutral F1 沒提升
**A**: 增加 Neutral 權重
```powershell
--class_weights 1.0 2.5 1.0  # 或 3.0
```

---

## 📞 快速參考

### 完整命令參數列表
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --help
```

### 查看所有文件
```
D:\Quinn_SmallHouse\2026_Thesis_v4\
├─ experiments/
│  └─ train_multiaspect.py           # 主訓練腳本
├─ utils/
│  ├─ focal_loss.py                  # Focal Loss實現
│  └─ visualize_results.py           # 可視化工具
├─ EXPERIMENT_ANALYSIS_AND_IMPROVEMENTS.md  # 問題分析
├─ FOCAL_LOSS_IMPLEMENTATION_SUMMARY.md     # Focal Loss總結
└─ QUICK_START.md                    # 本文檔
```

---

## 🎓 最終推薦命令 (複製貼上即可)

### Windows PowerShell / CMD
```powershell
cd D:\Quinn_SmallHouse\2026_Thesis_v4 && python experiments/train_multiaspect.py --epochs 30 --batch_size 16 --lr 2e-5 --use_pmac --use_iarm --pmac_mode sequential --iarm_mode transformer --hidden_dim 768 --dropout 0.1 --loss_type focal --focal_gamma 2.0 --class_weights 1.0 2.0 1.0
```

**預期結果**:
- Test Accuracy: **82-83%**
- Neutral F1: **0.55-0.60**
- 訓練時間: **~1.5 小時**

---

**狀態**: ✅ 系統就緒，可立即執行
**建議**: 複製上方命令到終端機，按Enter啟動訓練
