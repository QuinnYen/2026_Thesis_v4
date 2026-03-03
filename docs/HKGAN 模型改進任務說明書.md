# HKGAN 模型改進任務說明書

> **目的**：提供目前 HKGAN 模型的完整狀態，讓 Claude Code 了解困境並提出可行的改進策略。

---

## 一、研究背景

**任務**：面向級情感分析（Aspect-Based Sentiment Analysis, ABSA）
**目標**：給定一段評論文字與其中的 aspect（面向詞），預測該 aspect 的情感極性（Negative / Neutral / Positive）

**三大核心挑戰**：
1. **外部知識可信度問題**：SenticNet 知識庫存在覆蓋率不足與領域偏移，直接注入會引入雜訊
2. **跨面向情感干擾**：同一句話中多個 aspect 的情感極性會相互污染（cross-contamination）
3. **Neutral 類別系統性誤判**：資料集嚴重不平衡，Neutral 樣本稀少，模型傾向忽略 Neutral

---

## 二、HKGAN 架構概述（v3.0）

```
Input → BERT (12層) → 三層級特徵分離 (Low/Mid/High)
      → SenticNet 知識注入
        → Confidence Gate (可信度評估)
        → Dynamic Knowledge Gate (軟性融合 BERT + Knowledge)
      → 三層 Hierarchical GAT (Token/Phrase/Clause)
        → Sentiment-Aware Isolation (情感一致性動態隔離)
      → Classifier (768→768→3)
      → Asymmetric Logit Adjustment (推理時 Neutral boost)
      → Output: [Negative, Neutral, Positive]
```

**核心組件說明**：

| 組件 | 版本 | 解決的問題 |
|------|------|-----------|
| Confidence Gate | v2.1 | 過濾低可信度的 SenticNet 知識 |
| Dynamic Knowledge Gate | v3.0 | 軟性融合 BERT 特徵與知識特徵 |
| Sentiment-Aware Isolation | v2.3 | 根據情感一致性動態調整跨面向隔離程度 |
| Asymmetric Logit Adjustment | v3.0 | 推理時對 Neutral logit 加權補償 |
| Hierarchical GAT (3-level) | v3.0 | Token/Phrase/Clause 三層感受野 |

**訓練配置**：
```yaml
bert_model: bert-base-uncased
hidden_dim: 768
gat_heads: 4
gat_layers: 2
loss_type: focal (gamma=2.0)
class_weights: [0.8, 1.8, 0.8]  # [Neg, Neu, Pos]
lr: 3.0e-5
batch_size: 16 (+ accumulation_steps=2, 等效 batch=32)
epochs: 30 + early stopping (patience=10)
use_llrd: true (decay=0.95)
```

---

## 三、目前實驗結果（多種子平均，seeds=[42,123,2023,999,0]）

### 3.1 HKGAN vs BERT-CLS Baseline 完整比較

| Dataset | Baseline Acc | HKGAN Acc | Baseline F1 | HKGAN F1 | ΔF1 | Baseline Neu F1 | HKGAN Neu F1 | ΔNeu F1 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| REST16 | 85.85% | 89.03% | 69.86% | 77.81% | **+7.95%** | 39.25% | 56.63% | **+17.38%** |
| REST14 (Restaurants) | 80.69% | 83.16% | 72.22% | 75.66% | +3.44% | 52.03% | 57.44% | +5.41% |
| LAP14 (Laptops) | 75.22% | 75.80% | 69.63% | 71.59% | +1.96% | 51.52% | 57.28% | +5.76% |
| LAP16 | 82.09% | 83.31% | 66.67% | 68.48% | +1.81% | 30.77% | 34.15% | +3.38% |
| MAMS | 83.40% | 84.57% | 83.05% | 83.96% | +0.91% | 85.06% | 86.97% | +1.92% |
| **平均** | - | - | - | - | **+3.21%** | - | - | **+6.77%** |

### 3.2 各類別 F1 細節（HKGAN Mean）

| Dataset | Neg F1 | Neu F1 | Pos F1 | Macro-F1 |
|---------|:---:|:---:|:---:|:---:|
| REST16 | 83.14% | 56.63% | 93.66% | 77.81% |
| REST14 | 78.37% | 57.44% | 91.17% | 75.66% |
| LAP14 | 70.61% | 57.28% | 86.89% | 71.59% |
| LAP16 | 81.70% | 34.15% | 89.58% | 68.48% |
| MAMS | 81.11% | 86.97% | 83.81% | 83.96% |

### 3.3 統計顯著性（Paired t-test, α=0.05）

| Dataset | Baseline F1 | HKGAN F1 | ΔF1 | t-stat | p-value | Cohen's d | 顯著 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| REST16 | 69.05% | 77.81% | +8.76% | 5.963 | 0.0040 | 2.667 | **Yes \*\*** |
| REST14 | 71.95% | 75.66% | +3.71% | 4.023 | 0.0158 | 1.799 | **Yes \*** |
| MAMS | 82.49% | 83.96% | +1.47% | 5.275 | 0.0062 | 2.359 | **Yes \*\*** |
| LAP16 | 66.43% | 68.48% | +2.05% | 5.401 | 0.0057 | 2.415 | **Yes \*\*** |
| LAP14 | 68.85% | 71.59% | +2.75% | 2.403 | 0.0741 | 1.075 | No |

**✅ 5/5 資料集皆為大效果量（Cohen's d > 0.8）；4/5 達統計顯著（p < 0.05）**

---

## 四、當前困境分析

### 4.1 整體表現瓶頸

雖然 HKGAN 相較 BERT-CLS baseline 有所改進，但與文獻中的 SOTA 方法相比仍有差距：

**與文獻比較（REST14 Macro-F1 參考值）**：
- BiSyn-GAT+ (2022): ~78-79%
- DualGCN (2022): ~76-78%
- HKGAN (ours): **75.66%**

### 4.2 各資料集問題診斷

**LAP14（最弱，F1=71.59%）**：
- Neg F1 幾乎沒有改善（-0.06%），表示負面情感識別遇到天花板
- 筆電領域的技術詞彙對 SenticNet 覆蓋率不足
- 中性類別稀少（19.2%）但改善也相對有限

**LAP16（F1=68.48%，Neu F1 僅 34.15%）**：
- Neutral 樣本比例極低（6.8%），是最難的不平衡資料集
- AUC 甚至下降了 -1.04%，顯示模型在某些決策邊界上不穩定
- Macro-F1 標準差 σ=0.78%（尚可），但絕對值偏低

**MAMS（最平衡資料集，Neutral=32.7%，但改善最小，ΔF1=+0.91%）**：
- MAMS 設計為每句有多個 aspect 且情感各異，是「跨面向干擾」最嚴重的資料集
- 現有 Sentiment-Aware Isolation 機制改善有限
- Neg F1 甚至下降 -0.50%，出現輕微蹺蹺板效應

**LAP14 統計不顯著（p=0.074）**：
- 表示改善的穩定性不足，不同 seed 下結果波動較大
- Neu F1 標準差 σ=2.63%（最大），顯示 Neutral 識別不穩定

### 4.3 現有機制的已知限制

1. **Static class weights [0.8, 1.8, 0.8]**：固定權重在所有資料集上使用，但各資料集不平衡程度差異極大（Neutral 比例從 5.4% 到 32.7%）
2. **Asymmetric Logit Adjustment**：推理時直接加減 logit 是 heuristic 方法，缺乏理論根據，且需要手動調整每個資料集的參數
3. **SenticNet 知識覆蓋率**：筆電領域技術詞彙（如 "thermal throttling", "fps", "SSD"）在 SenticNet 中幾乎沒有對應
4. **知識融合方式**：Dynamic Knowledge Gate 是 concatenation + sigmoid gate，屬於較簡單的融合方式

---

## 五、可探索的改進方向

以下方向供 Claude Code 評估可行性與實作複雜度，**優先考慮對 LAP14 和 MAMS 有幫助的改進**：

### 方向 A：動態損失權重（高優先）
**問題**：固定 class weights 無法適應各資料集不同的不平衡程度
**想法**：根據訓練中實際的類別分佈動態調整 Neutral 的損失權重
```python
# 概念：依據當前 batch 的類別頻率動態調整
# 或使用 label-distribution-aware margin loss
```

### 方向 B：Supervised Contrastive Learning（中優先）
**問題**：Neutral 類別語義模糊，與 Pos/Neg 邊界不清晰
**想法**：在特徵空間中，讓同類別的 aspect representation 更聚集，不同類別更分散
```python
# 在 classifier 前的 features 上加 contrastive loss
# 可參考 SCL (Supervised Contrastive Learning, Khosla et al. 2020)
```

### 方向 C：MAMS 專用多面向協同增強（中優先）
**問題**：MAMS 中同一句的多個 aspect 情感各異，現有隔離機制不夠精準
**想法**：明確建模同一句內 aspect pair 之間的關係，利用已知的情感對比關係來增強識別

### 方向 D：資料增強（低優先，但可能有效）
**問題**：Neutral 樣本數量少
**想法**：對 Neutral 樣本做 Easy Data Augmentation（同義詞替換、back-translation）
**注意**：CKG (2024) 用 ChatGPT 生成增強資料取得好成績，但這屬於資料層面改進，需在論文中說明

### 方向 E：知識庫補強（較耗時）
**問題**：SenticNet 對技術領域詞彙覆蓋不足
**想法**：結合 VADER 或 domain-specific lexicon 作為補充知識來源

---

## 六、程式碼結構參考

```
專案根目錄/
├── models/
│   └── hkgan.py              # 核心模型（HKGAN class）
├── configs/
│   └── unified_hkgan.yaml    # 訓練配置
├── run_experiments.py         # 實驗執行入口
├── run_ablation.py            # 消融實驗
├── experiments/
│   ├── generate_hkgan_report.py
│   └── plot_thesis_figures.py
└── results/
    ├── HKGAN_MultiSeed_*.txt  # 多種子實驗報告
    └── Statistical_Significance_Report.txt
```

**關鍵訓練指令（Windows PowerShell）**：
```powershell
# 單資料集多種子實驗
python run_experiments.py --hkgan --dataset restaurants --multi-seed

# 全資料集
python run_experiments.py --hkgan --full-run --multi-seed

# 統計顯著性檢驗
python run_experiments.py --significance-test
```

---

## 七、給 Claude Code 的任務請求

請根據以上說明，協助評估並實作以下改進：

1. **首先診斷**：閱讀 `models/hkgan.py` 了解目前架構，確認 loss function 和 class weights 的實作位置
2. **提出具體修改方案**：針對上述方向 A（動態損失權重）或 B（Contrastive Learning）提出程式碼修改
3. **評估風險**：每個修改方案預計增加的訓練時間與記憶體消耗
4. **優先目標**：讓 LAP14 的統計顯著性達到 p < 0.05，以及提升 MAMS 和 LAP16 的 Neutral F1

**約束條件**：
- 不改變模型的對外介面（input/output 格式）
- 修改後需能用原有的 `run_experiments.py` 執行
- 不引入需要額外資料集或外部 API 的方法（保持自給自足）

---

*文件生成時間：2026-02-26*
*HKGAN v3.0 | BERT-base-uncased | 五資料集多種子實驗*
