# HPNet (2021) - 論文快速參考

**完整標題**: A hierarchical and parallel framework for End-to-End Aspect-based Sentiment Analysis

**作者**: Ding Xiao, Feiyang Ren, Xiaoxuan Pang, Ming Cai, et al.

**發表**: Neurocomputing 465 (2021) 549-560

**核心機構**: Zhejiang University

---

## 📋 論文概述

### 研究問題
**End-to-End Aspect-Based Sentiment Analysis (E2E-ABSA)** 包含兩個子任務：
1. **Aspect Extraction (AE)**: 識別評價對象 (aspect terms)
2. **Polarity Classification (PC)**: 預測情感極性 (positive/negative/neutral)

### 研究動機
過往研究發現 joint models 的性能一直不如 pipeline 和 collapsed models。HPNet 要探索 joint model 的潛力。

### 核心觀察
1. AE和PC分別屬於 **syntactic task** 和 **semantic task**，應該部署在不同的神經網路層
2. 深度神經網路的不同層有不同的語言學表示能力
3. 現有方法只用最後一層，忽略了 BERT 的階層特性

---

## 🎯 核心方法

### 任務定義

**輸入**: 句子 S = [s₁, s₂, ..., sₙ]

**輸出**:
- Aspect labels: E = [e₁, e₂, ..., eₙ], where eₜ ∈ {B, I, O}
- Polarity labels: P = [p₁, p₂, ..., pₙ], where pₜ ∈ {pos, neg, neu}

**範例**:
```
Sentence: "Great food but the service was dreadful!"
Aspects:  "food" (positive), "service" (negative)
```

### 兩種模型架構

#### 1. HPNet-S (Specific-Layer Joint Model)

**核心思想**: 手動為兩個子任務選擇特定層

**架構**:
```
BERT Backbone Network
├── Layer t (中間層) → CRF Layer → Aspect Extraction
└── Layer l (最後層) → Classification Layer → Polarity Classification
```

**層選擇策略**:
- **Aspect Extraction**: 使用中間層 (如 Layer 9)
  - 理由: Syntactic information 在中間層最明顯
- **Polarity Classification**: 使用最後層 (Layer 12)
  - 理由: Semantic information 在高層聚合

**數學表示**:
```python
# Aspect Extraction (使用 Layer t)
P = hₜWₜ + bₜ                    # CRF scores
lossₐₛₚ = -log p(Y|X)            # CRF loss

# Polarity Classification (使用 Layer l)
U = softmax(hₗWₛ + bₛ)           # Sentiment scores
lossₛₑₙ = -ΣΣ u·log(û)           # Cross-entropy loss

# 總損失
loss = lossₐₛₚ + lossₛₑₙ + λ||θ||²
```

#### 2. HPNet-M (Multiple-Layer Joint Model)

**核心思想**: 動態學習所有層的權重，為兩個任務分別組合

**架構**:
```
BERT Backbone Network (All Layers)
├── Weighted Combination → CRF Layer → Aspect Extraction
│   Mₐ = cₐ · Σ(wₐᵢ · hᵢ)
│
└── Weighted Combination → Classification Layer → Polarity Classification
    Mₛ = cₛ · Σ(wₛᵢ · hᵢ)
```

**數學表示**:
```python
# Aspect Extraction 的多層組合
Mₐ = cₐ · Σᵢ₌₁ˡ (wₐᵢ · hᵢ)
where: wₐ = [wₐ₁, wₐ₂, ..., wₐₗ] (softmax-normalized)
       cₐ: trainable scalar (initialized to 1)

# Polarity Classification 的多層組合  
Mₛ = cₛ · Σᵢ₌₁ˡ (wₛᵢ · hᵢ)
where: wₛ = [wₛ₁, wₛ₂, ..., wₛₗ] (softmax-normalized)
       cₛ: trainable scalar (initialized to 1)
```

**關鍵特點**:
- 每個子任務有獨立的權重集 (wₐ 和 wₛ)
- 權重通過反向傳播自動學習
- 受 ELMo (Peters et al. 2018) 啟發

### 創新點

#### 1. 階層式結構 (Hierarchical Structure)
- 利用 BERT 的階層特性
- 為不同任務選擇合適的層級
- 基於語言學證據 (Jawahar, Tenney, Hewitt 等研究)

#### 2. 平行執行 (Parallel Execution)
- 訓練和推理都平行執行兩個子任務
- **關鍵技巧**: 讓模型預測每個詞的sentiment，而非只預測aspect terms的sentiment
- 解決 target-polarity mismatch 問題
- 提升推理吞吐量

#### 3. 聯合學習 (Joint Learning)
- 共享同一個 BERT backbone
- 確保兩個子任務的關聯性和共性
- 避免 pipeline 的誤差傳播

---

## 📊 數據集

### 1. Restaurant Dataset
- **來源**: SemEval 2014, 2015, 2016 restaurant domain 的聯集
- **規模**:
  - Train: 3,452 句
  - Test: 973 句
  - Aspects: 4,821 (train) + 1,351 (test)

### 2. Laptop Dataset
- **來源**: SemEval 2014 Task 4
- **規模**:
  - Train: 2,163 句
  - Test: 638 句
  - Aspects: 2,041 (train) + 654 (test)

### 3. Twitter Dataset
- **來源**: Mitchell et al.
- **規模**: 6,940 句
- **特點**: 無 train-test split，使用 10-fold cross-validation

---

## 🔧 實驗設置

### 模型參數

| 參數 | 設定 |
|------|------|
| BERT 模型 | BERT-base (12 layers, 768 dim) |
| 最大句長 | 80 words |
| Batch Size | 32 |
| 初始化 | xavier uniform |
| L2 正則化 | λ = 0.01 |
| Dropout | 0.1 |
| 學習率 (一般) | 2e-5 |
| 學習率 (權重) | 5e-3 (HPNet-M 的 cₐ, cₛ, wₐ, wₛ) |
| Epochs | 5 |

### Baseline 模型

**Joint Models**:
- CMLA+ (Wang et al. 2017)
- MTL-E2E (Li et al. 2019)

**Collapsed Models**:
- MATEPC (He et al. 2019)
- MNN (Li & Lu 2019)
- BERT-GLCLD (Li et al. 2020)

**Pipeline Models**:
- BERT-PT (Xu et al. 2019)

---

## 📈 實驗成果

### 主要結果 (E2E-ABSA)

#### Restaurant Dataset

| Model | F1 Score |
|-------|----------|
| CMLA+ | 39.18% |
| MTL-E2E | 64.44% |
| MATEPC | 63.13% |
| MNN | 70.98% |
| BERT-PT | 71.47% |
| BERT-GLCLD | 72.16% |
| **HPNet-S(9,12)** | **73.23%** |
| **HPNet-M** | **73.28%** ⭐ |

#### Laptop Dataset

| Model | F1 Score |
|-------|----------|
| CMLA+ | 30.09% |
| MTL-E2E | 55.59% |
| MATEPC | 47.99% |
| MNN | 58.90% |
| BERT-PT | 56.90% |
| BERT-GLCLD | 57.27% |
| **HPNet-S(9,12)** | **59.25%** |
| **HPNet-M** | **59.33%** ⭐ |

#### Twitter Dataset (10-fold CV)

| Model | F1 Score |
|-------|----------|
| CMLA+ | 40.14% |
| MTL-E2E | 52.48% |
| MATEPC | 50.74% |
| MNN | 55.97% |
| BERT-PT | 57.77% |
| **HPNet-S(9,12)** | **58.97%** |
| **HPNet-M** | **59.21%** ⭐ |

### 單任務性能

#### Aspect Extraction (AE)

| Model | Restaurant | Laptop |
|-------|-----------|--------|
| MTL-E2E | 83.36% | 78.57% |
| CNN + WIN | 88.21% | 83.27% |
| BERT-GLCLD | **91.14%** | 77.42% |
| BAT | 81.50% | 85.57% |
| HPNet-S(9,12) | 88.69% | 84.49% |
| **HPNet-M** | 87.65% | **86.13%** ⭐ |

#### Polarity Classification (PC)

| Model | Restaurant | Laptop |
|-------|-----------|--------|
| BiGCN | 73.48% | 71.84% |
| G-ATT-U | 72.65% | 72.23% |
| MAN | 71.31% | 73.20% |
| BAT | 79.24% | 76.50% |
| HPNet-S(9,12) | 79.04% | 72.67% |
| **HPNet-M** | **79.34%** ⭐ | **76.65%** ⭐ |

### 使用 BERT-large 的結果

| Model | Restaurant | Laptop | Twitter |
|-------|-----------|--------|---------|
| HPNet-S (base) | 73.23% | 59.25% | 58.97% |
| HPNet-S (large) | 74.45% (+1.22%) | 60.55% (+1.30%) | 59.44% (+0.47%) |
| HPNet-M (base) | 73.28% | 59.33% | 59.21% |
| HPNet-M (large) | 74.61% (+1.33%) | 60.53% (+1.20%) | 59.52% (+0.31%) |

---

## 📊 評估指標

### 主要指標

**F1 Score (Macro)**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

Macro-F1 = (F1_pos + F1_neg + F1_neu) / 3
```

### E2E-ABSA 評估標準

**完全匹配 (Exact Match)**:
- Aspect boundary 必須完全正確 (B, I, O tags)
- Sentiment polarity 必須完全正確
- 只有兩者都對才算正確

**範例**:
```
Gold:  "Great [food]_pos but the [service]_neg was dreadful"
Pred:  "Great [food]_pos but the [service]_neg was dreadful"
→ 完全正確 ✅

Pred:  "Great [food]_neg but the [service]_neg was dreadful"  
→ sentiment 錯誤 ❌

Pred:  "Great [food and]_pos but the [service]_neg was dreadful"
→ boundary 錯誤 ❌
```

---

## 🔍 深入分析

### 1. HPNet-S 的層選擇實驗

**Restaurant Dataset**:

| AE Layer | PC Layer | F1 Score |
|----------|----------|----------|
| 6 | 12 | 72.14% |
| 9 | 12 | **73.23%** ⭐ |
| 12 | 12 | 71.66% |

**結論**: Layer 9 (syntactic) + Layer 12 (semantic) 組合最佳

### 2. HPNet-M 的權重分布可視化

**Restaurant Dataset**:
- **AE weights**: Layer 9 權重最高
- **PC weights**: Layer 12 權重最高
- 與 HPNet-S 的最佳配置一致 ✅

**Laptop Dataset**:
- **AE weights**: Layer 9 權重最高
- **PC weights**: Layer 8 權重最高
- 顯示 PC 可能在 AE 之前學好

**Twitter Dataset**:
- **AE weights**: Layer 6, 9 權重高
- **PC weights**: 分散在所有層
- 符合 Tenney et al. "semantics spread across entire model"

### 3. Attention 權重可視化

**觀察 (以 "Great food but the service was dreadful!" 為例)**:

- **Layer 1**: 隨機且均勻分布
- **Layer 9**: 
  - Attention 集中在 [SEP] token
  - 符合 "vertical pattern" (Kovaleva et al.)
  - 表示學習 syntactic information
- **Layer 10**: 
  - "food" 關注 "great"
  - "service" 關注 "dreadful"
  - 開始追蹤 semantic relations
- **Layer 12**: 
  - 關注標點符號和 [SEP]
  - 已完成所有 syntactic/semantic 處理

### 4. Ablation Study

**移除 Joint Training 的影響**:

| Model | AE F1 | PC F1 | E2E F1 |
|-------|-------|-------|--------|
| HPNet-M (Full) | 87.65% | 79.34% | 73.28% |
| HPNet-M (-AE Joint) | - | 78.69% | - |
| HPNet-M (-PC Joint) | 86.98% | - | - |

**結論**: Joint training 對兩個子任務都有幫助

---

## 💡 關鍵洞察

### 1. BERT 層級的語言學特性

| 層級 | 特性 | 任務適合度 |
|------|------|-----------|
| **Lower (1-4)** | Linear word order | - |
| **Middle (5-9)** | **Syntactic info** | **Aspect Extraction** ⭐ |
| **Higher (10-12)** | **Semantic info** | **Polarity Classification** ⭐ |

### 2. Joint Model 的優勢

✅ **優點**:
- 共享表示，減少參數
- 兩個任務互相促進
- 避免 pipeline 的誤差傳播

✅ **HPNet 的改進**:
- 平行執行 → 提升吞吐量
- 階層式設計 → 任務特定層選擇
- 解決 target-polarity mismatch

### 3. 為什麼 HPNet-M 優於 HPNet-S?

1. **更靈活**: 自動學習權重 vs. 手動選層
2. **更全面**: 利用所有層 vs. 只用特定層
3. **更泛化**: 不同數據集自動適應
4. **但代價**: 更多參數 (每個任務 12 個權重 + 2 個 scalar)

---

## 🔄 與你的研究對比

### 相似之處

| 特性 | HPNet | 你的研究 |
|------|-------|---------|
| 利用 BERT 階層特徵 | ✅ | ✅ |
| 認識層級語義差異 | ✅ | ✅ |
| 引用相同理論基礎 | Jawahar, Tenney | Jawahar, Tenney |
| 有權重學習機制 | HPNet-M | HBL (已放棄) |

### 關鍵差異

| 維度 | HPNet | 你的研究 |
|------|-------|---------|
| **任務** | E2E-ABSA (AE + PC) | Aspect-Level Classification |
| **子任務數** | 2 個 | 1 個 |
| **Aspect 來源** | 模型預測 | 已知/給定 |
| **層級設計** | Task-specific | Unified semantic hierarchy |
| **研究重點** | Parallel execution | **Fusion strategies** ⭐ |
| **應用場景** | 混合場景 | **100% 多面向** ⭐ |

### 你的獨特貢獻 ⭐

1. ✅ **系統性融合策略比較** (4種方法)
   - Concatenation
   - Weighted Average
   - Gated Fusion
   - Multi-head Fusion

2. ✅ **統一的語義層級劃分**
   - Low/Mid/High = 詞法/語義/任務

3. ✅ **多面向場景專門優化**
   - MAMS: 100% 多面向

4. ✅ **深入分析**
   - Ablation study (單層級貢獻)
   - 層級對不同情感類別的影響

---

## 📚 參考文獻引用格式

```bibtex
@article{xiao2021hpnet,
  title={A hierarchical and parallel framework for End-to-End Aspect-based Sentiment Analysis},
  author={Xiao, Ding and Ren, Feiyang and Pang, Xiaoxuan and Cai, Ming and Wang, Qianyu and He, Ming and Peng, Jiawei and Fu, Hao},
  journal={Neurocomputing},
  volume={465},
  pages={549--560},
  year={2021},
  publisher={Elsevier}
}
```

---

## 📝 論文撰寫時如何引用

### Related Work 段落範例

```markdown
Chen et al. (2021) 提出 HPNet，為 End-to-End ABSA 設計了階層式框架。
HPNet 同時處理 aspect extraction 和 sentiment classification 兩個子任務，
並為這兩個任務分別學習可訓練的層級權重。他們的 HPNet-M 模型證明了
BERT 不同層對 syntactic 和 semantic 任務的不同貢獻。

我們的工作與 HPNet 的關鍵區別在於：(1) 任務範圍不同，我們專注於
aspect-level sentiment classification (aspect 已知)，而 HPNet 是聯合
學習框架；(2) 研究重點不同，我們系統性比較了 4 種融合策略，而 HPNet
主要探索 joint model 的平行執行機制；(3) 層級設計哲學不同，我們提出
統一的語義層級劃分，而 HPNet 採用 task-specific 層選擇。
```

---

## ✅ 快速總結

| 項目 | 內容 |
|------|------|
| **任務** | End-to-End ABSA (Aspect Extraction + Polarity Classification) |
| **核心方法** | HPNet-S (手動選層) + HPNet-M (學習權重) |
| **關鍵創新** | 階層式設計 + 平行執行 + 聯合學習 |
| **數據集** | Restaurant, Laptop, Twitter |
| **最佳結果** | Restaurant: 73.28%, Laptop: 59.33%, Twitter: 59.21% |
| **評估指標** | Macro F1 Score (exact match) |
| **發表年份** | 2021 |
| **影響力** | 首個在 E2E-ABSA 上超越 pipeline/collapsed 的 joint model |

---

**文檔建立時間**: 2025-11-21
**最後更新**: 2025-11-21
