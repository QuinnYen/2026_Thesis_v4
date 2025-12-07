# HKGAN: Hierarchical Knowledge-enhanced Graph Attention Network

## 概述

HKGAN (v3.0) 是一個用於面向切面情感分析 (ABSA) 的新穎架構，結合了階層式 BERT 特徵、SenticNet 情感知識和圖注意力機制。

### 版本演進

| 版本 | 核心改進 | 解決問題 |
|------|----------|----------|
| v2.0 | Domain Filtering、Confidence Gating、Coverage Mask | Neutral 識別問題 |
| v2.1 | ConfidenceGate 公式修復、移除雙重抑制 | 門控範圍壓縮問題 |
| v2.2 | 情感隔離機制 (Sentiment Isolation) | 情感洩漏問題 |
| v2.3 | 情感感知隔離 (Sentiment-Aware Isolation) | 蹺蹺板效應 |
| **v3.0** | **動態知識門控 (Dynamic Knowledge Gating)** | **MAMS 複雜句問題** |

## 架構圖

```
                                    HKGAN Architecture (v3.0)
================================================================================

Input: [CLS] text [SEP] aspect [SEP]
              |
              v
+-----------------------------------------------------------------------------------+
|                              BERT Encoder (12 layers)                             |
|  +-----------------------------------------------------------------------+        |
|  |  Layer 1-4 (Low)    |  Layer 5-8 (Mid)    |  Layer 9-12 (High)        |        |
|  |  Lexical/Syntactic  |  Semantic           |  Task-specific            |        |
|  +-----------------------------------------------------------------------+        |
+-----------------------------------------------------------------------------------+
              |                        |                        |
              v                        v                        v
+----------------------+      +----------------------+      +----------------------+
|   Low Fusion         |      |   Mid Fusion         |      |   High Fusion        |
| Linear(768*4 -> 768) |      | Linear(768*4 -> 768) |      | Linear(768*4 -> 768) |
|   + LayerNorm        |      |   + LayerNorm        |      |   + LayerNorm        |
|   + GELU             |      |   + GELU             |      |   + GELU             |
+----------------------+      +----------------------+      +----------------------+
              |                        |                        |
              +------------------------+------------------------+
                                       |
                                       v
                    +------------------------------------------+
                    |          SenticNet Knowledge             |
                    |    token_id -> polarity_value [-1, 1]    |
                    +------------------------------------------+
                                       |
                                       v
                    +------------------------------------------+
                    |       Confidence Gate (v2.1)             |
                    |  gate = σ((Wh + bias) / temperature)     |
                    |  gated_polarity = gate * raw_polarity    |
                    +------------------------------------------+
                                       |
                                       v
                    +------------------------------------------+
                    |     Dynamic Knowledge Gate (v3.0)        |
                    |  Gate = σ(Linear([BERT, SenticNet]))     |
                    |  Feature = (1-Gate)*BERT + Gate*Senti    |
                    +------------------------------------------+
                                       |
              +------------------------+------------------------+
              |                        |                        |
              v                        v                        v
+-------------------+      +-------------------+      +-------------------+
|  Hierarchical GAT |      |  Hierarchical GAT |      |  Hierarchical GAT |
|  (Low Features)   |      |  (Mid Features)   |      |  (High Features)  |
+-------------------+      +-------------------+      +-------------------+
              |                        |                        |
              v                        v                        v
+-----------------------------------------------------------------------------------+
|                        Hierarchical GAT Layer (Detail)                            |
|  +-------------------------------------------------------------------------+      |
|  |                                                                         |      |
|  |   Level 1: Token-level     Level 2: Phrase-level    Level 3: Clause     |      |
|  |   (window=3)               (window=5)               (fully connected)   |      |
|  |        |                        |                        |              |      |
|  |        v                        v                        v              |      |
|  |   +-----------+            +-----------+            +-----------+       |      |
|  |   | Knowledge |            | Knowledge |            | Knowledge |       |      |
|  |   | Enhanced  |            | Enhanced  |            | Enhanced  |       |      |
|  |   |   GAT     |            |   GAT     |            |   GAT     |       |      |
|  |   +-----------+            +-----------+            +-----------+       |      |
|  |        |                        |                        |              |      |
|  |        +------------------------+------------------------+              |      |
|  |                                 |                                       |      |
|  |                                 v                                       |      |
|  |                    +-----------------------+                            |      |
|  |                    |    Level Fusion       |                            |      |
|  |                    | Weighted Sum + Concat |                            |      |
|  |                    +-----------------------+                            |      |
|  +-------------------------------------------------------------------------+      |
+-----------------------------------------------------------------------------------+
              |                        |                        |
              v                        v                        v
         [CLS] Pool              [CLS] Pool              [CLS] Pool
              |                        |                        |
              +------------------------+------------------------+
                                       |
                                       v
                    +------------------------------------------+
                    |         Cross-Level Fusion               |
                    |   w0*Low + w1*Mid + w2*High (weighted)   |
                    |   + Concat([Low, Mid, High]) -> Linear   |
                    +------------------------------------------+
                                       |
                                       v
                    +------------------------------------------+
                    |        Aspect Features                   |
                    |      [batch, max_aspects, 768]           |
                    +------------------------------------------+
                                       |
                                       v
+-----------------------------------------------------------------------------------+
|                  Inter-Aspect Attention + Sentiment-Aware Isolation (v2.3+)       |
|  +-------------------------------------------------------------------------+      |
|  |                                                                         |      |
|  |   1. Multi-Head Self-Attention across aspects                           |      |
|  |      Aspect_1 <-----> Aspect_2 <-----> Aspect_3                         |      |
|  |                                                                         |      |
|  |   2. Sentiment Prediction (lightweight predictor)                       |      |
|  |      self_sentiment = softmax(predictor(aspect_features))               |      |
|  |      context_sentiment = softmax(predictor(context_features))           |      |
|  |                                                                         |      |
|  |   3. Sentiment Consistency (consistency calculation)                    |      |
|  |      consistency = sum(self_sentiment * context_sentiment)              |      |
|  |                                                                         |      |
|  |   4. Sentiment-Aware Isolation (dynamic isolation)                      |      |
|  |      base_isolation = isolation_gate(aspect_features)                   |      |
|  |      adjusted = base * (1 - strength * consistency)                     |      |
|  |      effective = adjusted * (1 - base) + base                           |      |
|  |                                                                         |      |
|  |   5. Gated Fusion                                                       |      |
|  |      self_weight = effective + (1-effective) * relation_gate            |      |
|  |      context_weight = (1-effective) * (1-relation_gate)                 |      |
|  |      output = self_weight * self + context_weight * context             |      |
|  |                                                                         |      |
|  +-------------------------------------------------------------------------+      |
+-----------------------------------------------------------------------------------+
                                       |
                                       v
                    +----------------------------------------------+
                    |           Classifier                         |
                    |   Linear(768 -> 768) + LayerNorm + GELU      |
                    |   Linear(768 -> 3)                           |
                    +----------------------------------------------+
                                       |
                                       v
                    +----------------------------------------------+
                    |   Asymmetric Logit Adjustment (at inference) |
                    |   logits[:, 1] += neutral_boost              |
                    |   logits[:, 0] -= neg_suppress               |
                    |   logits[:, 2] -= pos_suppress               |
                    +----------------------------------------------+
                                       |
                                       v
                    +----------------------------------------------+
                    |             Output                           |
                    |   logits: [batch, max_aspects, 3]            |
                    |   (Negative, Neutral, Positive)              |
                    +----------------------------------------------+

================================================================================
```

## 核心組件

### 1. 階層式 BERT 特徵提取

BERT 的 12 層捕捉不同層次的語言資訊：

| 層級群組 | 層數 | 資訊類型 | 用途 |
|----------|------|----------|------|
| **Low** | 1-4 | 詞彙、句法 | 表面模式、詞性標註 |
| **Mid** | 5-8 | 語義 | 詞義、關係 |
| **High** | 9-12 | 任務特定 | 情感相關特徵 |

每個群組的輸出經過拼接和投影：
```
fusion(x) = GELU(LayerNorm(Linear(concat(layer_outputs))))
```

### 2. SenticNet 知識增強

SenticNet 提供詞彙的情感極性值：
- 範圍：[-1, 1]，其中 -1 = 負面，0 = 中性，+1 = 正面
- 範例："good" -> +0.8，"terrible" -> -0.9

**知識注入公式：**
```
e'_ij = e_ij + λ * knowledge_bias(polarity_i, polarity_j)
```

其中 `knowledge_bias` 由 MLP 學習：
```python
knowledge_bias = MLP([polarity_i, polarity_j])  # 2 -> 32 -> 1
```

### 3. 信心門控機制 (Confidence Gate) - v2.1

**問題背景**：SenticNet 提供的是「通用領域」的情感極性，在特定語境下可能是噪聲。

**範例**："high resolution" 中的 "high" 在 SenticNet 中有正向極性，但在技術描述中是中性的。

**解決方案**：讓 BERT 的上下文表示學習一個 gate ∈ [0, 1]
- gate ≈ 1：信任 SenticNet（這是情感表達）
- gate ≈ 0：忽略 SenticNet（這是客觀陳述）

**公式**：
```
gate = σ((W₂ · ReLU(W₁ · h + b₁) + b₂ + bias) / temperature)
effective_polarity = gate * senticnet_polarity
```

### 4. 動態知識門控 (Dynamic Knowledge Gating) - v3.0

**問題背景（MAMS 複雜句）**：v2.x 的知識注入是「硬性」的固定權重，在 MAMS 含轉折句時會造成衝突。

**範例**：
- "The food is great, but the service is terrible"
- v2.x：`great` 的正向極性會污染 `service` 的情感判斷
- v3.0：動態門控會學習在這種情況下降低對 SenticNet 的依賴

**舊版 v2.x（硬性注入）**：
```
Feature_new = Feature_BERT + λ * SenticNet（固定權重）
```

**新版 v3.0（軟性門控）**：
```python
# Gate 由 BERT 和 SenticNet 共同決定
Gate = Sigmoid(Linear([BERT_feature, SenticNet_embed]))

# 動態混合
Feature_new = (1 - Gate) * BERT_feature + Gate * SenticNet_embed
```

**行為**：
- 簡單句（Laptops/Restaurants）：Gate 高，利用 SenticNet 增強
- 複雜句（MAMS 轉折句 "But", "However"）：Gate 低，只相信 BERT

**預期效果**：MAMS 資料集 F1 提升至 84%+

### 5. 階層式圖注意力

三個層級的圖注意力，具有不同的感受野：

| 層級 | 窗口大小 | 捕捉內容 |
|------|----------|----------|
| Token | 3 | 局部依賴（相鄰詞） |
| Phrase | 5 | 短程模式（片語） |
| Clause | 完整 | 長程依賴（整個句子） |

**GAT 公式：**
```
e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
α_ij = softmax_j(e_ij)
h'_i = ELU(Σ_j α_ij * Wh_j)
```

**多頭注意力：**
```python
output = Concat(head_1, head_2, ..., head_k) @ W_o
```

### 6. 情感感知隔離機制 (Sentiment-Aware Isolation) - v2.3

**問題背景（蹺蹺板效應）**：v2.2 的隔離機制「盲目隔離」，可能阻擋有益的情感增強。

**範例**：
- "battery drains quickly" → `battery` 應該獲得 `quickly` 的負面增強
- "battery bad, screen 15 inches" → `bad` 不應該污染 `screen`

**解決方案**：根據「情感一致性」動態調整隔離程度
- 情感一致 → 允許流入（增強）
- 情感衝突 → 阻擋流入（隔離）

**公式**：
```python
# Step 1: 預測情感傾向
self_sentiment = softmax(predictor(aspect_features))      # [batch, aspects, 3]
context_sentiment = softmax(predictor(context_features))  # [batch, aspects, 3]

# Step 2: 計算情感一致性（點積）
consistency = sum(self_sentiment * context_sentiment)     # 範圍 [0.33, 1]

# Step 3: 根據一致性調整隔離
adjusted_isolation = base_isolation * (1 - consistency_strength * consistency)
effective_isolation = adjusted_isolation * (1 - base_isolation) + base_isolation

# Step 4: 組合門控
self_weight = effective_isolation + (1 - effective_isolation) * relation_gate
context_weight = (1 - effective_isolation) * (1 - relation_gate)
output = self_weight * self + context_weight * context
```

### 7. 非對稱 Logit 調整 (Asymmetric Logit Adjustment)

**問題**：Neutral 樣本容易被誤判為 Negative 或 Positive。

**解決方案**：推理時對 logits 進行非對稱調整：

```python
# 三個參數分別針對不同錯誤類型
adjusted_logits = logits.clone()
adjusted_logits[:, :, 1] += neutral_boost  # 解決 Neu 識別不足
adjusted_logits[:, :, 0] -= neg_suppress   # 解決 Neu→Neg
adjusted_logits[:, :, 2] -= pos_suppress   # 解決 Neu→Pos
preds = torch.argmax(adjusted_logits, dim=-1)
```

**數學效果**：
- 原本：L_Neg > L_Neu（被誤判為 Neg）
- 調整後：(L_Neg - neg_suppress) vs (L_Neu + neutral_boost)
- 等效翻轉力 = neutral_boost + neg_suppress

**資料集對應參數**：
| 資料集 | neutral_boost | neg_suppress | pos_suppress | 說明 |
|--------|---------------|--------------|--------------|------|
| **laptops** | 0.8 | 0.6 | 0.0 | 筆電領域：Neutral 比例高 |
| **lap16** | 0.8 | 0.6 | 0.0 | SemEval-2016 Laptops |
| **restaurants** | 0.6 | 0.0 | 0.8 | 餐廳領域：Positive 偏高 |
| **rest16** | 1.0 | 0.2 | 0.5 | SemEval-2016：Neutral F1 低 |
| **mams** | 0.0 | 0.0 | 0.0 | MAMS：歸零測試原始輸出 |

**優勢**：
- `neutral_boost` 解決 Neutral 識別不足
- `neg_suppress` 解決 Neu→Neg，不影響 Pos→Neu
- `pos_suppress` 解決 Neu→Pos（餐廳領域 Yelp 正向偏誤）

## 模型參數

| 組件 | 參數量 | 說明 |
|------|--------|------|
| BERT | ~110M | 預訓練語言模型 |
| 融合層 | 3 × (768×4 → 768) | ~7M |
| GAT 層 | 3 層級 × 4 頭 | ~2M |
| Confidence Gate | 768 → 192 → 1 | ~0.15M |
| Dynamic Knowledge Gate | (768+1) → 1 | ~0.8K |
| Sentiment Predictor | 768 → 192 → 3 | ~0.15M |
| IARN + Isolation | 4 頭注意力 + 門控 | ~1.5M |
| 分類器 | 768 → 768 → 3 | ~0.6M |
| **總計** | **~121M** | |

## 訓練配置

```yaml
model:
  improved: "hkgan"
  # DAPT 模型對應表：根據資料集自動選擇
  bert_model:
    laptops: "data/dapt/laptop_dapt/final"
    lap16: "data/dapt/laptop_dapt/final"
    restaurants: "data/dapt/restaurant_dapt/final"
    rest16: "data/dapt/restaurant_dapt/final"
    default: "bert-base-uncased"
  hidden_dim: 768
  dropout: 0.3
  gat_heads: 4
  gat_layers: 2
  use_senticnet: true
  knowledge_weight: 0.1
  use_confidence_gate: true
  use_dynamic_gate: true  # v3.0 新增

training:
  batch_size: 16
  accumulation_steps: 2  # 等效 batch = 32
  lr: 3e-5
  epochs: 30
  loss_type: "focal"
  focal_gamma: 2.0
  class_weights: [0.8, 1.8, 0.8]  # 提升 Neutral

  # 監督式對比學習
  contrastive_weight: 0.1
  contrastive_temperature: 0.07

  # 非對稱 Logit 調整（推理時，資料集對應）
  neutral_boost:
    laptops: 0.8
    lap16: 0.8
    restaurants: 0.6
    rest16: 1.0
    mams: 0.0
    default: 0.0

  neg_suppress:
    laptops: 0.6
    lap16: 0.6
    restaurants: 0.0
    rest16: 0.2
    mams: 0.0
    default: 0.0

  pos_suppress:
    laptops: 0.0
    lap16: 0.0
    restaurants: 0.8
    rest16: 0.5
    mams: 0.0
    default: 0.0

  # Layer-wise Learning Rate Decay
  use_llrd: true
  llrd_decay: 0.95
```

## Domain-Adaptive Pre-training (DAPT)

DAPT 是一種領域自適應預訓練技術，讓 BERT 在目標領域的無標籤資料上繼續進行 MLM（Masked Language Modeling）訓練，使其更好地理解領域特定的語言模式。

### DAPT 訓練流程

```
+-------------------------------------------------------------+
|  Step 1: Prepare Domain Unlabeled Data                      |
|  -----------------------------------------------------------+
|  Laptop domain: Amazon Electronics reviews                  |
|  Restaurant domain: Yelp Restaurant reviews                 |
+-------------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  Step 2: DAPT Pre-training (MLM Task)                       |
|  -----------------------------------------------------------+
|  python data/domain_pretrain.py --dataset laptops --epochs 3|
|  python data/domain_pretrain.py --dataset restaurants       |
|                                                             |
|  Output:                                                    |
|    - data/dapt/laptop_dapt/final/                           |
|    - data/dapt/restaurant_dapt/final/                       |
+-------------------------------------------------------------+
                           |
                           v
+-------------------------------------------------------------+
|  Step 3: Train ABSA with DAPT BERT                          |
|  -----------------------------------------------------------+
|  Config auto-selects corresponding DAPT model by dataset    |
+-------------------------------------------------------------+
```

### DAPT 配置說明

配置檔 (`configs/unified_hkgan.yaml`) 支援根據資料集自動選擇 BERT 模型：

```yaml
model:
  bert_model:
    laptops: "data/dapt/laptop_dapt/final"      # SemEval-2014 Laptops
    lap16: "data/dapt/laptop_dapt/final"        # SemEval-2016 Laptops
    restaurants: "data/dapt/restaurant_dapt/final"  # SemEval-2014 Restaurants
    rest16: "data/dapt/restaurant_dapt/final"   # SemEval-2016 Restaurants
    default: "bert-base-uncased"                # MAMS 或其他資料集
```

### 無標籤資料來源

| 領域 | 資料來源 | 檔案路徑 | 樣本數 |
|------|----------|----------|--------|
| **筆電** | Amazon Electronics 5-core | `data/unlabeled/Electronics_5.json` | ~50,000 |
| **餐廳** | Yelp Academic Dataset | `data/unlabeled/Yelp/yelp_restaurant_corpus.txt` | ~500,000 |

### 餐廳資料提取

由於 Yelp 原始資料包含多種商家類型，需要先提取餐廳評論：

```bash
# 從 Yelp 資料集提取餐廳評論
python data/extract_yelp_restaurants.py --max_samples 500000
```

提取流程：
1. 從 `yelp_academic_dataset_business.json` 篩選 `categories` 包含 "Restaurants" 或 "Food" 的商家
2. 從 `yelp_academic_dataset_review.json` 提取對應評論
3. 使用 Reservoir Sampling 隨機抽樣，避免記憶體爆炸

### DAPT 訓練參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `epochs` | 3 | MLM 訓練輪數 |
| `batch_size` | 16 | 批次大小 |
| `max_samples` | 50,000 | 最大訓練樣本數 |
| `mlm_probability` | 0.15 | Token 遮蔽比例 |
| `learning_rate` | 5e-5 | 學習率 |
| `max_length` | 128 | 最大序列長度 |

### DAPT 執行指令

```bash
# 筆電領域 DAPT
python data/domain_pretrain.py --dataset laptops --epochs 3

# 餐廳領域 DAPT
python data/domain_pretrain.py --dataset restaurants --epochs 3

# 自訂無標籤資料路徑
python data/domain_pretrain.py --dataset restaurants \
       --unlabeled-path data/unlabeled/Yelp/yelp_restaurant_corpus.txt
```

### DAPT 的效果

DAPT 讓 BERT 學習領域特定的語言模式，例如：

| 領域 | 學習內容 |
|------|----------|
| **筆電** | "battery drains", "screen resolution", "SSD speed" 等技術用語的語境 |
| **餐廳** | "tasty", "service", "ambience", "portion size" 等餐飲用語的語境 |

這使得後續的 ABSA 任務能更準確地理解領域特定的情感表達。

## 資料流程摘要

```
1. 輸入分詞
   "[CLS] The battery life is amazing [SEP] battery life [SEP]"

2. BERT 編碼
   → 12 層隱藏狀態 [batch, seq_len, 768]

3. 階層式特徵提取
   → Low/Mid/High 特徵 [batch, seq_len, 768] × 3

4. SenticNet 查詢 + Confidence Gate
   → 門控後的極性值 [batch, seq_len]

5. 階層式 GAT 處理
   → 知識增強的圖特徵 [batch, seq_len, 768] × 3

6. 跨層融合
   → 組合特徵 [batch, 768]

7. 跨面向注意力 + 情感感知隔離
   → 上下文化特徵 [batch, max_aspects, 768]

8. 分類 + Logit 調整
   → Logits [batch, max_aspects, 3]
```

## 實驗結果

### 數據集

| 數據集 | 領域 | 訓練集 | 測試集 | 多面向比例 | Neutral 比例 |
|--------|------|--------|--------|------------|--------------|
| **SemEval-2014 Restaurants** | 餐廳評論 | 3,608 | 1,120 | ~25% | ~18% |
| **SemEval-2014 Laptops** | 筆電評論 | 2,328 | 638 | ~20% | ~27% |
| **SemEval-2016 Restaurants** | 餐廳評論 | 2,000 | 676 | ~22% | ~16% |
| **SemEval-2016 Laptops** | 筆電評論 | 2,500 | 800 | ~18% | ~25% |
| **MAMS** | 餐廳（多面向） | 4,297 | 1,336 | 100% | ~33% |

### HKGAN v3.0 vs Baseline 性能比較（5-Seed 平均）

| 數據集 | 模型 | Macro-F1 (%) | Δ F1 | p-value | Cohen's d | 顯著性 |
|--------|------|--------------|------|---------|-----------|--------|
| **Restaurants** | Baseline | 71.95 | - | - | - | - |
| | **HKGAN** | **75.66** | **+3.71** | 0.0158 | 1.80 | Yes * |
| **Laptops** | Baseline | 68.85 | - | - | - | - |
| | **HKGAN** | **71.59** | **+2.75** | 0.0741 | 1.08 | No |
| **REST16** | Baseline | 69.05 | - | - | - | - |
| | **HKGAN** | **77.81** | **+8.76** | 0.0040 | 2.67 | Yes ** |
| **LAP16** | Baseline | 66.43 | - | - | - | - |
| | **HKGAN** | **68.48** | **+2.05** | 0.0057 | 2.42 | Yes ** |
| **MAMS** | Baseline | 82.49 | - | - | - | - |
| | **HKGAN** | **83.96** | **+1.47** | 0.0062 | 2.36 | Yes ** |

**統計顯著性**: 4/5 數據集達到 p < 0.05，所有數據集的 Cohen's d > 0.8（大效果量）。

### 多種子實驗結果

驗證模型魯棒性，使用 5 個不同的隨機種子（42, 123, 2023, 999, 0）：

| 數據集 | Baseline F1 (Mean ± Std) | HKGAN F1 (Mean ± Std) | 穩定性評估 |
|--------|--------------------------|----------------------|------------|
| **Restaurants** | 71.95% ± 1.18% | 75.66% ± 1.18% | Good (CV=1.57%) |
| **Laptops** | 68.85% ± 1.56% | 71.59% ± 1.56% | Good (CV=2.18%) |
| **REST16** | 69.05% ± 0.74% | 77.81% ± 0.74% | Excellent (CV=0.96%) |
| **LAP16** | 66.43% ± 0.78% | 68.48% ± 0.78% | Excellent (CV=1.13%) |
| **MAMS** | 82.49% ± 0.59% | 83.96% ± 0.59% | Excellent (CV=0.71%) |

**穩定性評估**:
- **Excellent**: 標準差 < 1%（REST16, LAP16, MAMS）
- **Good**: 1% ≤ 標準差 < 2%（Restaurants, Laptops）
- 所有 HKGAN 模型的變異係數 (CV) 均 < 3%，證明架構設計穩健

## 實作細節

### 檔案結構

| 組件 | 檔案路徑 |
|------|----------|
| HKGAN 模型 | `models/hkgan.py` |
| 配置文件 | `configs/unified_hkgan.yaml` |
| 訓練腳本 | `experiments/train_multiaspect.py` |
| 配置驅動訓練 | `experiments/train_from_config.py` |
| SenticNet 載入器 | `utils/senticnet_loader.py` |
| 報告生成器 | `experiments/generate_hkgan_report.py` |
| 實驗執行器 | `run_experiments.py` |
| **DAPT 預訓練** | `data/domain_pretrain.py` |
| **Yelp 資料提取** | `data/extract_yelp_restaurants.py` |

### 核心類別

```python
# models/hkgan.py
class GraphAttentionLayer(nn.Module):
    """純 PyTorch GAT 層（不依賴 torch_geometric）"""

class MultiHeadGAT(nn.Module):
    """多頭圖注意力，可選知識增強"""

class ConfidenceGate(nn.Module):
    """信心門控機制 v2.1 - 動態決定是否信任 SenticNet"""

class DynamicKnowledgeGate(nn.Module):
    """動態知識門控 v3.0 - 軟性融合 BERT 與 SenticNet"""

class KnowledgeEnhancedGAT(nn.Module):
    """帶 SenticNet 極性注入的 GAT"""

class HierarchicalGATLayer(nn.Module):
    """三層級 GAT：Token、Phrase、Clause"""

class HKGAN(nn.Module):
    """主模型：階層式知識增強圖注意力網路 v3.0

    包含：
    - 階層式 BERT 特徵提取
    - SenticNet 知識增強 + Confidence Gate
    - Dynamic Knowledge Gate（v3.0 新增）
    - 階層式 GAT
    - Inter-Aspect Attention
    - Sentiment-Aware Isolation（情感感知隔離）
    """
```

### 超參數敏感度

| 參數 | 測試範圍 | 最佳值 | 影響 |
|------|----------|--------|------|
| `gat_heads` | 2, 4, 8 | 4 | 更多頭數改善多面向處理 |
| `gat_layers` | 1, 2, 3 | 2 | 超過 2 層效益遞減 |
| `knowledge_weight` | 0.05-0.2 | 0.1 | 過高可能過擬合 SenticNet |
| `dropout` | 0.2-0.5 | 0.3 | 正則化與容量的平衡 |
| `lr` | 2e-5, 3e-5, 5e-5 | 3e-5 | 略高於 baseline 有益 |
| `neutral_boost` | 0.0-1.5 | **0.8** | 推理時 Neutral logits 的加法偏移 |
| `neg_suppress` | 0.0-1.0 | **0.6** | 推理時 Negative logits 的減法偏移 |
| `contrastive_weight` | 0.05-0.2 | 0.1 | 對比學習損失權重 |

### SenticNet 覆蓋率分析

| 數據集 | 覆蓋率 | 備註 |
|--------|--------|------|
| Restaurants | ~65% | 餐飲/服務詞彙覆蓋良好 |
| Laptops | ~41% | 技術術語（battery、SSD、CPU）缺失 |
| MAMS | ~62% | 與 Restaurants 類似 |

**注意**：Laptops 領域的 SenticNet 覆蓋率較低是一個限制。Confidence Gate 機制可以緩解這個問題，讓模型學習何時忽略不可靠的知識。

### 損失函數細節

**1. Focal Loss** 配合類別權重處理類別不平衡：

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

其中：
- γ = 2.0（聚焦參數）
- α = [0.8, 1.8, 0.8] 對應 [Negative, Neutral, Positive]
```

**2. 監督式對比學習 (SCL)** 解決中性類別語義塌陷：

```
L_total = L_focal + λ_scl * L_contrastive

其中：
- λ_scl = 0.1
- temperature = 0.07
```

## 論文創新點總結

1. **首次將情感知識庫 (SenticNet) 與階層式 GAT 結合**
2. **雙階層設計**：BERT 內部階層 + 語言學階層（Token/Phrase/Clause）
3. **知識增強的圖注意力機制**
4. **動態信心門控**：根據上下文決定是否信任外部知識
5. **動態知識門控 (v3.0)**：軟性融合機制，解決複雜句中知識衝突問題
6. **情感感知隔離**：根據情感一致性動態調整跨面向信息流
7. **非對稱 Logit 調整**：推理時針對 Neutral 誤判進行校正，支援資料集對應參數

## 參考文獻

- Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
- Velickovic et al. "Graph Attention Networks" (ICLR 2018)
- Cambria et al. "SenticNet 5" (AAAI 2018)
- Devlin et al. "BERT" (NAACL 2019)
- Lin et al. "Focal Loss for Dense Object Detection" (ICCV 2017)
- Khosla et al. "Supervised Contrastive Learning" (NeurIPS 2020)
