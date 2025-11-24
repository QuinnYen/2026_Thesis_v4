# Hierarchical BERT 改進方案

> 文檔日期: 2025-11-24
> 狀態: 待實作

---

## 一、當前架構問題診斷

### 1.1 實驗結果分析

| 數據集 | Hierarchical | 最佳模型 | 差距 | 主要弱點 |
|--------|-------------|---------|------|---------|
| **Restaurants** | 72.97% | 自己 | - | Neutral F1 僅 53.30% |
| **Laptops** | 70.56% | HSA (72.33%) | -1.77% | Neutral F1 僅 52.20% |
| **MAMS** | 83.29% | IARN (84.56%) | -1.27% | 無 aspect 交互 |

**核心問題**：
1. **Neutral 類別持續偏弱** - 三個數據集中 Neutral F1 都是最差類別
2. **缺乏 aspect 交互** - 在多 aspect 場景（MAMS）輸給 IARN
3. **固定層級權重** - 硬編碼 Low/Mid/High 組合，無法自適應

### 1.2 架構層面的限制

```
當前架構:
  BERT [CLS] from layers 1-4  →  Concat  →  Linear(3072→768)  →┐
  BERT [CLS] from layers 5-8  →  Concat  →  Linear(3072→768)  →├→ Concat(2304) → Classifier
  BERT [CLS] from layers 9-12 →  Concat  →  Linear(3072→768)  →┘
```

**問題**：
1. **只用 [CLS] token** - 忽略了序列中的 aspect-specific 信息
2. **固定拼接** - 三個層級權重固定為 1:1:1
3. **獨立處理每個 aspect** - 無法利用句子中其他 aspect 的信息

---

## 二、改進方向分析

### 方向 1: Layer-wise Attention (動態層級權重)

**原理**：參考 UDify (EMNLP 2019)，學習每層 BERT 的重要性權重

```
改進後:
  Low features  →  α₁ ×─┐
  Mid features  →  α₂ ×─┼→ Weighted Sum → Classifier
  High features →  α₃ ×─┘

  其中 [α₁, α₂, α₃] = softmax([w₁, w₂, w₃])  # 可學習參數
```

**優點**：
- 減少分類器參數（2304→768）
- 模型自動學習哪個層級最重要
- 提供可解釋性（權重可視化）

**預期效果**：+0.5~1.0% Macro-F1

**實現參考**：`HierarchicalBERT_LayerAttn` 類已有基礎實現

---

### 方向 2: Aspect-aware Pooling (Aspect 感知池化)

**問題**：當前只用 [CLS] token，忽略了 aspect 在句子中的位置信息

**改進方案**：

```python
# 方案 A: Aspect Span Pooling
aspect_start, aspect_end = aspect_positions
aspect_features = hidden_states[:, aspect_start:aspect_end, :].mean(dim=1)

# 方案 B: Aspect-guided Attention
query = aspect_embedding
attention_scores = torch.matmul(query, hidden_states.transpose(-1, -2))
weighted_features = torch.matmul(attention_scores, hidden_states)
```

**優點**：
- 捕捉 aspect 周圍的上下文
- 增強對修飾語的敏感度（"not good" vs "very good"）

**預期效果**：+1.0~2.0% Macro-F1，特別是 Neutral 類別

---

### 方向 3: Cross-level Interaction (跨層級交互)

**問題**：當前三個層級完全獨立處理，無信息交流

**改進方案**：

```
     Low ──┐         ┌── Low'
           ├→ Cross-Attention →┤
     Mid ──┤         ├── Mid'
           │         │
     High ─┘         └── High'
```

```python
# 讓 High-level 特徵能夠「回看」Low-level 的詞彙信息
cross_attn_output, _ = self.cross_attention(
    query=high_features,    # 高層特徵作為 query
    key=low_features,       # 低層特徵作為 key
    value=low_features      # 低層特徵作為 value
)
enhanced_high = high_features + cross_attn_output
```

**優點**：
- 高層任務特徵可以選擇性地利用低層詞彙信息
- 處理否定詞（not, never）更有效

**預期效果**：+0.5~1.5% Macro-F1

---

### 方向 4: Multi-scale Feature Aggregation (多尺度特徵聚合)

**問題**：固定的 4-4-4 層分組可能不是最優

**改進方案**：滑動窗口 + 可學習聚合

```python
# 對每個連續的 k 層做局部聚合
window_size = 3
local_features = []
for i in range(0, 12 - window_size + 1, 2):  # [0-2, 2-4, 4-6, 6-8, 8-10, 10-12]
    local = torch.stack(all_hidden_states[i:i+window_size]).mean(dim=0)
    local_features.append(local)

# 然後用 attention 聚合
aggregated = self.scale_attention(local_features)
```

**優點**：
- 更細粒度的層級表示
- 覆蓋「層級邊界」的特徵

---

### 方向 5: 輔助損失 (Auxiliary Loss)

**問題**：只有最終分類損失，中間層級缺乏監督

**改進方案**：為每個層級添加輔助分類頭

```python
# 每個層級都有獨立的分類器
low_logits = self.low_classifier(low_features)
mid_logits = self.mid_classifier(mid_features)
high_logits = self.high_classifier(high_features)
final_logits = self.final_classifier(concat_features)

# 聯合損失
loss = (
    main_loss(final_logits, labels) +
    0.3 * aux_loss(low_logits, labels) +
    0.3 * aux_loss(mid_logits, labels) +
    0.3 * aux_loss(high_logits, labels)
)
```

**優點**：
- 強化每個層級的表示學習
- 防止梯度消失
- 類似 Deep Supervision 的效果

**預期效果**：+0.5~1.0% Macro-F1

---

## 三、推薦的改進組合

根據投入產出比和實現複雜度：

| 優先級 | 改進方向 | 預期效果 | 實現難度 | 理由 |
|-------|---------|---------|---------|-----|
| **P0** | Layer-wise Attention | +0.5~1.0% | 低 | 已有 HBL 實現可參考，改動最小 |
| **P1** | Aspect-aware Pooling | +1.0~2.0% | 中 | 對 Neutral 類別提升最顯著 |
| **P2** | 輔助損失 | +0.5~1.0% | 低 | 無需改架構，只加損失項 |
| **P3** | Cross-level Interaction | +0.5~1.5% | 中 | 需要額外注意力模組 |

---

## 四、具體實現計畫

### Hierarchical BERT v2 (HB-v2) 整合架構

```
架構設計:

1. Multi-Level Feature Extraction
   ├── Low (1-4): [CLS] concat → Linear → 768
   ├── Mid (5-8): [CLS] concat → Linear → 768
   └── High (9-12): [CLS] concat → Linear → 768

2. Aspect-aware Refinement (新增)
   ├── Aspect query = BERT[aspect_positions].mean()
   └── Refined_features = CrossAttention(query=aspect, kv=level_features)

3. Layer-wise Attention (新增)
   ├── α = softmax([w_low, w_mid, w_high])
   └── Aggregated = α₁×Low + α₂×Mid + α₃×High

4. Classification with Auxiliary Loss (新增)
   ├── Main: final_logits = Classifier(Aggregated)
   ├── Aux1: low_logits = AuxClassifier(Low)
   ├── Aux2: mid_logits = AuxClassifier(Mid)
   └── Aux3: high_logits = AuxClassifier(High)
```

---

## 五、決策點

在實作前需確認：

### 5.1 Layer-wise Attention 的實現方式
- **選項 A**: 全局可學習參數（所有樣本共享權重）
- **選項 B**: 樣本相關的動態權重（使用 attention 機制）

### 5.2 Aspect-aware Pooling 的實現
- **選項 A**: 使用 aspect 位置的 span pooling
- **選項 B**: 使用 aspect 作為 query 的 attention pooling

### 5.3 輔助損失的權重
- 固定權重 (如 0.3)
- 可學習權重
- 隨訓練進度衰減

---

## 六、參考文獻

1. **UDify** (Kondratyuk & Straka, EMNLP 2019) - Layer-wise Attention
2. **Deep Supervision** - Auxiliary Loss 設計
3. **Aspect-aware Pooling** - ABSA 領域標準做法

---

## 附錄：已嘗試但效果有限的方法

### Supervised Contrastive Learning

- **實驗結果**：Neutral F1 只從 0.43 提升到 0.47（未達預期）
- **原因分析**：Neutral 類別語義模糊，難以形成有效的 positive pairs
- **結論**：已禁用 (`contrastive_weight: 0.0`)
