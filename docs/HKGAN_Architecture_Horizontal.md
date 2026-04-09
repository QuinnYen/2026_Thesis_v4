# HKGAN Architecture — Horizontal Layout (Word 版)

> 本檔為橫向簡化版，供 Word 排版使用。完整說明請參閱 `HKGAN_Architecture.md`。

## 主流程圖

```
                        HKGAN Architecture (v3.0)
================================================================================

 ┌──────────────┐   ┌────────────────────────────────┐   ┌─────────────────────┐
 │    Input     │   │      BERT Encoder  (12L)       │   │  SenticNet Lookup   │
 │  [CLS] txt   │──►│  Layer 1-4  → Low  features    │   │  token_id → pol     │
 │  [SEP] asp   │   │  Layer 5-8  → Mid  features    │   │  range: [-1, 1]     │
 │    [SEP]     │   │  Layer 9-12 → High features    │   │  (built once)       │
 └──────────────┘   └──────────────┬─────────────────┘   └──────────┬──────────┘
                                   │ feat ×3                        │ polarities
                   ┌───────────────▼────────────────────────────────▼──────────┐
                   │      Low / Mid / High Fusion    (768×4 → 768) ×3          │
                   └──────────────────────────────┬────────────────────────────┘
                                                  │
 ┌────────────────────────────────────────────────▼─────────────────────────────┐
 │      Hierarchical GAT  [Shared Weights, Called 3× for Low / Mid / High]      │
 │                                                                              │
 │  ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐  │
 │  │   Token  (w=3)     │    │   Phrase  (w=5)    │    │   Clause  (full)   │  │
 │  │────────────────────│    │────────────────────│    │────────────────────│  │
 │  │  DynGate  (v3.0)   │    │  DynGate  (v3.0)   │    │  DynGate  (v3.0)   │  │
 │  │  (node feat  adj)  │    │  (node feat  adj)  │    │  (node feat  adj)  │  │
 │  │  ConfGate  (v2.1)  │    │  ConfGate  (v2.1)  │    │  ConfGate  (v2.1)  │  │
 │  │  (edge wts   adj)  │    │  (edge wts   adj)  │    │  (edge wts   adj)  │  │
 │  │        GAT         │    │        GAT         │    │        GAT         │  │
 │  └──────────┬─────────┘    └──────────┬─────────┘    └──────────┬─────────┘  │
 │             └──────────────────── Level Fusion ─────────────────┘            │
 └──────────────────────────────────────────────────────────────────────────────┘
         │ Low                      │ Mid                      │ High
     [CLS Pool]                [CLS Pool]                [CLS Pool]
         └──────────────────────────┴──────────────────────────┘
                                    │
 ┌──────────────────────────────────▼───────────────────────────────────────────┐
 │                            Cross-Level Fusion                                │
 │       w0×Low + w1×Mid + w2×High  +  Linear( concat[Low, Mid, High] )         │
 └──────────────────────────────────────────────────────────────────────────────┘
                                    │
 ┌──────────────────────────────────▼──────────────────┐  ┌─────────────────┐  ┌──────────┐
 │          Inter-Aspect Attn + SAI  (v2.3)            │  │   Classifier    │  │          │
 │  MHSA(4h) → SentPred → Consistency → Isolation      │─►│ Linear(768→3)   │─►│  Output  │
 │                      → Gated Fusion                 │  │  Logit Adj.     │  │ [b,A,3]  │
 └─────────────────────────────────────────────────────┘  └─────────────────┘  └──────────┘

================================================================================
```

## 主流程說明

| 步驟 | 組件 | 輸入維度 | 輸出維度 | 說明 |
|------|------|---------|---------|------|
| ① | Input | - | [b, A, seq] | 每個 aspect 一個 [CLS] text [SEP] aspect [SEP] 序列 |
| ②a | BERT (12L) | [b·A, seq] | [b·A, seq, 768] | 12 層，分 Low/Mid/High 三組 |
| ②b | SenticNet Lookup | token_ids | [b·A, seq] | 並行查詢，與 BERT 無關 |
| ③ | Low/Mid/High Fusion | [b·A, seq, 768×4] | [b·A, seq, 768] | 各層 concat → Linear → LayerNorm → GELU |
| ④ | Hierarchical GAT | [b·A, seq, 768] + polarities | [b·A, seq, 768] | 共享實例，呼叫 3 次 |
| ⑤ | CLS Pooling | [b·A, seq, 768] | [b·A, 768] | 取 position 0 的 token |
| ⑥ | Cross-Level Fusion | [b·A, 768] × 3 | [b, A, 768] | 加權和 + Linear(concat) |
| ⑦ | Inter-Aspect Attn + SAI | [b, A, 768] | [b, A, 768] | 跨面向自注意力 + 情感感知隔離 |
| ⑧ | Classifier + Logit Adj | [b, A, 768] | [b, A, 3] | Linear(768→3) + 推理時非對稱調整 |

> 符號：b = batch size，A = max_aspects，seq = seq_len

---

## Hierarchical GAT 內部展開

```
  Hierarchical GAT Layer  (Shared Weights, Called 3× with Low / Mid / High features)
  ──────────────────────────────────────────────────────────────────────────────────

  Input: BERT_feat [b·A, seq, 768]  +  polarities [b·A, seq]
       │
       ├──── Level 1 (Token, window=3) ────────────────────────────────────────────────┐
       │                                                                               │
       │     BERT_feat ──► Dynamic Knowledge Gate (v3.0)                              │
       │                   polarity ──► polarity_embed (1→192→768)                    │
       │                   Gate = σ(MLP([BERT, polarity_embed]))     ← 節點特徵調整    │
       │                   feat = (1-Gate)×BERT + Gate×polarity_embed + residual       │
       │                         │                                                     │
       │                         ▼                                                     │
       │                   Confidence Gate (v2.1)                                     │
       │                   gate = σ(MLP(feat) / temperature)         ← 邊權重調整     │
       │                   gated_pol = gate × polarity                                │
       │                         │                                                     │
       │                         ▼                                                     │
       │                   Knowledge-Enhanced GAT                                     │
       │                   adj += knowledge_bias(pol_i, pol_j)                        │
       │                   output = MultiHeadGAT(feat, adj)                           │
       │                                                                               │
       ├──── Level 2 (Phrase, window=5)  (同 Level 1 結構)  ────────────────────────── ┤
       │                                                                               │
       └──── Level 3 (Clause, full)      (同 Level 1 結構)  ────────────────────────── ┘
                                                │
                                                ▼
                                          Level Fusion
                                   Weighted Sum + Linear(concat)
                                   output: [b·A, seq, 768]
```

---

## Inter-Aspect Attention + Sentiment-Aware Isolation 內部展開

```
  Inter-Aspect Attention + Sentiment-Aware Isolation  (v2.3)
  ──────────────────────────────────────────────────────────

  Input: aspect_features [batch, max_aspects, 768]
       │
       ├─① Multi-Head Self-Attention (4 heads) ─────────────────────────────────────────
       │      Aspect₁ ↔ Aspect₂ ↔ Aspect₃   →  context_features [b, A, 768]
       │
       ├─② Sentiment Prediction ────────────────────────────────────────────────────────
       │      self_sent    = softmax(predictor(aspect_features))    [b, A, 3]
       │      context_sent = softmax(predictor(context_features))   [b, A, 3]
       │
       ├─③ Sentiment Consistency ───────────────────────────────────────────────────────
       │      consistency = sum(self_sent × context_sent)           [b, A, 1]
       │
       ├─④ Sentiment-Aware Isolation ───────────────────────────────────────────────────
       │      base_iso   = isolation_gate(aspect_features)          [b, A, 1]
       │      adjusted   = base_iso × (1 - strength × consistency)
       │      effective  = adjusted × (1 - base_iso) + base_iso
       │
       └─⑤ Gated Fusion ────────────────────────────────────────────────────────────────
              relation_g  = relation_gate([aspect, context])        [b, A, 1]
              self_w      = effective + (1-effective) × relation_g
              context_w   = (1-effective) × (1-relation_g)
              output      = self_w × aspect + context_w × context   [b, A, 768]
```
