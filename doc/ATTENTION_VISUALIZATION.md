# Attention Heatmap 視覺化說明

本文檔說明 HMAC-Net 注意力機制的視覺化功能及其使用方式。

## 📊 概述

訓練完成後，系統會自動生成多種類型的注意力熱圖，幫助理解模型如何關注不同的詞彙來進行情感分析。

## 🎯 生成的視覺化類型

### 1. 詞級注意力 (Word-Level Attention)
**檔名**: `attention_word_sample_*.png`

**功能**: 顯示模型在詞級別如何關注句子中的各個詞彙

**特點**:
- 使用多個注意力頭（64d, 128d）
- 捕捉細粒度的詞彙關聯
- 高亮顯示 aspect 相關詞彙

**解讀**:
- 顏色越深（紅色）表示注意力權重越高
- aspect 詞彙會以紅色粗體標示
- 適合分析單個詞彙的重要性

**示例場景**:
```
句子: "The food was delicious but the service was terrible"
Aspect: "food"
詞級注意力: [0.05, 0.15, 0.45, 0.25, 0.05, 0.03, 0.02]
                ↑    ↑     ↑     ↑
              The  food  was  delicious

結果: 模型主要關注 "food" 和 "delicious"
```

---

### 2. 片語級注意力 (Phrase-Level Attention)
**檔名**: `attention_phrase_sample_*.png`

**功能**: 使用 CNN 提取局部片語特徵後的注意力分布

**特點**:
- 使用多尺度卷積核（64d, 128d, 256d）
- 捕捉 2-3 個詞的組合模式
- 識別常見的情感片語

**解讀**:
- 關注片語級別的語義組合
- 能識別 "not bad", "very good" 等固定搭配
- 對否定詞敏感度更高

**示例場景**:
```
句子: "not very good"
片語級注意力能捕捉到 "not very" 和 "very good" 的組合語義
避免錯誤地只關注 "good" 而忽略否定
```

---

### 3. 句子級注意力 (Sentence-Level Attention)
**檔名**: `attention_sentence_sample_*.png`

**功能**: 使用雙向 LSTM 捕捉全局語境後的注意力

**特點**:
- 使用多個注意力頭（64d, 128d, 256d）
- 考慮整個句子的上下文
- 能處理長距離依賴

**解讀**:
- 關注整體語義和句子結構
- 能識別轉折關係（but, however）
- 適合分析複雜句子的語義理解

**示例場景**:
```
句子: "The restaurant is expensive, but the food quality justifies the price"
句子級注意力會同時關注 "expensive" 和 "justifies"
理解整體是正面評價
```

---

### 4. 階層式注意力 (Hierarchical Attention)
**檔名**: `attention_hierarchical_sample_*.png`

**功能**: 同時顯示詞級、片語級、句子級三個層次的注意力

**特點**:
- 完整展示 AAHA 模組的工作方式
- 三個子圖分別對應三個層級
- 使用不同顏色方案區分（Blues, Greens, Oranges）

**解讀**:
- 上方：詞級注意力（藍色）- 細節關注
- 中間：片語級注意力（綠色）- 局部組合
- 下方：句子級注意力（橙色）- 全局理解

**示例**:
```
句子: "The food was delicious but service was terrible"
Aspect: "food"

詞級:    關注 "food", "delicious"
片語級:  關注 "food was delicious"
句子級:  同時考慮 "but" 之前的正面評價
```

---

## 🔧 使用方式

### 自動生成
訓練完成後，系統會自動生成 5 個驗證集樣本的注意力熱圖：

```python
# 在 train_bert.py 的 train() 方法中自動調用
trainer._generate_attention_heatmaps(num_samples=5)
```

### 手動生成
可以在訓練後單獨生成更多樣本：

```python
from experiments.train_bert import Trainer
from utils import AttentionVisualizer

# 創建視覺化器
visualizer = AttentionVisualizer(save_dir='results/visualizations')

# 載入訓練好的模型
model.load_state_dict(torch.load('results/checkpoints/best_model.pt'))
model.eval()

# 前向傳播並獲取注意力
with torch.no_grad():
    logits, attention_dict = model(
        text_ids, aspect_ids, text_mask,
        return_attention=True
    )

# 繪製注意力熱圖
aaha_attn = attention_dict['aaha']
visualizer.plot_hierarchical_attention(
    word_attention=aaha_attn['word'][0].cpu().numpy(),
    phrase_attention=aaha_attn['phrase'][0].cpu().numpy(),
    sentence_attention=aaha_attn['sentence'][0].cpu().numpy(),
    words=tokens,
    aspect='food'
)
```

### 自定義配置
修改生成樣本數量：

```python
# 在 train_bert.py 中修改
self._generate_attention_heatmaps(num_samples=10)  # 生成 10 個樣本
```

---

## 📈 注意力權重的解讀

### 權重範圍
- **0.0 - 0.1**: 幾乎不關注（白色/淺黃色）
- **0.1 - 0.3**: 輕微關注（黃色）
- **0.3 - 0.5**: 中等關注（橙色）
- **0.5 - 1.0**: 高度關注（深橙色/紅色）

### 正常模式
1. **集中型**: 權重集中在 2-3 個關鍵詞
   - 適合簡單句子
   - 情感詞明確

2. **分散型**: 權重分散在多個詞
   - 複雜句子
   - 需要綜合多個線索

3. **對稱型**: aspect 前後詞均衡關注
   - 修飾詞豐富
   - 上下文重要

### 異常模式（需要警惕）
1. **過度集中**: 只關注 1 個詞（權重 > 0.8）
   - 可能過擬合
   - 忽略上下文

2. **過度分散**: 所有詞權重相近（< 0.2）
   - 注意力機制失效
   - 模型困惑

3. **錯誤關注**: 關注無關詞彙
   - 需要檢查訓練數據
   - 可能需要調整 dropout

---

## 🎨 視覺化配置

### 顏色方案
```python
# 在 visualization.py 中配置
color_maps = {
    'word': 'Blues',      # 詞級：藍色
    'phrase': 'Greens',   # 片語級：綠色
    'sentence': 'Oranges', # 句子級：橙色
    'default': 'YlOrRd'   # 默認：黃-橙-紅
}
```

### 圖表大小
```python
# 單一注意力熱圖
figsize=(14, 3)

# 階層式注意力（3 個子圖）
figsize=(14, 8)

# 多尺度注意力（多個尺度）
figsize=(16, 3 * num_scales)
```

### DPI 設置
```python
# 高質量輸出
plt.savefig(save_path, dpi=300, bbox_inches='tight')
```

---

## 🔍 案例分析

### 案例 1: 簡單正面評價
```
句子: "The food is delicious"
Aspect: "food"
真實標籤: 正面
預測標籤: 正面

詞級注意力:
  The: 0.05
  food: 0.30
  is: 0.10
  delicious: 0.55  ← 最高權重

分析: 模型正確關注情感詞 "delicious"，預測準確
```

### 案例 2: 含否定的負面評價
```
句子: "The service is not good"
Aspect: "service"
真實標籤: 負面
預測標籤: 負面

片語級注意力:
  The: 0.05
  service: 0.25
  is: 0.10
  not: 0.35  ← 否定詞
  good: 0.25

分析: 片語級注意力正確捕捉 "not good" 組合，識別否定
```

### 案例 3: 轉折句
```
句子: "The food was delicious but the service was terrible"
Aspect: "food"
真實標籤: 正面
預測標籤: 正面

句子級注意力:
  food: 0.30
  delicious: 0.40
  but: 0.15  ← 轉折詞
  service: 0.05
  terrible: 0.05

分析: 句子級注意力理解 "but" 前後分別評價不同 aspect
```

### 案例 4: 中性評價（困難）
```
句子: "The restaurant has parking"
Aspect: "parking"
真實標籤: 中性
預測標籤: 中性

詞級注意力:
  restaurant: 0.20
  has: 0.15
  parking: 0.35

分析: 無明確情感詞，模型通過低置信度判斷為中性
```

---

## 🛠️ 新增功能

### 多尺度注意力視覺化
支援同一層級內不同維度的注意力對比：

```python
visualizer.plot_multi_scale_attention(
    attention_weights_list=[attn_64d, attn_128d, attn_256d],
    scale_names=['64維度', '128維度', '256維度'],
    words=tokens,
    aspect='food',
    true_label='正面',
    pred_label='正面',
    save_name='multi_scale_attention.png'
)
```

### 注意力對比圖
比較不同類型注意力的差異：

```python
attention_dict = {
    '詞級': word_attn,
    '片語級': phrase_attn,
    '句子級': sentence_attn
}

visualizer.plot_attention_comparison(
    attention_dict=attention_dict,
    words=tokens,
    aspect='food',
    save_name='attention_comparison.png'
)
```

---

## 📊 輸出檔案結構

訓練完成後，視覺化檔案會保存在以下位置：

```
results/visualizations/
├── attention_word_sample_1.png       # 樣本 1 詞級注意力
├── attention_word_sample_2.png       # 樣本 2 詞級注意力
├── ...
├── attention_phrase_sample_1.png     # 樣本 1 片語級注意力
├── attention_phrase_sample_2.png     # 樣本 2 片語級注意力
├── ...
├── attention_sentence_sample_1.png   # 樣本 1 句子級注意力
├── attention_sentence_sample_2.png   # 樣本 2 句子級注意力
├── ...
├── attention_hierarchical_sample_1.png  # 樣本 1 完整階層式
├── attention_hierarchical_sample_2.png  # 樣本 2 完整階層式
└── ...
```

---

## 🔬 研究用途

### 模型解釋性
- 理解模型決策過程
- 驗證注意力機制有效性
- 發現模型偏見

### 錯誤分析
- 分析預測錯誤的樣本
- 找出模型弱點
- 指導模型改進

### 論文撰寫
- 提供視覺化證據
- 展示模型優勢
- 與 baseline 對比

---

## 🎓 技術細節

### AAHAEnhanced 返回格式
```python
attention_weights = {
    'word': torch.Tensor,          # [batch, seq_len]
    'phrase': torch.Tensor,        # [batch, seq_len]
    'sentence': torch.Tensor,      # [batch, seq_len]
    'layer_weights': torch.Tensor  # [batch, 3] - 層級融合權重
}
```

### 視覺化流程
```
1. 載入最佳模型
2. 從驗證集隨機選擇樣本
3. 前向傳播獲取注意力權重
4. 將 token IDs 轉換為詞彙
5. 生成熱圖並保存
```

### 性能考量
- 每個樣本生成 4 張圖（3 個單層級 + 1 個階層式）
- 5 個樣本共 20 張圖
- 總耗時約 30-60 秒
- 不影響訓練速度（訓練後執行）

---

## 📝 最佳實踐

### 1. 選擇代表性樣本
- 正面、中性、負面各選幾個
- 包含簡單和複雜句子
- 包含成功和失敗案例

### 2. 對比不同模型
```python
# 保存不同模型的注意力圖到不同目錄
visualizer_baseline = AttentionVisualizer('results/baseline/attention')
visualizer_enhanced = AttentionVisualizer('results/enhanced/attention')
```

### 3. 記錄觀察結果
在訓練報告中添加注意力分析：
```
【注意力機制分析】
1. 詞級注意力: 主要關注情感詞和 aspect
2. 片語級注意力: 能識別否定和修飾結構
3. 句子級注意力: 理解全局語境和轉折關係
```

---

## 🚀 未來改進

1. **動態視覺化**: 製作 GIF 動畫展示注意力演化
2. **交互式視覺化**: 使用 Plotly 製作可交互圖表
3. **批量對比**: 自動生成多個樣本的對比分析
4. **注意力統計**: 計算平均注意力分布和方差

---

**注意力視覺化是理解和改進模型的重要工具！** 🎨
