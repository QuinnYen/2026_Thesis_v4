# HMAC-Net: 面向級情感分析實驗框架

HMAC-Net (Hierarchical Multi-Aspect Composition Network) 是一個用於面向級情感分析的深度學習模型，整合了三個核心創新模組。

## 📋 目錄

- [專案結構](#專案結構)
- [核心模組](#核心模組)
- [安裝](#安裝)
- [快速開始](#快速開始)
- [配置說明](#配置說明)
- [數據準備](#數據準備)
- [訓練模型](#訓練模型)
- [實驗功能](#實驗功能)

## 專案結構

```
HMAC-Net/
├── data/                      # 數據目錄
│   ├── raw/                  # 原始數據（SemEval-2014 等）
│   ├── processed/            # 預處理後數據
│   └── embeddings/           # 詞嵌入（GloVe）
│
├── models/                    # 模型定義
│   ├── base_model.py         # 基礎模型類
│   ├── aaha.py               # AAHA 模組（階層式注意力）
│   ├── pmac.py               # PMAC 模組（多面向組合）
│   ├── iarm.py               # IARM 模組（面向間關係）
│   ├── hmac_net.py           # 完整 HMAC-Net
│   └── baselines.py          # Baseline 模型
│
├── utils/                     # 工具模組
│   ├── logger.py             # 日誌記錄
│   ├── metrics.py            # 評估指標
│   ├── preprocessor.py       # 數據預處理
│   ├── data_loader.py        # 數據載入器
│   └── visualization.py      # 視覺化
│
├── experiments/               # 實驗腳本
│   ├── train.py              # 訓練腳本
│   ├── evaluate.py           # 評估腳本
│   ├── ablation_study.py     # 消融實驗
│   └── compare_baselines.py  # Baseline 比較
│
├── configs/                   # 配置檔案
│   ├── model_config.yaml     # 模型超參數
│   ├── experiment_config.yaml # 實驗配置
│   └── data_config.yaml      # 數據配置
│
├── results/                   # 實驗結果
│   ├── checkpoints/          # 模型檢查點
│   ├── logs/                 # 訓練日誌
│   ├── visualizations/       # 視覺化圖表
│   └── reports/              # 實驗報告
│
└── requirements.txt           # 依賴套件
```

## 核心模組

### 1. AAHA (Aspect-Aware Hierarchical Attention)
**面向感知階層式注意力**

- **詞級注意力**：關注單個詞與面向的關聯
- **片語級注意力**：使用 CNN 提取局部片語特徵
- **句子級注意力**：使用雙向 LSTM 捕捉全局資訊
- **動態層級融合**：自動學習三層注意力的最佳組合

### 2. PMAC (Progressive Multi-Aspect Composition)
**漸進式多面向組合**

- **多粒度表示**：從不同粒度提取面向特徵
- **門控融合機制**：動態控制特徵融合比例
- **漸進式組合**：逐步組合多個面向資訊

### 3. IARM (Inter-Aspect Relation Modeling)
**面向間關係建模**

- **圖注意力網路**：建模面向之間的依賴關係
- **Transformer 式交互**：使用自注意力機制
- **關係增強表示**：生成關係感知的面向表示

## 安裝

### 環境要求
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.0+ (GPU 加速，可選)

### 安裝步驟

```bash
# 克隆專案
cd HMAC-Net

# 安裝依賴
pip install -r requirements.txt

# （可選）安裝 spaCy 語言模型
python -m spacy download en_core_web_sm
```

## 快速開始

### 1. 準備數據

將 SemEval-2014 數據放入 `data/raw/semeval2014/` 目錄：

```
data/raw/semeval2014/
├── restaurant_train.xml
├── restaurant_test.xml
├── laptop_train.xml
└── laptop_test.xml
```

### 2. 下載詞嵌入

下載 GloVe 詞嵌入並放入 `data/embeddings/`：

```bash
# 下載 GloVe 840B 300d
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
unzip glove.840B.300d.zip -d data/embeddings/
```

### 3. 訓練模型

```bash
# 使用默認配置訓練
python experiments/train.py

# 使用自定義配置
python experiments/train.py --config configs/experiment_config.yaml
```

### 4. 評估模型

```bash
# 評估最佳模型
python experiments/evaluate.py --checkpoint results/checkpoints/hmac_net_best.pt
```

## 配置說明

### 模型配置 (`configs/model_config.yaml`)

```yaml
model:
  embedding_dim: 300        # 詞嵌入維度
  hidden_dim: 256           # 隱藏層維度
  num_layers: 2             # LSTM 層數
  dropout: 0.5              # Dropout 比率

aaha:
  word_attention_dim: 128   # 詞級注意力維度
  phrase_attention_dim: 128 # 片語級注意力維度
  sentence_attention_dim: 128 # 句子級注意力維度

pmac:
  fusion_method: "gated"    # 融合方法
  composition_layers: 2     # 組合層數

iarm:
  relation_type: "transformer"  # 關係建模類型
  num_heads: 4              # 注意力頭數
```

### 訓練配置 (`configs/experiment_config.yaml`)

```yaml
training:
  batch_size: 32
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001

early_stopping:
  enabled: true
  patience: 10
  metric: "macro_f1"
```

## 數據準備

### 使用 SemEval-2014

```python
from utils import SemEvalPreprocessor, load_semeval_2014

# 載入數據
preprocessor = SemEvalPreprocessor()
train_df, test_df = load_semeval_2014(
    data_dir='data/raw/semeval2014',
    domain='restaurant',
    preprocessor=preprocessor
)

# 保存詞彙表
preprocessor.save_vocabulary('data/processed/vocab.pkl')
```

### 自定義數據格式

數據應包含以下欄位：
- `text`: 句子文本
- `aspect`: 面向詞
- `polarity`: 情感極性 (positive/negative/neutral)

## 訓練模型

### 基本訓練

```python
from models import HMACNet
import torch

# 創建模型
model = HMACNet(
    vocab_size=5000,
    embedding_dim=300,
    hidden_dim=256,
    num_classes=3
)

# 訓練（參見 experiments/train.py）
```

### 使用預訓練嵌入

```python
from utils import load_glove_embeddings

# 載入 GloVe
embeddings = load_glove_embeddings(
    glove_path='data/embeddings/glove.840B.300d.txt',
    word2idx=preprocessor.word2idx,
    embedding_dim=300
)

# 創建模型時傳入
model = HMACNet(
    vocab_size=5000,
    pretrained_embeddings=torch.from_numpy(embeddings)
)
```

## 實驗功能

### 1. 消融實驗

測試各模組的貢獻：

```bash
python experiments/ablation_study.py
```

會測試以下變體：
- **完整模型**：AAHA + PMAC + IARM
- **w/o AAHA**：移除階層式注意力
- **w/o PMAC**：移除多面向組合
- **w/o IARM**：移除面向間關係

### 2. Baseline 比較

與其他模型比較：

```bash
python experiments/compare_baselines.py
```

包含的 Baseline：
- LSTM
- ATAE-LSTM
- IAN
- HMAC-Net（提出方法）

### 3. 注意力視覺化

```python
from utils import AttentionVisualizer

# 創建視覺化器
visualizer = AttentionVisualizer()

# 繪製階層式注意力
visualizer.plot_hierarchical_attention(
    word_attention=word_attn,
    phrase_attention=phrase_attn,
    sentence_attention=sentence_attn,
    words=tokens,
    aspect='food'
)
```

## 實驗結果

訓練完成後，結果會保存在 `results/` 目錄：

- **檢查點**：`results/checkpoints/hmac_net_best_f1_*.pt`
- **訓練曲線**：`results/visualizations/hmac_net_training_curves.png`
- **混淆矩陣**：`results/visualizations/confusion_matrix.png`
- **注意力視覺化**：`results/visualizations/attention_*.png`
- **日誌**：`results/logs/HMAC-Net_*.log`

## 進階使用

### 自定義模組

可以輕鬆替換或修改模組：

```python
from models import HMACNet, AAHA, PMAC, IARM

# 自定義 AAHA
class CustomAAHA(AAHA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 添加自定義層

    def forward(self, *args, **kwargs):
        # 自定義前向傳播
        pass

# 在 HMAC-Net 中使用
# 修改 models/hmac_net.py 中的 self.aaha
```

### 多 GPU 訓練

```python
# 使用 DataParallel
model = nn.DataParallel(model)

# 或使用 DistributedDataParallel（推薦）
# 參見 PyTorch 文檔
```

## 常見問題

### Q: 如何調整超參數？
A: 編輯 `configs/model_config.yaml` 和 `configs/experiment_config.yaml`

### Q: 訓練很慢怎麼辦？
A:
1. 使用 GPU（設置 `use_cuda: true`）
2. 增加 batch size
3. 減少 LSTM 層數或隱藏層維度

### Q: 如何處理 OOM（記憶體不足）？
A:
1. 減少 batch size
2. 減少序列最大長度
3. 使用梯度累積

## 引用

如果您使用了本程式碼，請引用：

```bibtex
@article{hmacnet2024,
  title={HMAC-Net: Hierarchical Multi-Aspect Composition Network for Aspect-Level Sentiment Analysis},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## 授權

本專案採用 MIT 授權。

## 聯繫方式

如有問題或建議，請開 Issue 或聯繫作者。

---

**祝實驗順利！** 🚀
