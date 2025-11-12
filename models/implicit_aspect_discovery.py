"""
Implicit Aspect Discovery Module (隱含面向發現模組)

核心創新：
- 自動從句子級別文本中發現隱含的語義面向
- 結合預定義的語義 aspects 和數據驅動的發現
- 與 PMAC 和 IARM 無縫整合

應用場景：
- IMDB 等句子級別情感分析
- 任何沒有明確 aspect 標註的文本分類任務
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicitAspectDiscovery(nn.Module):
    """
    隱含面向發現模組

    從句子的 BERT embeddings 中自動發現 K 個隱含的語義面向

    方法：
    1. Learnable Aspect Queries（可學習的面向查詢）
    2. Cross-Attention 機制將句子映射到不同面向
    3. 每個面向得到一個表示向量
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_aspects: int = 5,  # 發現的隱含 aspects 數量
        num_heads: int = 8,
        dropout: float = 0.1,
        use_predefined_aspects: bool = False,  # 是否使用預定義的 aspect 名稱
        predefined_aspect_names: list = None
    ):
        super(ImplicitAspectDiscovery, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_aspects = num_aspects
        self.num_heads = num_heads

        # 可學習的 Aspect Queries（類似 DETR 的 object queries）
        # 這些向量會學習去"捕捉"句子中不同的語義面向
        self.aspect_queries = nn.Parameter(
            torch.randn(num_aspects, hidden_dim)
        )

        # Cross-Attention: Aspect Queries attend to sentence tokens
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward 網絡（用於精煉每個 aspect 的表示）
        self.aspect_refine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )

        # Layer Normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # 如果使用預定義的 aspect 名稱（可選，用於可解釋性）
        self.use_predefined_aspects = use_predefined_aspects
        self.aspect_names = predefined_aspect_names or [
            f"implicit_aspect_{i}" for i in range(num_aspects)
        ]

        # Aspect 重要性權重（學習哪些 aspects 更重要）
        self.aspect_importance = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        text_embeddings: torch.Tensor,
        attention_mask: torch.Tensor = None
    ):
        """
        從文本中發現隱含的 aspects

        參數:
            text_embeddings: [batch, seq_len, hidden_dim] - BERT 輸出
            attention_mask: [batch, seq_len] - 注意力遮罩

        返回:
            aspect_representations: [batch, num_aspects, hidden_dim]
            aspect_weights: [batch, num_aspects] - 每個 aspect 的重要性
            attention_maps: [batch, num_aspects, seq_len] - 可視化用
        """
        batch_size = text_embeddings.size(0)

        # 1. 擴展 aspect queries 到 batch
        queries = self.aspect_queries.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, num_aspects, hidden_dim]

        # 2. Cross-Attention: queries attend to text
        # 準備 key_padding_mask（True 表示需要被 mask 的位置）
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)  # [batch, seq_len]
        else:
            key_padding_mask = None

        attn_output, attn_weights = self.cross_attention(
            query=queries,
            key=text_embeddings,
            value=text_embeddings,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False  # 保留所有 head 的權重
        )  # attn_output: [batch, num_aspects, hidden_dim]
           # attn_weights: [batch, num_heads, num_aspects, seq_len]

        # 3. 殘差連接 + Layer Norm
        aspect_repr = self.layer_norm1(queries + attn_output)

        # 4. Feed-forward 精煉
        refined = self.aspect_refine(aspect_repr)
        aspect_repr = self.layer_norm2(aspect_repr + refined)
        # [batch, num_aspects, hidden_dim]

        # 5. 計算每個 aspect 的重要性權重
        importance_weights = self.aspect_importance(aspect_repr).squeeze(-1)
        # [batch, num_aspects]

        # 6. 平均所有 attention heads 的權重（用於可視化）
        avg_attn_weights = attn_weights.mean(dim=1)
        # [batch, num_aspects, seq_len]

        return aspect_repr, importance_weights, avg_attn_weights

    def get_aspect_names(self):
        """返回 aspect 名稱（用於可解釋性）"""
        return self.aspect_names

    def visualize_aspects(
        self,
        text_tokens: list,
        attention_weights: torch.Tensor,
        top_k: int = 5
    ):
        """
        可視化每個 aspect 關注的詞彙

        參數:
            text_tokens: 文本的 token 列表
            attention_weights: [num_aspects, seq_len]
            top_k: 顯示前 k 個最相關的詞

        返回:
            aspect_highlights: dict
        """
        num_aspects = attention_weights.size(0)
        aspect_highlights = {}

        for i in range(num_aspects):
            aspect_name = self.aspect_names[i]
            weights = attention_weights[i].cpu().numpy()

            # 找出權重最高的 top_k 個詞
            top_indices = weights.argsort()[-top_k:][::-1]
            top_words = [(text_tokens[idx], weights[idx]) for idx in top_indices]

            aspect_highlights[aspect_name] = top_words

        return aspect_highlights


class SentenceLevelAspectAdapter(nn.Module):
    """
    句子級別 Aspect 適配器

    將隱含 aspects 的表示轉換為統一格式，
    以便與原有的 PMAC 和 IARM 模組無縫整合
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_implicit_aspects: int = 5,
        num_predefined_aspects: int = 3,
        fusion_mode: str = 'concat'  # 'concat', 'add', 'hybrid'
    ):
        super(SentenceLevelAspectAdapter, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_implicit = num_implicit_aspects
        self.num_predefined = num_predefined_aspects
        self.fusion_mode = fusion_mode

        total_aspects = num_implicit_aspects + num_predefined_aspects

        # 融合層（如果需要）
        if fusion_mode == 'concat':
            self.fusion = nn.Linear(hidden_dim, hidden_dim)
        elif fusion_mode == 'hybrid':
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )

    def forward(
        self,
        implicit_aspects: torch.Tensor,
        predefined_aspects: torch.Tensor = None
    ):
        """
        融合隱含和預定義的 aspects

        參數:
            implicit_aspects: [batch, num_implicit, hidden_dim]
            predefined_aspects: [batch, num_predefined, hidden_dim] (可選)

        返回:
            unified_aspects: [batch, total_aspects, hidden_dim]
        """
        if predefined_aspects is None:
            return implicit_aspects

        if self.fusion_mode == 'concat':
            # 簡單拼接
            unified = torch.cat([implicit_aspects, predefined_aspects], dim=1)
        elif self.fusion_mode == 'add':
            # 加權相加（需要數量相同）
            unified = implicit_aspects + predefined_aspects
        elif self.fusion_mode == 'hybrid':
            # 混合融合
            batch_size = implicit_aspects.size(0)
            fused = []
            for i in range(min(self.num_implicit, self.num_predefined)):
                concat_feat = torch.cat([
                    implicit_aspects[:, i],
                    predefined_aspects[:, i]
                ], dim=-1)
                fused_feat = self.fusion(concat_feat)
                fused.append(fused_feat)
            unified = torch.stack(fused, dim=1)

        return unified


# 預定義的語義 Aspects（針對不同領域）
PREDEFINED_ASPECTS = {
    'movie_review': [
        'overall impression',
        'story and plot',
        'acting performance',
        'technical quality',
        'emotional impact'
    ],
    'product_review': [
        'overall satisfaction',
        'quality',
        'value for money',
        'features',
        'usability'
    ],
    'restaurant_review': [
        'overall experience',
        'food quality',
        'service',
        'ambiance',
        'value'
    ],
    'generic': [
        'overall',
        'positive aspects',
        'negative aspects',
        'neutral observations',
        'emotional tone'
    ]
}


if __name__ == '__main__':
    # 測試代碼
    print("測試 Implicit Aspect Discovery Module...")

    batch_size = 4
    seq_len = 50
    hidden_dim = 768
    num_aspects = 5

    # 模擬 BERT 輸出
    text_embeddings = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)

    # 創建模組
    iadm = ImplicitAspectDiscovery(
        hidden_dim=hidden_dim,
        num_aspects=num_aspects,
        predefined_aspect_names=PREDEFINED_ASPECTS['movie_review']
    )

    # Forward
    aspect_repr, importance, attn_weights = iadm(text_embeddings, attention_mask)

    print(f"輸入形狀: {text_embeddings.shape}")
    print(f"Aspect 表示: {aspect_repr.shape}")
    print(f"Aspect 重要性: {importance.shape}")
    print(f"注意力權重: {attn_weights.shape}")

    print(f"\nAspect 名稱:")
    for i, name in enumerate(iadm.get_aspect_names()):
        print(f"  {i}: {name}")

    print("\n✓ Implicit Aspect Discovery 測試通過！")
