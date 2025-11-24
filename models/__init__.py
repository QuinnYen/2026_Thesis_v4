"""
模型模組

包含當前架構使用的核心模組:
- BaseModel: 基礎模型類
- BERTForABSA: BERT 編碼器
- HierarchicalSyntaxAttention: HSA 模型 (Method 3)
"""

from .base_model import BaseModel, EmbeddingLayer, AttentionPooling, MLP

# BERT 支援（需要 transformers 庫）
try:
    from .bert_embedding import BERTEmbedding, HybridEmbedding, BERTForABSA
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    BERTEmbedding = None
    HybridEmbedding = None
    BERTForABSA = None

# HSA 模型
try:
    from .hierarchical_syntax import HierarchicalSyntaxAttention, create_hsa_model
    HSA_AVAILABLE = True
except ImportError:
    HSA_AVAILABLE = False
    HierarchicalSyntaxAttention = None
    create_hsa_model = None

__all__ = [
    'BaseModel',
    'EmbeddingLayer',
    'AttentionPooling',
    'MLP',
    'BERTEmbedding',
    'HybridEmbedding',
    'BERTForABSA',
    'BERT_AVAILABLE',
    'HierarchicalSyntaxAttention',
    'create_hsa_model',
    'HSA_AVAILABLE'
]
