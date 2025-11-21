"""
模型模組

包含當前架構使用的核心模組:
- BaseModel: 基礎模型類
- BERTForABSA: BERT 編碼器
- AAHAEnhanced: Aspect-Aware Hierarchical Attention (保留供參考)
"""

from .base_model import BaseModel, EmbeddingLayer, AttentionPooling, MLP
from .aaha_enhanced import AAHAEnhanced

# BERT 支援（需要 transformers 庫）
try:
    from .bert_embedding import BERTEmbedding, HybridEmbedding, BERTForABSA
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    BERTEmbedding = None
    HybridEmbedding = None
    BERTForABSA = None

__all__ = [
    'BaseModel',
    'EmbeddingLayer',
    'AttentionPooling',
    'MLP',
    'AAHAEnhanced',
    'BERTEmbedding',
    'HybridEmbedding',
    'BERTForABSA',
    'BERT_AVAILABLE'
]
