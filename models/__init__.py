"""
模型模組

包含當前架構使用的核心模組:
- BaseModel: 基礎模型類
- BERTForABSA: BERT 編碼器
"""

from .base_model import BaseModel

# BERT 支援（需要 transformers 庫）
try:
    from .bert_embedding import BERTEmbedding, BERTForABSA
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    BERTEmbedding = None
    BERTForABSA = None

__all__ = [
    'BaseModel',
    'BERTEmbedding',
    'BERTForABSA',
    'BERT_AVAILABLE',
]
