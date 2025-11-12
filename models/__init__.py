"""
HMAC-Net 模型模組
只保留當前架構使用的增強版模組
"""

from .base_model import BaseModel, EmbeddingLayer, AttentionPooling, MLP
from .aaha_enhanced import AAHAEnhanced
from .pmac_selective import SelectivePMACMultiAspect, SelectivePMAC
from .iarm_enhanced import IARMMultiAspect, IARMEnhanced

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
    'SelectivePMACMultiAspect',
    'SelectivePMAC',
    'IARMEnhanced',
    'IARMMultiAspect',
    'BERTEmbedding',
    'HybridEmbedding',
    'BERTForABSA',
    'BERT_AVAILABLE'
]
