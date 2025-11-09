"""
HMAC-Net 模型模組
"""

from .base_model import BaseModel, EmbeddingLayer, AttentionPooling, MLP
from .aaha_enhanced import AAHAEnhanced
from .pmac_enhanced import PMACEnhanced
from .iarm_enhanced import IARMEnhanced
from .hmac_net import HMACNet, HMACNetMultiAspect

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
    'PMACEnhanced',
    'IARMEnhanced',
    'HMACNet',
    'HMACNetMultiAspect',
    'BERTEmbedding',
    'HybridEmbedding',
    'BERTForABSA',
    'BERT_AVAILABLE'
]
