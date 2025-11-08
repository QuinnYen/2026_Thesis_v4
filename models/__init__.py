"""
HMAC-Net 模型模組
"""

from .base_model import BaseModel, EmbeddingLayer, AttentionPooling, MLP
from .aaha import AAHA
from .pmac import PMAC
from .iarm import IARM
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
    'AAHA',
    'PMAC',
    'IARM',
    'HMACNet',
    'HMACNetMultiAspect',
    'BERTEmbedding',
    'HybridEmbedding',
    'BERTForABSA',
    'BERT_AVAILABLE'
]
