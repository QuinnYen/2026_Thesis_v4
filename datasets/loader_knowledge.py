"""
知識庫統一入口

hkgan.py 呼叫 get_senticnet() / reset_senticnet()，
實際回傳 SenticNetKnowledge 單例。
"""

from datasets.loader_senticnet import (
    SenticNetKnowledge,
    get_senticnet,
    reset_senticnet,
)

__all__ = ['get_senticnet', 'reset_senticnet', 'SenticNetKnowledge']
