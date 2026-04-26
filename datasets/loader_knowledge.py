"""
知識庫統一入口

一鍵切換 SenticNet 或 NRC-VAD 作為 HKGAN 的知識來源。
hkgan.py 呼叫 get_senticnet() / reset_senticnet()，
實際回傳哪個 backend 由此模組的 KNOWLEDGE_BACKEND 控制。

用法：
    # configs/unified_hkgan.yaml 加入：
    #   knowledge_backend: nrc_vad   # 或 senticnet

    # hkgan.py 不需要任何改動，仍 import：
    #   from datasets.loader_senticnet import get_senticnet, reset_senticnet
    # 但 train_multiaspect.py 在建模型前呼叫：
    #   from datasets.loader_knowledge import set_knowledge_backend
    #   set_knowledge_backend(config.get('knowledge_backend', 'nrc_vad'))
"""

from typing import Union
from datasets.loader_senticnet import (
    SenticNetKnowledge,
    get_senticnet   as _get_senticnet_sn,
    reset_senticnet as _reset_senticnet_sn,
)
from datasets.loader_nrc_vad import NRCVADKnowledge

# 目前啟用的 backend：'nrc_vad' 或 'senticnet'
KNOWLEDGE_BACKEND: str = 'nrc_vad'

# 單例
_instance: Union[SenticNetKnowledge, NRCVADKnowledge, None] = None


def set_knowledge_backend(backend: str):
    """
    切換知識庫 backend。需在 reset_senticnet() 後、get_senticnet() 前呼叫。

    Args:
        backend: 'nrc_vad'（預設）或 'senticnet'
    """
    global KNOWLEDGE_BACKEND
    assert backend in ('nrc_vad', 'senticnet'), f"未知 backend: {backend!r}"
    KNOWLEDGE_BACKEND = backend


def get_senticnet(path: str = None) -> Union[SenticNetKnowledge, NRCVADKnowledge]:
    """
    獲取知識庫單例。依 KNOWLEDGE_BACKEND 回傳對應實例。
    介面與 SenticNetKnowledge 完全相容，hkgan.py 不需要修改。
    """
    global _instance
    if _instance is None:
        if KNOWLEDGE_BACKEND == 'nrc_vad':
            _instance = NRCVADKnowledge(path)
        else:
            _instance = SenticNetKnowledge(path)
    return _instance


def reset_senticnet():
    """重置單例（切換 backend 前必須先呼叫）"""
    global _instance
    _instance = None
