"""
NRC VAD Lexicon v2.1 知識庫 Loader

使用 Valence 分數作為情感極性值，與 SenticNetKnowledge 介面完全相容，
可直接替換 SenticNet 作為 HKGAN 的知識來源。

Reference:
    Mohammad, S.M. (2025). "NRC VAD Lexicon v2: Norms for Valence, Arousal,
    and Dominance for over 55k English Terms." arXiv:2503.23547.
    https://saifmohammad.com/WebPages/nrc-vad.html
"""

import re
import torch
from pathlib import Path
from typing import Dict, List, Optional, Set

# 共用領域過濾表（與 loader_senticnet.py 同一份定義）
from datasets.loader_senticnet import DOMAIN_TECHNICAL_TERMS


class NRCVADKnowledge:
    """
    NRC VAD Lexicon v2.1 知識庫

    - 載入 Unigrams 版（44,728 個單字），valence [-1, 1] 直接作為極性值
    - 與 SenticNetKnowledge 介面完全相容：
        get_polarity() / get_polarity_with_coverage() / get_coverage_mask() / batch_lookup()
    - 支援 Domain Filter（共用 loader_senticnet.py 的 DOMAIN_TECHNICAL_TERMS）
    - 未知詞 fallback 為 mean_valence（≈ 0，不引入偏差）
    """

    DEFAULT_PATH = (
        "data/NRC-VAD-Lexicon-v2.1/NRC-VAD-Lexicon-v2.1"
        "/Unigrams/unigrams-NRC-VAD-Lexicon-v2.1.txt"
    )

    def __init__(self, path: str = None, domain: str = None):
        self.lexicon: Dict[str, float] = {}
        self.polarity_cache: Dict[str, float] = {}

        self.domain: Optional[str] = None
        self.domain_filter_terms: Set[str] = set()
        self.domain_filter_enabled: bool = False

        resolved = (
            Path(__file__).parent.parent / self.DEFAULT_PATH
            if path is None
            else Path(path)
        )
        self._load(resolved)

        self._neutral_mean_polarity: float = (
            sum(self.lexicon.values()) / len(self.lexicon)
            if self.lexicon else 0.0
        )
        print(
            f"[NRC-VAD] Loaded {len(self.lexicon)} unigrams, "
            f"mean_valence={self._neutral_mean_polarity:+.4f}"
        )

        if domain:
            self.set_domain(domain)

    # -------------------------------------------------------------------------
    # 載入
    # -------------------------------------------------------------------------

    def _load(self, path: Path):
        if not path.exists():
            print(f"[Warning] NRC-VAD file not found: {path}")
            print("[Warning] Knowledge enhancement will be disabled")
            return
        with open(path, encoding='utf-8') as f:
            next(f)  # 跳過 header 行
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    word = parts[0].strip().lower()
                    try:
                        self.lexicon[word] = float(parts[1])  # valence 已是 [-1, 1]
                    except ValueError:
                        pass

    # -------------------------------------------------------------------------
    # 詞彙清理（與 SenticNetKnowledge._clean_word 邏輯一致）
    # -------------------------------------------------------------------------

    def _clean_word(self, word: str) -> str:
        if not word:
            return ""
        word = word.replace('##', '')           # 去 BERT subword 前綴
        if word.startswith('[') and word.endswith(']'):
            return ""                            # 跳過特殊 token
        word = word.lower()
        word = re.sub(r'[^a-z0-9\s]', '', word)
        return word.strip()

    # -------------------------------------------------------------------------
    # Domain Filter
    # -------------------------------------------------------------------------

    def set_domain(self, domain: str, enable_filter: bool = True):
        domain_lower = domain.lower()
        self.domain = domain_lower
        if domain_lower in DOMAIN_TECHNICAL_TERMS:
            self.domain_filter_terms = DOMAIN_TECHNICAL_TERMS[domain_lower]
            self.domain_filter_enabled = enable_filter
            print(
                f"[NRC-VAD] Domain filter enabled: '{domain}' "
                f"({len(self.domain_filter_terms)} terms)"
            )
        else:
            self.domain_filter_terms = set()
            self.domain_filter_enabled = False
            print(f"[NRC-VAD] Unknown domain '{domain}', filter disabled")
        self.polarity_cache.clear()

    def disable_domain_filter(self):
        self.domain_filter_enabled = False
        self.polarity_cache.clear()

    def enable_domain_filter(self):
        if self.domain_filter_terms:
            self.domain_filter_enabled = True
            self.polarity_cache.clear()

    # -------------------------------------------------------------------------
    # 極性查詢（與 SenticNetKnowledge 相容介面）
    # -------------------------------------------------------------------------

    def get_polarity(self, word: str) -> float:
        if word in self.polarity_cache:
            return self.polarity_cache[word]
        clean = self._clean_word(word)
        if not clean:
            return 0.0
        if self.domain_filter_enabled and clean in self.domain_filter_terms:
            self.polarity_cache[word] = 0.0
            return 0.0
        pol = self.lexicon.get(clean, self._neutral_mean_polarity)
        self.polarity_cache[word] = pol
        return pol

    def get_polarity_with_coverage(self, word: str) -> tuple:
        clean = self._clean_word(word)
        if not clean:
            return 0.0, False
        if self.domain_filter_enabled and clean in self.domain_filter_terms:
            return 0.0, True
        if clean in self.lexicon:
            return self.lexicon[clean], True
        return self._neutral_mean_polarity, False

    def get_coverage_mask(self, words: List[str]) -> torch.Tensor:
        mask = [1.0 if self._clean_word(w) in self.lexicon else 0.0 for w in words]
        return torch.tensor(mask, dtype=torch.float32)

    def batch_lookup(
        self, tokens: List[str], device: torch.device = None
    ) -> torch.Tensor:
        tensor = torch.tensor(
            [self.get_polarity(t) for t in tokens], dtype=torch.float32
        )
        return tensor.to(device) if device is not None else tensor

    # -------------------------------------------------------------------------
    # 魔術方法
    # -------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.lexicon)

    def __contains__(self, word: str) -> bool:
        return self._clean_word(word) in self.lexicon


# =============================================================================
# 快速測試
# =============================================================================

if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    nrc = NRCVADKnowledge()
    print(f"\n=== NRC VAD 統計 ===")
    print(f"  詞彙數：{len(nrc)}")
    print(f"  mean_valence：{nrc._neutral_mean_polarity:+.4f}")

    print("\n=== 情感詞正確性 ===")
    for w in ['good', 'bad', 'excellent', 'terrible', 'delicious', 'awful']:
        pol, known = nrc.get_polarity_with_coverage(w)
        print(f"  {w:12s}  pol={pol:+.3f}  known={known}")

    print("\n=== LAP 技術詞覆蓋率 ===")
    lap = ['trackpad','ssd','processor','keyboard','display','battery',
           'cpu','ram','charger','webcam','touchpad','backlit','screen',
           'drive','port','speaker','camera','fan','boot','install']
    hits = sum(1 for w in lap if w in nrc)
    for w in lap:
        pol, known = nrc.get_polarity_with_coverage(w)
        print(f"  {w:12s}  known={known}  pol={pol:+.3f}")
    print(f"  命中率: {hits}/{len(lap)} = {hits/len(lap)*100:.0f}%")
