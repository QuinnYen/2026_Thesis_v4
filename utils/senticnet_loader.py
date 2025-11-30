"""
SenticNet 5.0 Knowledge Base Loader

提供情感知識增強功能：
- 從 SenticNet 5.0 加載詞彙情感極性
- 支援 ABSA 任務的知識注入
- 高效的批量查詢和緩存機制
- 領域特定遮罩（Domain-specific Filtering）解決技術術語誤判問題

SenticNet 5.0 數據格式:
    senticnet[concept] = [introspection, temper, attitude, sensitivity,
                         primary_emotion, secondary_emotion, polarity_label,
                         polarity_value, semantics1-5...]

核心改進（解決 Neutral 識別問題）：
1. Domain Filtering: 針對特定領域（如 Laptops）的技術術語，
   遮蔽其通用情感極性，避免將客觀描述誤判為情感表達
2. 未登錄詞標記：區分「中性詞（polarity=0）」與「未知詞」

Reference:
    Cambria et al. "SenticNet 5: Discovering Conceptual Primitives for
    Sentiment Analysis by Means of Context Embeddings" (AAAI 2018)
"""

import os
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union, Set
import re


# =============================================================================
# 領域特定技術術語表（Domain-specific Technical Terms）
# 這些詞在特定領域下是客觀描述，不應帶有情感極性
# =============================================================================

# Laptops 領域：電腦硬體規格、技術參數
LAPTOPS_TECHNICAL_TERMS = {
    # 尺寸與規格相關
    'high', 'low', 'small', 'large', 'big', 'thin', 'thick', 'light', 'heavy',
    'fast', 'slow', 'quick', 'long', 'short',
    # 硬體組件
    'screen', 'display', 'keyboard', 'trackpad', 'touchpad', 'battery',
    'processor', 'cpu', 'gpu', 'ram', 'memory', 'storage', 'ssd', 'hdd',
    'drive', 'port', 'usb', 'hdmi', 'speaker', 'camera', 'webcam', 'mic',
    'fan', 'cooling', 'charger', 'adapter', 'cable',
    # 規格單位
    'gb', 'tb', 'mb', 'ghz', 'mhz', 'inch', 'inches', 'pixel', 'pixels',
    'resolution', 'hz', 'watt', 'watts', 'mah',
    # 技術特性
    'wireless', 'bluetooth', 'wifi', 'ethernet', 'optical',
    'backlit', 'illuminated', 'matte', 'glossy', 'retina', 'ips', 'oled', 'lcd',
    # 外觀描述（在技術語境下可能是中性的）
    'compact', 'portable', 'lightweight', 'slim', 'sleek',
    # 系統相關
    'boot', 'startup', 'shutdown', 'install', 'update', 'upgrade',
    'windows', 'mac', 'linux', 'os', 'system', 'software', 'hardware',
    # 形容詞在技術語境下的中性用法
    'standard', 'basic', 'normal', 'regular', 'default', 'typical',
}

# Restaurants 領域：餐廳相關的中性描述詞
RESTAURANTS_TECHNICAL_TERMS = {
    # 份量描述
    'large', 'small', 'big', 'huge', 'tiny', 'medium',
    # 位置與環境（可能是中性描述）
    'crowded', 'busy', 'quiet', 'loud', 'dark', 'bright',
    # 等待相關
    'long', 'short', 'quick', 'slow', 'fast',
    # 價格相關（可能是客觀陳述）
    'cheap', 'expensive', 'pricey', 'costly', 'affordable',
    # 食物類型（不帶情感）
    'hot', 'cold', 'warm', 'fresh', 'raw', 'cooked', 'fried', 'grilled',
    'spicy', 'mild', 'sweet', 'sour', 'salty', 'bitter',
}

# 領域技術術語映射
DOMAIN_TECHNICAL_TERMS = {
    'laptops': LAPTOPS_TECHNICAL_TERMS,
    'laptop': LAPTOPS_TECHNICAL_TERMS,
    'restaurants': RESTAURANTS_TECHNICAL_TERMS,
    'restaurant': RESTAURANTS_TECHNICAL_TERMS,
    'rest14': RESTAURANTS_TECHNICAL_TERMS,
    'rest15': RESTAURANTS_TECHNICAL_TERMS,
    'rest16': RESTAURANTS_TECHNICAL_TERMS,
    'lap14': LAPTOPS_TECHNICAL_TERMS,
    'lap15': LAPTOPS_TECHNICAL_TERMS,
    'lap16': LAPTOPS_TECHNICAL_TERMS,
    'mams': RESTAURANTS_TECHNICAL_TERMS,  # MAMS 也是餐廳領域
}


class SenticNetKnowledge:
    """
    SenticNet 5.0 情感知識庫加載器

    功能：
    - 加載 SenticNet 5.0 詞彙表
    - 提供詞彙 → 情感極性映射 [-1, 1]
    - 支持批量查詢和緩存
    - 處理 BERT subword tokens
    - 領域特定過濾（Domain Filtering）
    - 區分「中性詞」與「未登錄詞」

    使用示例:
        senticnet = SenticNetKnowledge()
        polarity = senticnet.get_polarity("good")  # 返回正值
        polarity = senticnet.get_polarity("bad")   # 返回負值

        # 啟用領域過濾
        senticnet.set_domain("laptops")
        polarity = senticnet.get_polarity("high")  # 在 laptops 領域返回 0（中性）

        # 區分中性與未知
        polarity, is_known = senticnet.get_polarity_with_coverage("unknown_word")
    """

    # 默認 SenticNet 路徑
    DEFAULT_PATH = "data/SenticNet_5.0/senticnet.py"

    # 用於標記未登錄詞的特殊值（區分真正的中性 0.0）
    UNKNOWN_MARKER = float('nan')

    def __init__(self, senticnet_path: str = None, domain: str = None):
        """
        初始化 SenticNet 知識庫

        Args:
            senticnet_path: SenticNet Python 文件路徑
                           如果為 None，使用默認路徑
            domain: 領域名稱（如 'laptops', 'restaurants'）
                   用於啟用領域特定過濾
        """
        self.senticnet: Dict[str, List] = {}
        self.polarity_cache: Dict[str, float] = {}

        # 領域過濾設置
        self.domain: Optional[str] = None
        self.domain_filter_terms: Set[str] = set()
        self.domain_filter_enabled: bool = False

        # 確定路徑
        if senticnet_path is None:
            # 從項目根目錄查找
            project_root = Path(__file__).parent.parent
            senticnet_path = project_root / self.DEFAULT_PATH
        else:
            senticnet_path = Path(senticnet_path)

        # 加載 SenticNet
        self._load_senticnet(senticnet_path)

        # 設置領域過濾
        if domain:
            self.set_domain(domain)

        print(f"[SenticNet] Loaded {len(self.senticnet)} concepts")

    def _load_senticnet(self, path: Path):
        """
        加載 SenticNet Python 字典文件

        Args:
            path: senticnet.py 文件路徑
        """
        if not path.exists():
            print(f"[Warning] SenticNet file not found: {path}")
            print("[Warning] Knowledge enhancement will be disabled")
            return

        # 讀取文件內容並逐行解析（避免 exec 的語法問題）
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 解析每一行
        import ast
        self.senticnet = {}
        error_count = 0

        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or line == 'senticnet={}':
                continue

            # 格式: senticnet['concept'] = [...]
            if line.startswith("senticnet['") and '] = [' in line:
                try:
                    # 提取 concept 名稱
                    start = line.find("['") + 2
                    end = line.find("']")
                    concept = line[start:end]

                    # 提取值列表
                    value_start = line.find('] = [') + 5
                    value_str = line[value_start:-1] if line.endswith(']') else line[value_start:]

                    # 安全解析列表
                    try:
                        values = ast.literal_eval('[' + value_str + ']')
                        self.senticnet[concept] = values
                    except:
                        # 如果 ast 解析失敗，手動提取 polarity_value
                        # polarity_value 通常是第 8 個逗號分隔的值
                        parts = value_str.split(', ')
                        if len(parts) >= 8:
                            try:
                                polarity_value = float(parts[7])
                                polarity_label = parts[6].strip("'")
                                # 創建最小化的條目
                                self.senticnet[concept] = [
                                    0, 0, 0, 0, None, None,
                                    polarity_label, polarity_value
                                ]
                            except:
                                error_count += 1
                        else:
                            error_count += 1
                except Exception as e:
                    error_count += 1

        if error_count > 0:
            print(f"[SenticNet] Skipped {error_count} malformed entries")

    # =========================================================================
    # 領域特定過濾（Domain-specific Filtering）
    # =========================================================================

    def set_domain(self, domain: str, enable_filter: bool = True):
        """
        設置當前領域並啟用領域過濾

        Args:
            domain: 領域名稱（'laptops', 'restaurants', 'mams' 等）
            enable_filter: 是否啟用過濾（默認 True）
        """
        domain_lower = domain.lower()
        self.domain = domain_lower

        if domain_lower in DOMAIN_TECHNICAL_TERMS:
            self.domain_filter_terms = DOMAIN_TECHNICAL_TERMS[domain_lower]
            self.domain_filter_enabled = enable_filter
            print(f"[SenticNet] Domain filter enabled: '{domain}' "
                  f"({len(self.domain_filter_terms)} terms)")
        else:
            self.domain_filter_terms = set()
            self.domain_filter_enabled = False
            print(f"[SenticNet] Unknown domain '{domain}', filter disabled")

        # 清空緩存（因為過濾規則變了）
        self.polarity_cache.clear()

    def disable_domain_filter(self):
        """禁用領域過濾"""
        self.domain_filter_enabled = False
        self.polarity_cache.clear()
        print("[SenticNet] Domain filter disabled")

    def enable_domain_filter(self):
        """啟用領域過濾（需先設置領域）"""
        if self.domain_filter_terms:
            self.domain_filter_enabled = True
            self.polarity_cache.clear()
            print("[SenticNet] Domain filter enabled")
        else:
            print("[SenticNet] Warning: No domain set, cannot enable filter")

    def is_technical_term(self, word: str) -> bool:
        """
        檢查詞彙是否為當前領域的技術術語

        Args:
            word: 待檢查的詞彙

        Returns:
            bool: True 如果是技術術語，應被過濾
        """
        if not self.domain_filter_enabled:
            return False

        clean_word = self._clean_word(word)
        return clean_word in self.domain_filter_terms

    def add_technical_terms(self, terms: List[str]):
        """
        動態添加技術術語到過濾列表

        Args:
            terms: 要添加的術語列表
        """
        self.domain_filter_terms.update(term.lower() for term in terms)
        self.polarity_cache.clear()
        print(f"[SenticNet] Added {len(terms)} custom technical terms")

    # =========================================================================
    # 極性查詢（含領域過濾）
    # =========================================================================

    def get_polarity(self, word: str) -> float:
        """
        獲取詞彙的情感極性值（含領域過濾）

        Args:
            word: 詞彙（會自動轉小寫並清理）

        Returns:
            float: 極性值 [-1, 1]，0 表示中性或未找到
        """
        # 檢查緩存
        if word in self.polarity_cache:
            return self.polarity_cache[word]

        # 清理詞彙
        clean_word = self._clean_word(word)

        if not clean_word:
            return 0.0

        # ========== 領域過濾：技術術語強制返回中性 ==========
        if self.domain_filter_enabled and clean_word in self.domain_filter_terms:
            self.polarity_cache[word] = 0.0
            return 0.0

        # 查詢 SenticNet
        if clean_word in self.senticnet:
            # polarity_value 是第 8 個元素（索引 7）
            polarity = self.senticnet[clean_word][7]
            self.polarity_cache[word] = polarity
            return polarity

        # 嘗試下劃線連接的多詞概念
        underscored = clean_word.replace(' ', '_')
        if underscored in self.senticnet:
            polarity = self.senticnet[underscored][7]
            self.polarity_cache[word] = polarity
            return polarity

        # 未找到，返回中性
        self.polarity_cache[word] = 0.0
        return 0.0

    def get_polarity_with_coverage(self, word: str) -> tuple:
        """
        獲取詞彙極性值，同時返回是否在知識庫中

        這個方法解決「中性詞」與「未登錄詞」的混淆問題：
        - is_known=True, polarity=0.0: 這個詞在 SenticNet 中，且是中性的
        - is_known=False: 這個詞不在 SenticNet 中，我們不確定它的情感

        Args:
            word: 詞彙

        Returns:
            (polarity, is_known): 極性值和是否已知
        """
        clean_word = self._clean_word(word)

        if not clean_word:
            return 0.0, False

        # 領域過濾的詞視為「已知的中性」
        if self.domain_filter_enabled and clean_word in self.domain_filter_terms:
            return 0.0, True

        # 在 SenticNet 中
        if clean_word in self.senticnet:
            return self.senticnet[clean_word][7], True

        # 下劃線形式
        underscored = clean_word.replace(' ', '_')
        if underscored in self.senticnet:
            return self.senticnet[underscored][7], True

        # 未找到
        return 0.0, False

    def get_coverage_mask(self, words: List[str]) -> torch.Tensor:
        """
        獲取詞彙列表的覆蓋掩碼

        Args:
            words: 詞彙列表

        Returns:
            mask: [len(words)]，1 表示在知識庫中，0 表示未知
        """
        mask = []
        for word in words:
            _, is_known = self.get_polarity_with_coverage(word)
            mask.append(1.0 if is_known else 0.0)
        return torch.tensor(mask, dtype=torch.float32)

    def _clean_word(self, word: str) -> str:
        """
        清理詞彙用於查詢

        處理：
        - BERT subword 前綴 (##)
        - 特殊標記 ([CLS], [SEP], [PAD])
        - 大小寫
        - 標點符號
        """
        if not word:
            return ""

        # 移除 BERT subword 前綴
        word = word.replace('##', '')

        # 跳過特殊標記
        if word.startswith('[') and word.endswith(']'):
            return ""

        # 轉小寫
        word = word.lower()

        # 移除標點（保留字母和數字）
        word = re.sub(r'[^a-z0-9\s]', '', word)

        return word.strip()

    def get_polarity_label(self, word: str) -> str:
        """
        獲取詞彙的極性標籤

        Returns:
            'positive', 'negative', 或 'neutral'
        """
        clean_word = self._clean_word(word)

        if clean_word in self.senticnet:
            return self.senticnet[clean_word][6]  # polarity_label

        return 'neutral'

    def get_primary_emotion(self, word: str) -> Optional[str]:
        """
        獲取詞彙的主要情感

        Returns:
            情感標籤如 '#joy', '#sadness' 等，或 None
        """
        clean_word = self._clean_word(word)

        if clean_word in self.senticnet:
            return self.senticnet[clean_word][4]  # primary_emotion

        return None

    def batch_lookup(
        self,
        tokens: List[str],
        device: torch.device = None
    ) -> torch.Tensor:
        """
        批量查詢多個詞彙的情感極性

        Args:
            tokens: 詞彙列表
            device: 輸出張量的設備

        Returns:
            torch.Tensor: [len(tokens)] 極性值張量
        """
        polarities = [self.get_polarity(token) for token in tokens]
        tensor = torch.tensor(polarities, dtype=torch.float32)

        if device is not None:
            tensor = tensor.to(device)

        return tensor

    def batch_lookup_2d(
        self,
        token_lists: List[List[str]],
        device: torch.device = None,
        pad_value: float = 0.0
    ) -> torch.Tensor:
        """
        批量查詢 2D token 列表（用於 batch 處理）

        Args:
            token_lists: 2D 詞彙列表 [batch_size, seq_len]
            device: 輸出張量的設備
            pad_value: 填充值

        Returns:
            torch.Tensor: [batch_size, seq_len] 極性值張量
        """
        batch_size = len(token_lists)
        max_len = max(len(tokens) for tokens in token_lists)

        # 初始化為填充值
        polarities = torch.full((batch_size, max_len), pad_value)

        for i, tokens in enumerate(token_lists):
            for j, token in enumerate(tokens):
                polarities[i, j] = self.get_polarity(token)

        if device is not None:
            polarities = polarities.to(device)

        return polarities

    def get_related_concepts(self, word: str, top_k: int = 5) -> List[str]:
        """
        獲取相關概念（語義相似詞）

        Args:
            word: 查詢詞彙
            top_k: 返回前 k 個相關概念

        Returns:
            相關概念列表
        """
        clean_word = self._clean_word(word)

        if clean_word not in self.senticnet:
            return []

        # semantics 是索引 8-12
        semantics = self.senticnet[clean_word][8:13]

        # 過濾掉重複和自身
        unique_concepts = []
        for concept in semantics:
            if concept and concept != clean_word and concept not in unique_concepts:
                unique_concepts.append(concept)

        return unique_concepts[:top_k]

    def __len__(self) -> int:
        """返回知識庫大小"""
        return len(self.senticnet)

    def __contains__(self, word: str) -> bool:
        """檢查詞彙是否在知識庫中"""
        clean_word = self._clean_word(word)
        return clean_word in self.senticnet


# 全局單例（避免重複加載）
_senticnet_instance: Optional[SenticNetKnowledge] = None


def get_senticnet(path: str = None) -> SenticNetKnowledge:
    """
    獲取 SenticNet 單例實例

    Args:
        path: SenticNet 文件路徑（僅首次調用時有效）

    Returns:
        SenticNetKnowledge 實例
    """
    global _senticnet_instance

    if _senticnet_instance is None:
        _senticnet_instance = SenticNetKnowledge(path)

    return _senticnet_instance


def reset_senticnet():
    """重置 SenticNet 單例（用於測試）"""
    global _senticnet_instance
    _senticnet_instance = None


if __name__ == "__main__":
    # 測試
    senticnet = SenticNetKnowledge()

    test_words = ['good', 'bad', 'excellent', 'terrible', 'neutral', 'food', 'delicious']

    print("\n=== SenticNet Polarity Test ===")
    for word in test_words:
        polarity = senticnet.get_polarity(word)
        label = senticnet.get_polarity_label(word)
        emotion = senticnet.get_primary_emotion(word)
        print(f"  {word:15s} → polarity={polarity:+.3f}, label={label:10s}, emotion={emotion}")

    print("\n=== Related Concepts ===")
    for word in ['good', 'bad']:
        related = senticnet.get_related_concepts(word)
        print(f"  {word}: {related}")
