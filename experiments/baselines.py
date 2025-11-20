"""
Baseline Models for ABSA

這個檔案包含用於對比實驗的 baseline 模型：
1. Baseline_BERT_Only: 最基礎的 BERT baseline
2. Baseline_BERT_AAHA: BERT + AAHA (無 PMAC/IARM)

用途：
- 證明 PMAC 和 IARM 創新的有效性
- 提供消融實驗的對照組
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bert_embedding import BERTForABSA
from models.aaha_enhanced import AAHAEnhanced
from models.base_model import BaseModel


class Baseline_BERT_Only(BaseModel):
    """
    Baseline 1: BERT Only (最基礎的 baseline)

    架構:
        Text + Aspect → BERT → [CLS] → Classifier → Logit

    特點:
        - 最簡單的方法
        - 每個 aspect 獨立處理
        - 無任何 aspect 建模機制
        - 無 aspect 間交互

    預期:
        - 性能最差
        - 作為最基礎的對照組
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super(Baseline_BERT_Only, self).__init__()

        self.num_classes = num_classes

        # BERT 編碼器
        self.bert = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert.hidden_size

        # 簡單的分類器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_size, num_classes)
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        參數:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            aspect_input_ids: [batch, max_aspects, aspect_len]
            aspect_attention_mask: [batch, max_aspects, aspect_len]
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
        """
        batch_size, max_aspects, aspect_len = aspect_input_ids.shape
        seq_len = text_input_ids.shape[1]

        logits_list = []

        for i in range(max_aspects):
            # 只處理有效的 aspects
            if aspect_mask[:, i].any():
                # 拼接 text 和 aspect: [CLS] text [SEP] aspect [SEP]
                # text_input_ids: [batch, seq_len]
                # aspect_input_ids[:, i, :]: [batch, aspect_len]

                combined_ids = torch.cat([
                    text_input_ids,
                    aspect_input_ids[:, i, :]
                ], dim=1)  # [batch, seq_len + aspect_len]

                combined_mask = torch.cat([
                    text_attention_mask,
                    aspect_attention_mask[:, i, :]
                ], dim=1)  # [batch, seq_len + aspect_len]

                # BERT encoding
                embeddings = self.bert.bert_embedding(
                    combined_ids,
                    attention_mask=combined_mask
                )  # [batch, seq_len + aspect_len, hidden_size]

                # 使用 [CLS] token (第一個 token)
                cls_token = embeddings[:, 0, :]  # [batch, hidden_size]

                # 分類
                logit = self.classifier(cls_token)  # [batch, num_classes]
                logits_list.append(logit)
            else:
                # 無效 aspect，填充零向量
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=text_input_ids.device)
                )

        # 堆疊所有 aspects 的 logits
        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]

        return logits, None  # None 是 gate_stats (baseline 沒有 gate)


class Baseline_BERT_AAHA(BaseModel):
    """
    Baseline 2: BERT + AAHA (無 PMAC/IARM)

    架構:
        Text → BERT → Text Embeddings
        Aspect → BERT → Aspect Embeddings
        (Text, Aspect) → AAHA → Context → Classifier → Logit

    特點:
        - 包含階層式注意力 (AAHA)
        - 每個 aspect 仍然獨立處理
        - 無 aspect 間的組合 (PMAC)
        - 無 aspect 間的關係建模 (IARM)

    用途:
        - 證明 PMAC 和 IARM 的貢獻
        - 這是您目前實驗中 "No PMAC/IARM" 的版本

    預期:
        - 比 BERT Only 好
        - 但應該比 Full Model 差
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super(Baseline_BERT_AAHA, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # BERT 編碼器
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert_absa.hidden_size

        # 投影層（如果需要）
        if bert_hidden_size != hidden_dim:
            self.text_projection = nn.Linear(bert_hidden_size, hidden_dim)
            self.aspect_projection = nn.Linear(bert_hidden_size, hidden_dim)
        else:
            self.text_projection = nn.Identity()
            self.aspect_projection = nn.Identity()

        # AAHA 模組（為每個 aspect 提取上下文）
        self.aaha = AAHAEnhanced(
            hidden_dim=hidden_dim,
            aspect_dim=hidden_dim,
            word_attention_dims=[hidden_dim // 2],
            phrase_attention_dims=[hidden_dim // 2],
            sentence_attention_dims=[hidden_dim],
            attention_dropout=0.0,
            output_dropout=dropout
        )

        # 分類器（為每個 aspect 獨立分類）
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        參數:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            aspect_input_ids: [batch, max_aspects, aspect_len]
            aspect_attention_mask: [batch, max_aspects, aspect_len]
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
        """
        batch_size, max_aspects, aspect_len = aspect_input_ids.shape

        # 1. BERT 編碼文本（一次）
        text_emb = self.bert_absa.bert_embedding(
            text_input_ids,
            attention_mask=text_attention_mask
        )  # [batch, seq_len, bert_dim]

        text_hidden = self.text_projection(text_emb)  # [batch, seq_len, hidden_dim]

        # 2. BERT 編碼每個 aspect
        aspect_hidden_list = []
        for i in range(max_aspects):
            # 只處理有效的 aspects
            if aspect_mask[:, i].any():
                asp_emb = self.bert_absa.bert_embedding(
                    aspect_input_ids[:, i, :],
                    attention_mask=aspect_attention_mask[:, i, :]
                )  # [batch, aspect_len, bert_dim]

                # 使用 [CLS] token
                asp_repr = asp_emb[:, 0, :]  # [batch, bert_dim]
                asp_hidden = self.aspect_projection(asp_repr)  # [batch, hidden_dim]
            else:
                asp_hidden = torch.zeros(batch_size, self.hidden_dim, device=text_hidden.device)

            aspect_hidden_list.append(asp_hidden)

        aspect_hiddens = torch.stack(aspect_hidden_list, dim=1)  # [batch, max_aspects, hidden_dim]

        # 3. AAHA - 為每個 aspect 提取上下文
        context_vectors = []
        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                ctx, _ = self.aaha(
                    text_hidden,
                    aspect_hiddens[:, i, :],
                    text_attention_mask.float()
                )
                context_vectors.append(ctx)
            else:
                context_vectors.append(
                    torch.zeros(batch_size, self.hidden_dim, device=text_hidden.device)
                )

        context_vectors = torch.stack(context_vectors, dim=1)  # [batch, max_aspects, hidden_dim]

        # 4. 分類（無 PMAC，無 IARM）
        logits = self.classifier(context_vectors)  # [batch, max_aspects, num_classes]

        return logits, None  # None 是 gate_stats


class Baseline_BERT_MeanPooling(BaseModel):
    """
    Baseline 3: BERT + Mean Pooling (可選的 baseline)

    架構:
        Text → BERT → Mean Pooling → Classifier → Logit

    特點:
        - 將所有 aspects 的表示簡單平均
        - 完全無 aspect 特定建模
        - 更簡單但可能效果很差

    用途:
        - 展示簡單方法的局限性
        - 可選的對照組
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super(Baseline_BERT_MeanPooling, self).__init__()

        self.bert = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_size, num_classes)
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """前向傳播"""
        batch_size, max_aspects, _ = aspect_input_ids.shape

        # 簡單編碼文本
        text_emb = self.bert.bert_embedding(
            text_input_ids,
            attention_mask=text_attention_mask
        )  # [batch, seq_len, hidden_size]

        # Mean pooling
        # 使用 attention_mask 進行加權平均
        mask_expanded = text_attention_mask.unsqueeze(-1).expand(text_emb.size()).float()
        sum_embeddings = torch.sum(text_emb * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask  # [batch, hidden_size]

        # 為每個 aspect 使用相同的表示（簡化版）
        logits_per_aspect = self.classifier(mean_pooled)  # [batch, num_classes]

        # 複製到所有 aspects
        logits = logits_per_aspect.unsqueeze(1).repeat(1, max_aspects, 1)  # [batch, max_aspects, num_classes]

        # 應用 aspect_mask
        mask_expanded = aspect_mask.unsqueeze(-1).float()
        logits = logits * mask_expanded

        return logits, None


# 工廠函數：方便創建不同的 baseline
class Baseline_HierarchicalBERT(BaseModel):
    """
    Baseline 4: Hierarchical BERT (階層式BERT)

    架構:
        BERT Layers → Extract Multi-Level Features → Hierarchical Fusion → Classifier

    階層:
        - Low-level (layers 1-2): Syntactic features (詞法特徵)
        - Mid-level (layers 3-4): Semantic features (語義特徵)
        - High-level (layers 5-6): Task-specific features (任務特徵)

    特點:
        - 利用BERT不同層的階層特性
        - 融合多層級特徵
        - 簡單、有效、可解釋

    預期:
        - 優於BERT Only (利用階層信息)
        - 與AAHA競爭 (但更簡單)
    """

    def __init__(
        self,
        bert_model_name: str = 'distilbert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 768,
        num_classes: int = 3,
        dropout: float = 0.1
    ):
        super(Baseline_HierarchicalBERT, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # BERT with hierarchical output
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        # Enable output of all hidden states
        self.bert_absa.bert_embedding.output_hidden_states = True
        self.bert_absa.bert_embedding.bert.config.output_hidden_states = True

        bert_hidden_size = self.bert_absa.hidden_size

        # Projection if needed
        if bert_hidden_size != hidden_dim:
            self.text_projection = nn.Linear(bert_hidden_size, hidden_dim)
            self.aspect_projection = nn.Linear(bert_hidden_size, hidden_dim)
        else:
            self.text_projection = nn.Identity()
            self.aspect_projection = nn.Identity()

        # Hierarchical fusion layers
        # DistilBERT has 6 layers, BERT has 12 layers
        is_distilbert = 'distilbert' in bert_model_name.lower()

        if is_distilbert:
            # DistilBERT: 6 layers
            # Low: 1-2, Mid: 3-4, High: 5-6
            self.low_layers = [1, 2]
            self.mid_layers = [3, 4]
            self.high_layers = [5, 6]
        else:
            # BERT: 12 layers
            # Low: 1-4, Mid: 5-8, High: 9-12
            self.low_layers = [1, 2, 3, 4]
            self.mid_layers = [5, 6, 7, 8]
            self.high_layers = [9, 10, 11, 12]

        # Fusion layers for each level
        self.low_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(self.low_layers), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.mid_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(self.mid_layers), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.high_fusion = nn.Sequential(
            nn.Linear(hidden_dim * len(self.high_layers), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 levels
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        aspect_mask: torch.Tensor
    ):
        """
        前向傳播

        參數:
            text_input_ids: [batch, seq_len]
            text_attention_mask: [batch, seq_len]
            aspect_input_ids: [batch, max_aspects, aspect_len]
            aspect_attention_mask: [batch, max_aspects, aspect_len]
            aspect_mask: [batch, max_aspects] - bool，標記有效 aspects

        返回:
            logits: [batch, max_aspects, num_classes]
        """
        batch_size, max_aspects, aspect_len = aspect_input_ids.shape

        logits_list = []

        for i in range(max_aspects):
            if aspect_mask[:, i].any():
                # Concatenate text and aspect
                combined_ids = torch.cat([
                    text_input_ids,
                    aspect_input_ids[:, i, :]
                ], dim=1)

                combined_mask = torch.cat([
                    text_attention_mask,
                    aspect_attention_mask[:, i, :]
                ], dim=1)

                # Get hierarchical features from BERT
                outputs = self.bert_absa.bert_embedding.bert(
                    input_ids=combined_ids,
                    attention_mask=combined_mask,
                    return_dict=True
                )

                all_hidden_states = outputs.hidden_states  # tuple of (num_layers+1) tensors

                # Extract CLS token from different layers
                # all_hidden_states[0] is embedding layer, so layers start from index 1
                low_features = []
                for layer_idx in self.low_layers:
                    low_features.append(all_hidden_states[layer_idx][:, 0, :])  # CLS token
                low_features = torch.cat(low_features, dim=-1)  # [batch, hidden*num_low]

                mid_features = []
                for layer_idx in self.mid_layers:
                    mid_features.append(all_hidden_states[layer_idx][:, 0, :])
                mid_features = torch.cat(mid_features, dim=-1)

                high_features = []
                for layer_idx in self.high_layers:
                    high_features.append(all_hidden_states[layer_idx][:, 0, :])
                high_features = torch.cat(high_features, dim=-1)

                # Hierarchical fusion
                low_fused = self.low_fusion(low_features)   # [batch, hidden]
                mid_fused = self.mid_fusion(mid_features)   # [batch, hidden]
                high_fused = self.high_fusion(high_features) # [batch, hidden]

                # Concatenate all levels
                hierarchical_repr = torch.cat([low_fused, mid_fused, high_fused], dim=-1)  # [batch, hidden*3]

                # Classification
                logit = self.classifier(hierarchical_repr)  # [batch, num_classes]
                logits_list.append(logit)
            else:
                logits_list.append(
                    torch.zeros(batch_size, self.num_classes, device=text_input_ids.device)
                )

        logits = torch.stack(logits_list, dim=1)  # [batch, max_aspects, num_classes]

        return logits, None  # None for gate_stats


def create_baseline(
    baseline_type: str,
    bert_model_name: str = 'distilbert-base-uncased',
    freeze_bert: bool = False,
    hidden_dim: int = 768,
    num_classes: int = 3,
    dropout: float = 0.1
):
    """
    創建 baseline 模型

    參數:
        baseline_type: 'bert_only', 'bert_aaha', 'bert_mean', 'bert_hierarchical'
        其他參數與模型一致

    返回:
        baseline 模型實例
    """
    if baseline_type == 'bert_only':
        return Baseline_BERT_Only(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            num_classes=num_classes,
            dropout=dropout
        )

    elif baseline_type == 'bert_aaha':
        return Baseline_BERT_AAHA(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    elif baseline_type == 'bert_mean':
        return Baseline_BERT_MeanPooling(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            num_classes=num_classes,
            dropout=dropout
        )

    elif baseline_type == 'bert_hierarchical':
        return Baseline_HierarchicalBERT(
            bert_model_name=bert_model_name,
            freeze_bert=freeze_bert,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout
        )

    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}. "
                        f"Choose from: 'bert_only', 'bert_aaha', 'bert_mean', 'bert_hierarchical'")


def run_all_baselines():
    """
    一鍵執行所有 baseline 實驗並生成報告

    執行方法:
        python experiments/baselines.py --run_all
    """
    import subprocess
    import json
    from datetime import datetime
    import pandas as pd

    project_root = Path(__file__).parent.parent
    baseline_dir = project_root / "results" / "baseline"

    # 定義三個 baseline 實驗配置
    baselines = [
        ("bert_only", "BERT Only", 32),
        ("bert_aaha", "BERT + AAHA", 24),
        ("bert_mean", "BERT + Mean Pooling", 32)
    ]

    print("\n" + "="*80)
    print("開始執行所有 Baseline 實驗")
    print("="*80)
    print(f"總共 {len(baselines)} 個實驗")
    print("預估總時長: 約 1.5-3 小時 (視 GPU 性能而定)\n")

    # 依序運行所有實驗
    for idx, (baseline_type, description, batch_size) in enumerate(baselines, 1):
        print(f"\n[{idx}/{len(baselines)}] 開始運行: {description}")
        print("="*80 + "\n")

        cmd = [
            "python", "experiments/train_multiaspect.py",
            "--baseline", baseline_type,
            "--epochs", "30",
            "--batch_size", str(batch_size),
            "--accumulation_steps", "1",
            "--lr", "2e-5",
            "--dropout", "0.3",
            "--loss_type", "focal",
            "--focal_gamma", "2.5",  # 與 Full Model 一致
            "--class_weights", "1.0", "5.0", "1.0"  # 與 Full Model 一致
        ]

        try:
            # 直接運行，輸出到終端
            result = subprocess.run(cmd, cwd=str(project_root), check=True)
            print(f"\n✓ {description} 完成\n")
        except subprocess.CalledProcessError as e:
            print(f"\n✗ {description} 失敗 (錯誤碼: {e.returncode})\n")
        except KeyboardInterrupt:
            print(f"\n中斷執行")
            return

    print("\n" + "="*80)
    print("所有 Baseline 實驗完成！")
    print("="*80 + "\n")

    # 生成報告
    print("="*80)
    print("正在生成統整報告...")
    print("="*80 + "\n")

    # 調用獨立的報告生成腳本
    subprocess.run(["python", "experiments/generate_baseline_report.py"], cwd=str(project_root))


def generate_baseline_report():
    """生成 baseline 比較報告"""
    import json
    from datetime import datetime
    import pandas as pd

    project_root = Path(__file__).parent.parent
    baseline_dir = project_root / "results" / "baseline"
    report_dir = project_root / "results" / "baseline_comparison"
    report_dir.mkdir(parents=True, exist_ok=True)

    if not baseline_dir.exists():
        print("錯誤: results/baseline/ 目錄不存在")
        print("請先運行 baseline 實驗")
        return

    # 定義三個 baseline
    baselines = [
        ("bert_only", "BERT Only", 32),
        ("bert_aaha", "BERT + AAHA", 24),
        ("bert_mean", "BERT + Mean Pooling", 32)
    ]

    results = []

    print("收集 Baseline 實驗結果\n")

    for baseline_type, description, batch_size in baselines:
        print(f"查找 {description}...")
        exp_dir = _find_latest_baseline(baseline_dir, baseline_type)

        if exp_dir:
            metrics = _read_metrics(exp_dir)
            results.append({
                'type': baseline_type,
                'description': description,
                'batch_size': batch_size,
                'exp_dir': exp_dir,
                **metrics
            })
            print(f"  ✓ 找到: {exp_dir.name}")
            if metrics['test_acc']:
                print(f"    Test Acc: {metrics['test_acc']:.4f}, F1: {metrics['test_f1']:.4f}")
        else:
            print(f"  ✗ 未找到 {baseline_type} 的實驗結果")
            results.append({
                'type': baseline_type,
                'description': description,
                'batch_size': batch_size,
                'exp_dir': None,
                'test_acc': None,
                'test_f1': None,
                'best_epoch': None,
                'timestamp': None
            })

    if not any(r['test_acc'] for r in results):
        print("\n錯誤: 沒有找到任何有效的實驗結果")
        return

    # 生成報告
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Markdown 報告
    md_report = _generate_markdown(results)
    md_path = report_dir / f"baseline_comparison_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)

    # JSON 報告
    json_data = [{k: str(v) if isinstance(v, Path) else v for k, v in r.items()} for r in results]
    json_path = report_dir / f"baseline_comparison_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)

    # CSV 報告
    csv_data = []
    for r in results:
        csv_data.append({
            'Baseline': r['description'],
            'Type': r['type'],
            'Batch_Size': r['batch_size'],
            'Test_Accuracy': r['test_acc'],
            'Test_F1': r['test_f1'],
            'Best_Epoch': r['best_epoch'],
            'Timestamp': r['timestamp']
        })
    df = pd.DataFrame(csv_data)
    csv_path = report_dir / f"baseline_comparison_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("\n" + "="*80)
    print("報告生成完成:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {json_path}")
    print(f"  - CSV: {csv_path}")
    print("="*80 + "\n")

    # 打印摘要
    _print_summary(results)


def _find_latest_baseline(baseline_dir, baseline_type):
    """查找最新的 baseline 實驗結果"""
    pattern = f"*_baseline_{baseline_type}_*"
    dirs = [d for d in baseline_dir.glob(pattern) if d.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda x: x.stat().st_mtime)


def _read_metrics(exp_dir):
    """從實驗目錄讀取指標"""
    from datetime import datetime
    import json

    metrics = {
        'test_acc': None,
        'test_f1': None,
        'best_epoch': None,
        'timestamp': None
    }

    # 嘗試從 experiment_results.json 讀取
    results_file = exp_dir / "reports" / "experiment_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 讀取 test_metrics
                test_metrics = data.get('test_metrics', {})
                metrics['test_acc'] = test_metrics.get('accuracy')
                metrics['test_f1'] = test_metrics.get('f1_macro')

                # 讀取 best epoch (從 validation F1 最高的 epoch)
                if 'best_val_f1' in data and 'history' in data:
                    val_f1_list = data['history'].get('val_f1_macro', [])
                    if val_f1_list:
                        best_f1 = data['best_val_f1']
                        # 找到對應的 epoch
                        for i, f1 in enumerate(val_f1_list):
                            if abs(f1 - best_f1) < 1e-6:
                                metrics['best_epoch'] = i + 1
                                break
        except Exception as e:
            print(f"    讀取錯誤: {e}")
            pass

    # 嘗試從報告文件讀取
    if metrics['test_acc'] is None:
        report_files = list((exp_dir / "reports").glob("*.txt"))
        if report_files:
            latest_report = max(report_files, key=lambda x: x.stat().st_mtime)
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if 'Best Test Accuracy:' in line or 'Best Test Acc:' in line:
                            try:
                                metrics['test_acc'] = float(line.split(':')[1].strip())
                            except:
                                pass
                        elif 'Best Test F1:' in line:
                            try:
                                metrics['test_f1'] = float(line.split(':')[1].strip())
                            except:
                                pass
                        elif 'Best Epoch:' in line:
                            try:
                                metrics['best_epoch'] = int(line.split(':')[1].strip())
                            except:
                                pass
            except:
                pass

    # 獲取時間戳
    dir_name = exp_dir.name
    timestamp_str = dir_name.split('_')[0] + '_' + dir_name.split('_')[1]
    try:
        metrics['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
    except:
        metrics['timestamp'] = datetime.fromtimestamp(exp_dir.stat().st_mtime)

    return metrics


def _generate_markdown(results):
    """生成 Markdown 報告"""
    from datetime import datetime

    report = "# Baseline 實驗比較報告\n\n"
    report += f"生成時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    report += "## 實驗配置\n\n"
    report += "| 參數 | 值 |\n"
    report += "|------|----|\n"
    report += "| Epochs | 30 |\n"
    report += "| Learning Rate | 2e-5 |\n"
    report += "| Dropout | 0.3 |\n"
    report += "| Loss Type | Focal Loss (gamma=2.0) |\n"
    report += "| Class Weights | [1.0, 3.0, 1.0] |\n"
    report += "| GPU | RTX 3090 (24GB) |\n\n"

    report += "## 實驗結果對比\n\n"
    report += "| Baseline | Batch Size | Test Acc | Test F1 | Best Epoch | 實驗時間 |\n"
    report += "|----------|------------|----------|---------|------------|----------|\n"

    for r in results:
        acc = f"{r['test_acc']:.4f}" if r['test_acc'] else "N/A"
        f1 = f"{r['test_f1']:.4f}" if r['test_f1'] else "N/A"
        epoch = r['best_epoch'] if r['best_epoch'] else "N/A"
        ts = r['timestamp'].strftime('%Y-%m-%d %H:%M') if r['timestamp'] else "N/A"

        report += f"| {r['description']} | {r['batch_size']} | {acc} | {f1} | {epoch} | {ts} |\n"

    # 找出最佳模型
    valid_results = [r for r in results if r['test_acc']]
    if valid_results:
        best_acc = max(valid_results, key=lambda x: x['test_acc'])
        best_f1 = max(valid_results, key=lambda x: x['test_f1'] or 0)

        report += "\n## 詳細分析\n\n"
        report += f"### 最佳準確率\n"
        report += f"- 模型: **{best_acc['description']}**\n"
        report += f"- Test Accuracy: **{best_acc['test_acc']:.4f}**\n"
        report += f"- Test F1: **{best_acc['test_f1']:.4f}**\n\n"

        report += f"### 最佳 F1 分數\n"
        report += f"- 模型: **{best_f1['description']}**\n"
        report += f"- Test F1: **{best_f1['test_f1']:.4f}**\n"
        report += f"- Test Accuracy: **{best_f1['test_acc']:.4f}**\n\n"

    report += "## 結論\n\n"
    report += "根據以上實驗結果，可以觀察到:\n\n"
    report += "1. **BERT Only**: 最簡單的基線，僅使用 BERT 的 [CLS] token\n"
    report += "2. **BERT + AAHA**: 加入層次化注意力機制，但不包含 PMAC/IARM\n"
    report += "3. **BERT + Mean Pooling**: 簡單的平均池化方法\n\n"
    report += "這些基線將作為評估 PMAC 和 IARM 創新模組效果的對照組。\n"

    return report


def _print_summary(results):
    """打印摘要"""
    print("\n" + "="*80)
    print("BASELINE 實驗結果摘要")
    print("="*80 + "\n")

    for r in results:
        print(f"【{r['description']}】")
        if r['test_acc']:
            print(f"  Test Accuracy: {r['test_acc']:.4f}")
            print(f"  Test F1: {r['test_f1']:.4f}")
            print(f"  Best Epoch: {r['best_epoch']}")
        else:
            print(f"  未找到實驗結果")
        print()


if __name__ == '__main__':
    """
    使用方法:
        1. 執行所有 baseline 實驗並生成報告:
           python experiments/baselines.py --run_all

        2. 測試 baseline 模型:
           python experiments/baselines.py --test

        3. 僅生成報告 (不重新訓練):
           python experiments/generate_baseline_report.py
    """
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--run_all':
            run_all_baselines()
        elif sys.argv[1] == '--test':
            # 原本的測試代碼
            print("Testing Baseline Models...")
            print("="*60)

            batch_size = 4
            seq_len = 64
            aspect_len = 8
            max_aspects = 3
            num_classes = 3

            text_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            text_attention_mask = torch.ones(batch_size, seq_len)
            aspect_input_ids = torch.randint(0, 1000, (batch_size, max_aspects, aspect_len))
            aspect_attention_mask = torch.ones(batch_size, max_aspects, aspect_len)
            aspect_mask = torch.tensor([
                [True, True, False],
                [True, True, True],
                [True, False, False],
                [True, True, True]
            ])

            for baseline_type in ['bert_only', 'bert_aaha', 'bert_mean']:
                print(f"\nTesting: {baseline_type}")
                print("-"*60)

                model = create_baseline(
                    baseline_type=baseline_type,
                    bert_model_name='distilbert-base-uncased',
                    num_classes=num_classes,
                    dropout=0.1
                )

                logits, gate_stats = model(
                    text_input_ids,
                    text_attention_mask,
                    aspect_input_ids,
                    aspect_attention_mask,
                    aspect_mask
                )

                print(f"[OK] Model created successfully")
                print(f"  Input shape: text={text_input_ids.shape}, aspects={aspect_input_ids.shape}")
                print(f"  Output shape: {logits.shape}")
                print(f"  Gate stats: {gate_stats}")

                num_params = sum(p.numel() for p in model.parameters())
                print(f"  Total parameters: {num_params:,}")

            print("\n" + "="*60)
            print("All baseline models tested successfully!")
        else:
            print("未知參數。請使用:")
            print("  --run_all : 執行所有實驗並生成報告")
            print("  --test    : 測試模型")
    else:
        print("請指定參數:")
        print("  python experiments/baselines.py --run_all  # 執行所有實驗並生成報告")
        print("  python experiments/baselines.py --test     # 測試模型")
        print("  python experiments/generate_baseline_report.py  # 僅生成報告")
