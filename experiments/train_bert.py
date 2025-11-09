"""
HMAC-Net with BERT 訓練腳本（碩士論文實驗）

功能:
    - 使用真實 SemEval-2014 數據
    - BERT 動態嵌入（比靜態 GloVe 更好）
    - 完整的訓練、驗證流程
    - Early Stopping
    - 模型保存和視覺化
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np
from transformers import BertTokenizer

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bert_embedding import BERTForABSA
from models import AAHA, PMAC, IARM
from models.base_model import BaseModel, MLP
from utils import (
    get_logger,
    MetricsCalculator,
    RunningMetrics,
    SemEvalPreprocessor,
    split_train_val,
    TrainingVisualizer,
    MetricsVisualizer
)
import pandas as pd


class HMACNetBERT(BaseModel):
    """
    HMAC-Net with BERT 嵌入

    使用 BERT 替代傳統的靜態詞嵌入，可以獲得：
    - 上下文感知的詞表示
    - 更好的語義理解
    - 預訓練知識遷移
    """

    def __init__(
        self,
        bert_model_name: str = 'bert-base-uncased',
        freeze_bert: bool = False,
        hidden_dim: int = 256,
        num_classes: int = 3,
        dropout: float = 0.5,
        # AAHA 參數
        word_attention_dim: int = 128,
        phrase_attention_dim: int = 128,
        sentence_attention_dim: int = 128,
        # PMAC 參數
        fusion_dim: int = 256,
        num_composition_layers: int = 2,
        fusion_method: str = "gated",
        # IARM 參數
        use_iarm: bool = True,
        relation_dim: int = 128,
        relation_type: str = "transformer",
        num_heads: int = 4,
        # 分類器參數
        classifier_hidden_dims: list = None
    ):
        """初始化 HMAC-Net with BERT"""
        super(HMACNetBERT, self).__init__()

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [128, 64]

        # BERT 嵌入
        self.bert_absa = BERTForABSA(
            model_name=bert_model_name,
            freeze_bert=freeze_bert
        )

        bert_hidden_size = self.bert_absa.hidden_size

        # 投影層（將 BERT 768 維投影到指定的 hidden_dim）
        self.text_projection = nn.Linear(bert_hidden_size, hidden_dim)
        self.aspect_projection = nn.Linear(bert_hidden_size, hidden_dim)

        # AAHA 模組
        self.aaha = AAHA(
            hidden_dim=hidden_dim,
            aspect_dim=hidden_dim,
            word_attention_dim=word_attention_dim,
            phrase_attention_dim=phrase_attention_dim,
            sentence_attention_dim=sentence_attention_dim,
            dropout=dropout
        )

        # PMAC 模組
        self.pmac = PMAC(
            input_dim=hidden_dim,
            fusion_dim=fusion_dim,
            num_composition_layers=num_composition_layers,
            fusion_method=fusion_method,
            dropout=dropout
        )

        # IARM 模組（可選）
        self.use_iarm = use_iarm
        if use_iarm:
            self.iarm = IARM(
                input_dim=fusion_dim,
                relation_dim=relation_dim,
                relation_type=relation_type,
                num_heads=num_heads,
                num_layers=2,
                dropout=dropout
            )

        # 分類器
        self.classifier = MLP(
            input_dim=fusion_dim,
            hidden_dims=classifier_hidden_dims,
            output_dim=num_classes,
            dropout=dropout,
            activation='relu',
            use_batch_norm=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        aspect_input_ids: torch.Tensor,
        aspect_attention_mask: torch.Tensor,
        return_attention: bool = False
    ):
        """
        前向傳播

        參數:
            text_input_ids: BERT 文本 token IDs [batch, text_len]
            text_attention_mask: 文本注意力掩碼 [batch, text_len]
            aspect_input_ids: BERT 面向 token IDs [batch, aspect_len]
            aspect_attention_mask: 面向注意力掩碼 [batch, aspect_len]
            return_attention: 是否返回注意力權重

        返回:
            分類 logits [batch, num_classes]
        """
        # BERT 編碼
        text_emb, aspect_emb = self.bert_absa(
            text_input_ids, text_attention_mask,
            aspect_input_ids, aspect_attention_mask
        )

        # 投影到指定維度
        text_hidden = self.text_projection(text_emb)
        aspect_hidden = self.aspect_projection(aspect_emb)

        # AAHA 模組
        context_vector, aaha_attention = self.aaha(
            text_hidden, aspect_hidden, text_attention_mask.float()
        )

        # PMAC 模組
        composed_repr = self.pmac(context_vector)

        # IARM 模組
        if self.use_iarm:
            aspect_repr = composed_repr.unsqueeze(1)
            enhanced_repr, iarm_attention = self.iarm(aspect_repr)
            final_repr = enhanced_repr.squeeze(1)
        else:
            final_repr = composed_repr

        # 分類
        logits = self.classifier(final_repr)

        if return_attention:
            attention_weights = {'aaha': aaha_attention}
            return logits, attention_weights
        else:
            return logits, None


class BERTDataset(torch.utils.data.Dataset):
    """
    BERT 數據集（預先 tokenize 版本，加速訓練）
    在初始化時一次性處理所有 tokenization
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 128):
        self.df = dataframe.reset_index(drop=True)
        self.max_length = max_length

        # 預先 tokenize 所有數據（大幅加速訓練）
        print(f"預先 tokenize {len(self.df)} 個樣本...")
        self.encodings = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]

            # 編碼文本
            text_encoding = tokenizer.encode_plus(
                row['text'],
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # 編碼面向
            aspect_encoding = tokenizer.encode_plus(
                row['aspect'],
                max_length=20,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            self.encodings.append({
                'text_input_ids': text_encoding['input_ids'].squeeze(0),
                'text_attention_mask': text_encoding['attention_mask'].squeeze(0),
                'aspect_input_ids': aspect_encoding['input_ids'].squeeze(0),
                'aspect_attention_mask': aspect_encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(row['label'], dtype=torch.long)
            })

        print(f"[OK] Tokenization 完成")

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def collate_fn_bert(batch):
    """BERT 批次整理函數"""
    return {
        'text_input_ids': torch.stack([item['text_input_ids'] for item in batch]),
        'text_attention_mask': torch.stack([item['text_attention_mask'] for item in batch]),
        'aspect_input_ids': torch.stack([item['aspect_input_ids'] for item in batch]),
        'aspect_attention_mask': torch.stack([item['aspect_attention_mask'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch])
    }


class Trainer:
    """訓練器"""

    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, logger, config, project_root=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.config = config
        self.project_root = project_root if project_root else Path.cwd()

        # Early Stopping
        self.patience = config['training']['early_stopping']['patience']
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        # 學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6
        )

        # 指標計算器
        self.metrics_calculator = MetricsCalculator(num_classes=3)

        # 訓練歷史
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_accuracy': [], 'val_accuracy': [],
            'train_macro_f1': [], 'val_macro_f1': []
        }

        # 最佳模型的預測結果（用於最終繪製混淆矩陣）
        self.best_val_labels = None
        self.best_val_preds = None
        self.best_epoch = 0

    def train_epoch(self, epoch):
        """訓練一個 epoch"""
        self.model.train()
        running_metrics = RunningMetrics()
        all_preds, all_labels = [], []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [訓練]')

        for batch in pbar:
            # 移動到設備
            text_ids = batch['text_input_ids'].to(self.device)
            text_mask = batch['text_attention_mask'].to(self.device)
            aspect_ids = batch['aspect_input_ids'].to(self.device)
            aspect_mask = batch['aspect_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向傳播
            logits, _ = self.model(text_ids, text_mask, aspect_ids, aspect_mask)

            # 計算損失
            loss = self.criterion(logits, labels)

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.optimizer.step()

            # 記錄指標
            running_metrics.update(loss=loss.item())
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 計算 epoch 指標
        metrics = self.metrics_calculator.calculate_all(
            np.array(all_labels), np.array(all_preds)
        )

        return {
            'loss': running_metrics.get_average('loss'),
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1']
        }

    @torch.no_grad()
    def validate(self, epoch):
        """驗證"""
        self.model.eval()
        running_metrics = RunningMetrics()
        all_preds, all_labels = [], []

        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [驗證]')

        for batch in pbar:
            text_ids = batch['text_input_ids'].to(self.device)
            text_mask = batch['text_attention_mask'].to(self.device)
            aspect_ids = batch['aspect_input_ids'].to(self.device)
            aspect_mask = batch['aspect_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits, _ = self.model(text_ids, text_mask, aspect_ids, aspect_mask)
            loss = self.criterion(logits, labels)

            running_metrics.update(loss=loss.item())
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        metrics = self.metrics_calculator.calculate_all(
            np.array(all_labels), np.array(all_preds)
        )

        return {
            'loss': running_metrics.get_average('loss'),
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1'],
            'all_preds': all_preds,
            'all_labels': all_labels
        }

    def train(self, num_epochs):
        """完整訓練流程"""
        self.logger.info("=" * 60)
        self.logger.info("開始訓練 HMAC-Net with BERT")
        self.logger.info("=" * 60)

        for epoch in range(1, num_epochs + 1):
            # 訓練
            train_metrics = self.train_epoch(epoch)

            # 驗證
            val_metrics = self.validate(epoch)

            # 記錄歷史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_accuracy'].append(train_metrics['accuracy'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            self.history['train_macro_f1'].append(train_metrics['macro_f1'])
            self.history['val_macro_f1'].append(val_metrics['macro_f1'])

            # 打印指標
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"訓練損失: {train_metrics['loss']:.4f}, F1: {train_metrics['macro_f1']:.4f} | "
                f"驗證損失: {val_metrics['loss']:.4f}, F1: {val_metrics['macro_f1']:.4f}"
            )

            # 學習率調整
            self.scheduler.step(val_metrics['macro_f1'])
            self.logger.info(f"當前學習率: {self.optimizer.param_groups[0]['lr']:.6f}")

            # 保存最佳模型
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.best_epoch = epoch
                self.patience_counter = 0

                save_path = self.project_root / 'results' / 'checkpoints' / f"hmac_bert_best_f1_{self.best_val_f1:.4f}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_f1': self.best_val_f1,
                    'config': self.config
                }, save_path)

                self.logger.info(f"[OK] 保存最佳模型: {save_path}")

                # 儲存最佳 epoch 的預測結果供最後繪圖使用
                self.best_val_labels = val_metrics['all_labels']
                self.best_val_preds = val_metrics['all_preds']

            else:
                self.patience_counter += 1
                self.logger.info(f"驗證 F1 沒有提升 ({self.patience_counter}/{self.patience})")

            # Early Stopping
            if self.patience_counter >= self.patience:
                self.logger.info("Early Stopping 觸發，停止訓練")
                break

        self.logger.info("=" * 60)
        self.logger.info(f"訓練完成！最佳驗證 F1: {self.best_val_f1:.4f}")
        self.logger.info("=" * 60)

        # 繪製訓練曲線
        viz_dir = str(self.project_root / 'results' / 'visualizations')
        viz = TrainingVisualizer(save_dir=viz_dir)
        viz.plot_training_curves(self.history, save_name='hmac_bert_training_curves.png')

        # 繪製最佳模型的混淆矩陣（訓練結束時只生成一張）
        if self.best_val_labels is not None and self.best_val_preds is not None:
            self.logger.info("繪製最佳模型的混淆矩陣...")
            viz = MetricsVisualizer(save_dir=viz_dir)
            cm = self.metrics_calculator.get_confusion_matrix(
                self.best_val_labels, self.best_val_preds
            )
            viz.plot_confusion_matrix(
                cm, ['負面', '中性', '正面'],
                normalize=True,
                save_name='confusion_matrix_best_model.png'
            )
            self.logger.info("[OK] 混淆矩陣已保存")

        # 生成訓練報告 txt
        self._generate_training_report()

    def _generate_training_report(self):
        """生成詳細的訓練報告（txt 格式）"""
        report_path = self.project_root / 'results' / 'reports' / 'training_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("HMAC-Net with BERT 訓練報告\n")
            f.write("=" * 80 + "\n\n")

            # 基本資訊
            f.write("【訓練配置】\n")
            f.write(f"  模型架構: HMAC-Net with BERT\n")
            f.write(f"  BERT 模型: {self.config.get('model', {}).get('bert_model_name', 'bert-base-uncased')}\n")
            f.write(f"  設備: {self.device}\n")
            f.write(f"  批次大小: {self.train_loader.batch_size}\n")
            f.write(f"  學習率: {self.optimizer.param_groups[0]['lr']}\n")
            f.write(f"  優化器: {self.optimizer.__class__.__name__}\n")
            f.write(f"\n")

            # 訓練結果
            f.write("【訓練結果】\n")
            f.write(f"  總訓練輪數: {len(self.history['train_loss'])}\n")
            f.write(f"  最佳驗證 F1: {self.best_val_f1:.4f}\n")
            f.write(f"  最佳模型出現於: Epoch {self.best_epoch}\n")
            f.write(f"\n")

            # 最終指標
            if len(self.history['train_loss']) > 0:
                last_idx = -1
                f.write("【最終訓練指標】\n")
                f.write(f"  訓練損失: {self.history['train_loss'][last_idx]:.4f}\n")
                f.write(f"  訓練準確率: {self.history['train_accuracy'][last_idx]:.4f}\n")
                f.write(f"  訓練 Macro F1: {self.history['train_macro_f1'][last_idx]:.4f}\n")
                f.write(f"\n")

                f.write("【最終驗證指標】\n")
                f.write(f"  驗證損失: {self.history['val_loss'][last_idx]:.4f}\n")
                f.write(f"  驗證準確率: {self.history['val_accuracy'][last_idx]:.4f}\n")
                f.write(f"  驗證 Macro F1: {self.history['val_macro_f1'][last_idx]:.4f}\n")
                f.write(f"\n")

            # 最佳模型的詳細指標
            if self.best_val_labels is not None and self.best_val_preds is not None:
                f.write("【最佳模型詳細指標】\n")
                metrics = self.metrics_calculator.calculate_all(
                    np.array(self.best_val_labels), np.array(self.best_val_preds)
                )
                f.write(f"  準確率 (Accuracy): {metrics['accuracy']:.4f}\n")
                f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
                f.write(f"  Macro Precision: {metrics['macro_precision']:.4f}\n")
                f.write(f"  Macro Recall: {metrics['macro_recall']:.4f}\n")
                f.write(f"\n")

                # 各類別指標
                f.write("【各類別表現】\n")
                class_names = ['負面', '中性', '正面']
                for class_name in class_names:
                    precision = metrics.get(f'{class_name}_precision', 0.0)
                    recall = metrics.get(f'{class_name}_recall', 0.0)
                    f1 = metrics.get(f'{class_name}_f1', 0.0)
                    support = metrics.get(f'{class_name}_support', 0)
                    f.write(f"  {class_name}:\n")
                    f.write(f"    Precision: {precision:.4f}\n")
                    f.write(f"    Recall: {recall:.4f}\n")
                    f.write(f"    F1-Score: {f1:.4f}\n")
                    f.write(f"    Support: {int(support)}\n")
                f.write(f"\n")

                # 混淆矩陣
                cm = self.metrics_calculator.get_confusion_matrix(
                    self.best_val_labels, self.best_val_preds
                )
                f.write("【混淆矩陣】\n")
                f.write("         預測→   負面   中性   正面\n")
                f.write("  實際↓\n")
                for i, class_name in enumerate(class_names):
                    f.write(f"  {class_name:4s}         {cm[i][0]:5d}  {cm[i][1]:5d}  {cm[i][2]:5d}\n")
                f.write(f"\n")

            # 訓練歷史
            f.write("【訓練歷史】\n")
            f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>12} {'Train F1':>12} {'Val F1':>12}\n")
            f.write("-" * 60 + "\n")
            for i in range(len(self.history['train_loss'])):
                f.write(f"{i+1:6d} {self.history['train_loss'][i]:12.4f} "
                       f"{self.history['val_loss'][i]:12.4f} "
                       f"{self.history['train_macro_f1'][i]:12.4f} "
                       f"{self.history['val_macro_f1'][i]:12.4f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("報告生成時間: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"[OK] 訓練報告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='訓練 HMAC-Net with BERT')
    parser.add_argument('--domain', type=str, default='restaurant', choices=['restaurant', 'laptop'],
                        help='數據集領域')
    parser.add_argument('--bert_model', type=str, default='bert-base-uncased',
                        help='BERT 模型名稱')
    parser.add_argument('--freeze_bert', action='store_true',
                        help='是否凍結 BERT 參數')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='訓練輪數')
    parser.add_argument('--lr', type=float, default=2e-5,
                        help='學習率')
    args = parser.parse_args()

    # 獲取專案根目錄
    project_root = Path(__file__).parent.parent

    # 載入配置
    config_path = project_root / 'configs' / 'experiment_config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 初始化日誌
    log_dir = project_root / 'results' / 'logs'
    logger = get_logger('HMAC-BERT-Training', log_dir=str(log_dir), use_tensorboard=False)

    logger.info("HMAC-Net with BERT 訓練腳本（碩士論文實驗）")
    logger.info(f"數據集: SemEval-2014 {args.domain.capitalize()}")
    logger.info(f"BERT 模型: {args.bert_model}")
    logger.info(f"BERT 微調: {'否（凍結）' if args.freeze_bert else '是'}")

    # 設定設備（詳細診斷）
    logger.info("\n" + "=" * 60)
    logger.info("環境診斷")
    logger.info("=" * 60)
    logger.info(f"Python 執行檔: {sys.executable}")
    logger.info(f"PyTorch 版本: {torch.__version__}")
    logger.info(f"PyTorch 路徑: {torch.__file__}")

    cuda_available = torch.cuda.is_available()
    logger.info(f"CUDA 可用: {cuda_available}")

    if cuda_available:
        logger.info(f"CUDA 版本: {torch.version.cuda}")
        logger.info(f"GPU 數量: {torch.cuda.device_count()}")
        logger.info(f"GPU 名稱: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda')
        logger.info(f"使用設備: {device}")
    else:
        logger.error("=" * 60)
        logger.error("[錯誤] CUDA 不可用！")
        logger.error("請檢查:")
        logger.error(f"1. Python 路徑: {sys.executable}")
        logger.error(f"2. PyTorch 版本: {torch.__version__}")
        logger.error("3. 是否使用了 CPU-only 的 Python 環境？")
        logger.error("4. 建議使用: run_training_cuda.bat")
        logger.error("=" * 60)
        device = torch.device('cpu')
        logger.warning("繼續使用 CPU 訓練（會非常慢）...")

    logger.info("=" * 60)

    # 設定隨機種子
    torch.manual_seed(42)
    np.random.seed(42)
    if cuda_available:
        torch.cuda.manual_seed(42)

    # ==================== 載入真實數據 ====================
    logger.info("\n" + "=" * 60)
    logger.info("載入 SemEval-2014 真實數據")
    logger.info("=" * 60)

    preprocessor = SemEvalPreprocessor()

    # 載入數據
    if args.domain == 'restaurant':
        train_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Train_v2.xml'
        test_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Test_Data_phaseB.xml'
    else:
        train_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Laptop_Train_v2.xml'
        test_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Laptops_Test_Data_phaseB.xml'

    logger.info(f"訓練檔案: {train_file}")
    logger.info(f"測試檔案: {test_file}")

    # 解析 XML
    train_samples = preprocessor.parse_semeval_xml(str(train_file))
    test_samples = preprocessor.parse_semeval_xml(str(test_file))

    # 處理樣本（不需要構建詞彙表，因為使用 BERT）
    train_df = preprocessor.process_samples(train_samples, build_vocab=True)
    test_df = preprocessor.process_samples(test_samples, build_vocab=False)

    # 劃分訓練/驗證集
    train_df, val_df = split_train_val(train_df, val_ratio=0.15, stratify=True)

    logger.info(f"\n訓練集: {len(train_df)} 樣本")
    logger.info(f"驗證集: {len(val_df)} 樣本")
    logger.info(f"測試集: {len(test_df)} 樣本")

    # 打印統計
    preprocessor.print_statistics(train_df)

    # ==================== 創建 BERT DataLoader ====================
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_dataset = BERTDataset(train_df, tokenizer)
    val_dataset = BERTDataset(val_df, tokenizer)

    # 設定 DataLoader 參數
    # Windows 上 num_workers=0 通常更快（避免多進程開銷）
    # Linux 可改為 num_workers=4
    use_pin_memory = torch.cuda.is_available()
    num_workers = 0  # Windows 最佳設定

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn_bert, num_workers=num_workers,
        pin_memory=use_pin_memory, persistent_workers=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn_bert, num_workers=num_workers,
        pin_memory=use_pin_memory, persistent_workers=False
    )

    # ==================== 創建模型 ====================
    logger.info("\n" + "=" * 60)
    logger.info("創建 HMAC-Net with BERT 模型")
    logger.info("=" * 60)

    model = HMACNetBERT(
        bert_model_name=args.bert_model,
        freeze_bert=args.freeze_bert,
        hidden_dim=256,
        num_classes=3,
        dropout=0.3,
        use_iarm=True,
        relation_type='transformer'
    ).to(device)

    model.print_model_summary()

    # ==================== 訓練 ====================
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(
        model, train_loader, val_loader, optimizer, criterion,
        device, logger, config, project_root=project_root
    )
    trainer.train(num_epochs=args.epochs)

    logger.close()


if __name__ == "__main__":
    main()
