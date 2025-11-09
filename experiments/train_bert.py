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
from models import AAHAEnhanced, PMACEnhanced, IARMEnhanced
from models.base_model import BaseModel, MLP
from utils import (
    get_logger,
    MetricsCalculator,
    RunningMetrics,
    SemEvalPreprocessor,
    split_train_val,
    TrainingVisualizer,
    MetricsVisualizer,
    AttentionVisualizer,
    get_loss_function,
    get_scheduler,
    EmbeddingMixup,
    AdversarialTraining
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

        # AAHA 模組（使用增強版）
        self.aaha = AAHAEnhanced(
            hidden_dim=hidden_dim,
            aspect_dim=hidden_dim,
            word_attention_dims=[64, 128],  # 多尺度注意力
            phrase_attention_dims=[64, 128, 256],
            sentence_attention_dims=[64, 128, 256],
            attention_dropout=0.1,  # 注意力專用 dropout
            output_dropout=dropout
        )

        # PMAC 模組（使用增強版）
        self.pmac = PMACEnhanced(
            input_dim=hidden_dim,
            fusion_dim=fusion_dim,
            num_composition_layers=num_composition_layers,
            hidden_dim=128,  # 門控網絡隱藏層維度
            dropout=dropout,
            use_aspect_bn=True,  # 使用 aspect-specific BN
            num_aspects=3,  # 最多3個aspect類別
            progressive_training=False  # 可選：漸進式訓練
        )

        # IARM 模組（增強版，可選）
        self.use_iarm = use_iarm
        if use_iarm:
            self.iarm = IARMEnhanced(
                input_dim=fusion_dim,
                relation_dim=relation_dim,
                num_heads=num_heads,
                num_layers=2,
                dropout=dropout,
                use_edge_features=True,
                use_relation_pooling=True,
                pooling_heads=4
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

        # IARM 模組（增強版）
        iarm_info = None
        if self.use_iarm:
            aspect_repr = composed_repr.unsqueeze(1)
            enhanced_repr, iarm_info = self.iarm(aspect_repr)
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

    def __init__(self, model, train_loader, val_loader, test_loader, optimizer, criterion, device, logger, config, project_root=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
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
        scheduler_config = config['training']['scheduler']
        scheduler_type = scheduler_config['type']

        if scheduler_type == 'cosine':
            # Cosine Annealing with Warmup
            num_epochs = config['training']['num_epochs']
            self.scheduler = get_scheduler(
                optimizer,
                scheduler_type='cosine',
                warmup_epochs=scheduler_config.get('warmup_epochs', 2),
                max_epochs=num_epochs,
                min_lr=scheduler_config.get('min_lr', 1e-6)
            )
            self.scheduler_needs_metric = False
        elif scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max',
                factor=scheduler_config['factor'],
                patience=scheduler_config['patience'],
                min_lr=scheduler_config['min_lr']
            )
            self.scheduler_needs_metric = True
        else:
            self.scheduler = None
            self.scheduler_needs_metric = False

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

        # 資料增強
        aug_config = config['training']['augmentation']
        self.mixup = EmbeddingMixup(
            alpha=aug_config.get('mixup_alpha', 0.2),
            enabled=aug_config.get('mixup_enabled', False)
        )

        # 對抗訓練
        adv_config = aug_config.get('adversarial_training', {})
        if adv_config.get('enabled', False):
            self.adv_training = AdversarialTraining(
                model,
                epsilon=adv_config.get('epsilon', 1.0),
                method=adv_config.get('method', 'fgm')
            )
        else:
            self.adv_training = None

        # 注意力視覺化器
        self.attention_visualizer = AttentionVisualizer(
            save_dir=str(self.project_root / 'results' / 'visualizations')
        )

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
            if self.scheduler is not None:
                if self.scheduler_needs_metric:
                    # ReduceLROnPlateau 需要 metric
                    self.scheduler.step(val_metrics['macro_f1'])
                else:
                    # Cosine Annealing 只需要 step()
                    self.scheduler.step()

            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"當前學習率: {current_lr:.6e}")

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

        # 生成注意力熱圖視覺化
        self._generate_attention_heatmaps(num_samples=5)

        # 生成訓練報告 txt
        self._generate_training_report()

        # 在測試集上評估最佳模型
        if self.test_loader is not None:
            self._evaluate_on_test_set()

    @torch.no_grad()
    def _evaluate_on_test_set(self):
        """在測試集上評估最佳模型"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("在測試集上評估最佳模型")
        self.logger.info("=" * 60)

        self.model.eval()
        all_preds, all_labels = [], []

        for batch in tqdm(self.test_loader, desc='測試集評估'):
            text_ids = batch['text_input_ids'].to(self.device)
            text_mask = batch['text_attention_mask'].to(self.device)
            aspect_ids = batch['aspect_input_ids'].to(self.device)
            aspect_mask = batch['aspect_attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            logits, _ = self.model(text_ids, text_mask, aspect_ids, aspect_mask)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # 計算測試集指標
        test_metrics = self.metrics_calculator.calculate_all(
            np.array(all_labels), np.array(all_preds)
        )

        self.logger.info(f"測試集準確率: {test_metrics['accuracy']:.4f}")
        self.logger.info(f"測試集 Macro F1: {test_metrics['macro_f1']:.4f}")
        self.logger.info(f"測試集 Macro Precision: {test_metrics['macro_precision']:.4f}")
        self.logger.info(f"測試集 Macro Recall: {test_metrics['macro_recall']:.4f}")

        # 保存測試集結果
        self.test_labels = all_labels
        self.test_preds = all_preds
        self.test_metrics = test_metrics

        self.logger.info("=" * 60)

        # 生成測試集報告
        self._generate_test_report()

    def _generate_test_report(self):
        """生成測試集評估報告"""
        report_path = self.project_root / 'results' / 'reports' / 'test_report.txt'
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("測試集評估報告\n")
            f.write("=" * 80 + "\n\n")

            f.write("【測試集整體指標】\n")
            f.write(f"  準確率 (Accuracy): {self.test_metrics['accuracy']:.4f}\n")
            f.write(f"  Macro F1: {self.test_metrics['macro_f1']:.4f}\n")
            f.write(f"  Macro Precision: {self.test_metrics['macro_precision']:.4f}\n")
            f.write(f"  Macro Recall: {self.test_metrics['macro_recall']:.4f}\n")
            f.write(f"\n")

            # 各類別指標
            f.write("【各類別表現】\n")
            class_names = ['負面', '中性', '正面']
            for class_name in class_names:
                precision = self.test_metrics.get(f'{class_name}_precision', 0.0)
                recall = self.test_metrics.get(f'{class_name}_recall', 0.0)
                f1 = self.test_metrics.get(f'{class_name}_f1', 0.0)
                support = self.test_metrics.get(f'{class_name}_support', 0)
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {precision:.4f}\n")
                f.write(f"    Recall: {recall:.4f}\n")
                f.write(f"    F1-Score: {f1:.4f}\n")
                f.write(f"    Support: {int(support)}\n")
            f.write(f"\n")

            # 混淆矩陣
            cm = self.metrics_calculator.get_confusion_matrix(
                self.test_labels, self.test_preds
            )
            f.write("【混淆矩陣】\n")
            f.write("         預測→   負面   中性   正面\n")
            f.write("  實際↓\n")
            for i, class_name in enumerate(class_names):
                f.write(f"  {class_name:4s}         {cm[i][0]:5d}  {cm[i][1]:5d}  {cm[i][2]:5d}\n")
            f.write(f"\n")

            # 中性類別識別分析
            f.write("【中性類別識別分析】\n")
            neutral_idx = 1  # 中性類別索引
            neutral_total = cm[neutral_idx].sum()
            neutral_correct = cm[neutral_idx][neutral_idx]
            neutral_to_neg = cm[neutral_idx][0]
            neutral_to_pos = cm[neutral_idx][2]

            if neutral_total > 0:
                neutral_acc = neutral_correct / neutral_total
                neutral_to_neg_rate = neutral_to_neg / neutral_total
                neutral_to_pos_rate = neutral_to_pos / neutral_total

                f.write(f"  中性類別準確率: {neutral_acc:.2%}\n")
                f.write(f"  中性→負面 誤判率: {neutral_to_neg_rate:.2%} ({neutral_to_neg}/{neutral_total})\n")
                f.write(f"  中性→正面 誤判率: {neutral_to_pos_rate:.2%} ({neutral_to_pos}/{neutral_total})\n")

                if neutral_to_neg_rate > 0.20:
                    f.write(f"  ⚠️ 警告: 中性類別傾向被誤判為負面\n")
                if neutral_to_pos_rate > 0.20:
                    f.write(f"  ⚠️ 警告: 中性類別傾向被誤判為正面\n")
                if neutral_acc < 0.65:
                    f.write(f"  ⚠️ 建議: 中性類別識別能力不足，考慮增加 class_weights[1]\n")
            f.write(f"\n")

            f.write("=" * 80 + "\n")
            f.write("報告生成時間: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")

        self.logger.info(f"[OK] 測試集報告已保存: {report_path}")

        # 繪製測試集混淆矩陣
        viz_dir = str(self.project_root / 'results' / 'visualizations')
        viz = MetricsVisualizer(save_dir=viz_dir)
        viz.plot_confusion_matrix(
            cm, class_names,
            normalize=True,
            title='測試集混淆矩陣',
            save_name='confusion_matrix_test.png'
        )
        self.logger.info("[OK] 測試集混淆矩陣已保存")

    def _generate_attention_heatmaps(self, num_samples=5):
        """
        生成注意力熱圖視覺化

        參數:
            num_samples: 要視覺化的樣本數量
        """
        self.logger.info(f"[*] 生成注意力熱圖視覺化 ({num_samples} 個樣本)...")

        self.model.eval()
        tokenizer = BertTokenizer.from_pretrained(
            self.config.get('model', {}).get('bert_model_name', 'bert-base-uncased')
        )

        # 收集樣本
        samples_collected = 0
        sentiment_labels = ['負面', '中性', '正面']

        with torch.no_grad():
            for batch in self.val_loader:
                if samples_collected >= num_samples:
                    break

                text_ids = batch['text_ids'].to(self.device)
                aspect_ids = batch['aspect_ids'].to(self.device)
                text_mask = batch['text_mask'].to(self.device)
                labels = batch['label']

                # 前向傳播並獲取注意力權重
                logits, attention_dict = self.model(
                    text_ids, aspect_ids, text_mask,
                    return_attention=True
                )

                # 對每個樣本生成熱圖
                batch_size = text_ids.size(0)
                for i in range(min(batch_size, num_samples - samples_collected)):
                    # 獲取詞彙
                    tokens = tokenizer.convert_ids_to_tokens(text_ids[i].cpu().numpy())
                    # 移除 padding
                    valid_len = text_mask[i].sum().item()
                    tokens = tokens[:valid_len]

                    # 獲取 aspect 詞彙
                    aspect_tokens = tokenizer.convert_ids_to_tokens(aspect_ids[i].cpu().numpy())
                    aspect_text = ' '.join([t for t in aspect_tokens if t not in ['[PAD]', '[CLS]', '[SEP]']])

                    # 獲取真實標籤和預測標籤
                    true_label = sentiment_labels[labels[i].item()]
                    pred_label = sentiment_labels[torch.argmax(logits[i]).item()]

                    # 獲取 AAHA 注意力權重
                    if attention_dict and 'aaha' in attention_dict and attention_dict['aaha'] is not None:
                        aaha_attn = attention_dict['aaha']

                        # 檢查是否有多尺度注意力
                        if isinstance(aaha_attn, dict):
                            # 詞級注意力
                            if 'word' in aaha_attn:
                                word_attn = aaha_attn['word'][i].cpu().numpy()[:valid_len]

                                # 繪製詞級注意力熱圖
                                self.attention_visualizer.plot_attention(
                                    attention_weights=word_attn,
                                    words=tokens,
                                    aspect_words=aspect_tokens,
                                    title=f'詞級注意力 - Aspect: {aspect_text}\n真實: {true_label}, 預測: {pred_label}',
                                    save_name=f'attention_word_sample_{samples_collected+1}.png',
                                    figsize=(14, 3)
                                )

                            # 片語級注意力
                            if 'phrase' in aaha_attn:
                                phrase_attn = aaha_attn['phrase'][i].cpu().numpy()[:valid_len]

                                self.attention_visualizer.plot_attention(
                                    attention_weights=phrase_attn,
                                    words=tokens,
                                    aspect_words=aspect_tokens,
                                    title=f'片語級注意力 - Aspect: {aspect_text}\n真實: {true_label}, 預測: {pred_label}',
                                    save_name=f'attention_phrase_sample_{samples_collected+1}.png',
                                    figsize=(14, 3)
                                )

                            # 句子級注意力
                            if 'sentence' in aaha_attn:
                                sentence_attn = aaha_attn['sentence'][i].cpu().numpy()[:valid_len]

                                self.attention_visualizer.plot_attention(
                                    attention_weights=sentence_attn,
                                    words=tokens,
                                    aspect_words=aspect_tokens,
                                    title=f'句子級注意力 - Aspect: {aspect_text}\n真實: {true_label}, 預測: {pred_label}',
                                    save_name=f'attention_sentence_sample_{samples_collected+1}.png',
                                    figsize=(14, 3)
                                )

                            # 如果有所有三個層級，生成階層式注意力視覺化
                            if 'word' in aaha_attn and 'phrase' in aaha_attn and 'sentence' in aaha_attn:
                                word_attn = aaha_attn['word'][i].cpu().numpy()[:valid_len]
                                phrase_attn = aaha_attn['phrase'][i].cpu().numpy()[:valid_len]
                                sentence_attn = aaha_attn['sentence'][i].cpu().numpy()[:valid_len]

                                self.attention_visualizer.plot_hierarchical_attention(
                                    word_attention=word_attn,
                                    phrase_attention=phrase_attn,
                                    sentence_attention=sentence_attn,
                                    words=tokens,
                                    aspect=aspect_text,
                                    save_name=f'attention_hierarchical_sample_{samples_collected+1}.png'
                                )
                        else:
                            # 單一注意力權重
                            attn = aaha_attn[i].cpu().numpy()[:valid_len]

                            self.attention_visualizer.plot_attention(
                                attention_weights=attn,
                                words=tokens,
                                aspect_words=aspect_tokens,
                                title=f'注意力權重 - Aspect: {aspect_text}\n真實: {true_label}, 預測: {pred_label}',
                                save_name=f'attention_sample_{samples_collected+1}.png',
                                figsize=(14, 3)
                            )

                    samples_collected += 1
                    if samples_collected >= num_samples:
                        break

        self.logger.info(f"[OK] 已生成 {samples_collected} 個樣本的注意力熱圖")

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

            # 過擬合檢測
            if len(self.history['train_macro_f1']) > 0 and len(self.history['val_macro_f1']) > 0:
                final_train_f1 = self.history['train_macro_f1'][-1]
                final_val_f1 = self.history['val_macro_f1'][-1]
                overfitting_gap = final_train_f1 - final_val_f1
                f.write(f"  訓練/驗證 F1 差距: {overfitting_gap:.4f}")
                if overfitting_gap > 0.15:
                    f.write(f" ⚠️ 過擬合警告\n")
                elif overfitting_gap > 0.10:
                    f.write(f" ⚠ 輕微過擬合\n")
                else:
                    f.write(f" ✓ 良好泛化\n")
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

                # 中性類別識別分析
                f.write("【中性類別識別分析】\n")
                neutral_idx = 1  # 中性類別索引
                neutral_total = cm[neutral_idx].sum()
                neutral_correct = cm[neutral_idx][neutral_idx]
                neutral_to_neg = cm[neutral_idx][0]
                neutral_to_pos = cm[neutral_idx][2]

                if neutral_total > 0:
                    neutral_acc = neutral_correct / neutral_total
                    neutral_to_neg_rate = neutral_to_neg / neutral_total
                    neutral_to_pos_rate = neutral_to_pos / neutral_total

                    f.write(f"  中性類別準確率: {neutral_acc:.2%}\n")
                    f.write(f"  中性→負面 誤判率: {neutral_to_neg_rate:.2%} ({neutral_to_neg}/{neutral_total})\n")
                    f.write(f"  中性→正面 誤判率: {neutral_to_pos_rate:.2%} ({neutral_to_pos}/{neutral_total})\n")

                    if neutral_to_neg_rate > 0.20:
                        f.write(f"  ⚠️ 警告: 中性類別傾向被誤判為負面\n")
                    if neutral_to_pos_rate > 0.20:
                        f.write(f"  ⚠️ 警告: 中性類別傾向被誤判為正面\n")
                    if neutral_acc < 0.65:
                        f.write(f"  ⚠️ 建議: 中性類別識別能力不足，考慮增加 class_weights[1]\n")
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

            # 視覺化檔案說明
            f.write("\n【生成的視覺化檔案】\n")
            f.write("  1. 訓練曲線: results/visualizations/hmac_net_training_curves.png\n")
            f.write("  2. 混淆矩陣: results/visualizations/confusion_matrix_best_model.png\n")
            f.write("  3. 注意力熱圖: results/visualizations/attention_*.png\n")
            f.write("     - attention_word_sample_*.png: 詞級注意力\n")
            f.write("     - attention_phrase_sample_*.png: 片語級注意力\n")
            f.write("     - attention_sentence_sample_*.png: 句子級注意力\n")
            f.write("     - attention_hierarchical_sample_*.png: 階層式注意力（完整）\n")
            f.write("\n" + "=" * 80 + "\n")

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

    # 載入數據（使用帶標籤的 Gold 測試集）
    if args.domain == 'restaurant':
        train_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Train_v2.xml'
        test_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Restaurants_Test_Gold.xml'
    else:
        train_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Laptop_Train_v2.xml'
        test_file = project_root / 'data' / 'raw' / 'semeval2014' / 'Laptops_Test_Gold.xml'

    logger.info(f"訓練檔案: {train_file}")
    logger.info(f"測試檔案: {test_file} (Gold 標準)")

    # 解析 XML
    train_samples = preprocessor.parse_semeval_xml(str(train_file))
    test_samples = preprocessor.parse_semeval_xml(str(test_file))

    # 處理樣本（不需要構建詞彙表，因為使用 BERT）
    train_df = preprocessor.process_samples(train_samples, build_vocab=True)
    test_df = preprocessor.process_samples(test_samples, build_vocab=False)

    # 從訓練集劃分出驗證集（85% 訓練，15% 驗證）
    train_df, val_df = split_train_val(train_df, val_ratio=0.15, stratify=True)

    logger.info(f"\n數據集劃分:")
    logger.info(f"  訓練集: {len(train_df)} 樣本")
    logger.info(f"  驗證集: {len(val_df)} 樣本 (從訓練集劃分)")
    logger.info(f"  測試集: {len(test_df)} 樣本 (官方 Gold 標準)")
    logger.info(f"  總計: {len(train_df) + len(val_df) + len(test_df)} 樣本")

    # 打印統計
    preprocessor.print_statistics(train_df)

    # ==================== 創建 BERT DataLoader ====================
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_dataset = BERTDataset(train_df, tokenizer)
    val_dataset = BERTDataset(val_df, tokenizer)
    test_dataset = BERTDataset(test_df, tokenizer)

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

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
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
        dropout=0.5,  # 增加 dropout 以防止過擬合
        use_iarm=True,
        relation_type='transformer'
    ).to(device)

    model.print_model_summary()

    # ==================== 訓練 ====================
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 創建損失函數（Focal Loss + 類別權重 + Label Smoothing）
    loss_config = config['training']['loss']
    loss_type = loss_config['type']
    gamma = loss_config.get('gamma', 2.0)
    class_weights = loss_config.get('class_weights', None) if loss_config.get('use_class_weights', False) else None
    label_smoothing = config['training']['augmentation']['label_smoothing']

    logger.info("\n損失函數配置:")
    logger.info(f"  類型: {loss_type}")
    logger.info(f"  Gamma: {gamma}")
    logger.info(f"  類別權重: {class_weights}")
    logger.info(f"  Label Smoothing: {label_smoothing}")

    criterion = get_loss_function(
        loss_type=loss_type,
        num_classes=3,
        class_weights=class_weights,
        gamma=gamma,
        label_smoothing=label_smoothing
    )

    trainer = Trainer(
        model, train_loader, val_loader, test_loader, optimizer, criterion,
        device, logger, config, project_root=project_root
    )
    trainer.train(num_epochs=args.epochs)

    logger.close()


if __name__ == "__main__":
    main()
