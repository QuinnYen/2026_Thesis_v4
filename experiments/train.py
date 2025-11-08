"""
HMAC-Net 訓練腳本

功能:
    - 載入和預處理數據
    - 訓練 HMAC-Net 模型
    - 支援 Early Stopping
    - 保存最佳模型和訓練曲線
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

# 添加專案路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import HMACNet
from utils import (
    get_logger,
    MetricsCalculator,
    RunningMetrics,
    SemEvalPreprocessor,
    load_semeval_2014,
    split_train_val,
    create_data_loader,
    load_glove_embeddings,
    TrainingVisualizer
)


class Trainer:
    """
    HMAC-Net 訓練器

    功能:
        - 訓練循環
        - 驗證
        - Early Stopping
        - 模型保存
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        device,
        logger,
        config: dict
    ):
        """初始化訓練器"""
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.config = config

        # Early Stopping
        self.patience = config['training']['early_stopping']['patience']
        self.best_val_f1 = 0.0
        self.patience_counter = 0

        # 學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            min_lr=config['training']['scheduler']['min_lr']
        )

        # 指標計算器
        self.metrics_calculator = MetricsCalculator(num_classes=3)

        # 訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': [],
            'train_macro_f1': [],
            'val_macro_f1': []
        }

    def train_epoch(self, epoch: int) -> dict:
        """訓練一個 epoch"""
        self.model.train()
        running_metrics = RunningMetrics()

        all_preds = []
        all_labels = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [訓練]')

        for batch in pbar:
            # 移動到設備
            text_indices = batch['text_indices'].to(self.device)
            aspect_indices = batch['aspect_indices'].to(self.device)
            text_len = batch['text_len'].to(self.device)
            aspect_len = batch['aspect_len'].to(self.device)
            labels = batch['labels'].to(self.device)

            # 前向傳播
            logits, _ = self.model(
                text_indices=text_indices,
                aspect_indices=aspect_indices,
                text_len=text_len,
                aspect_len=aspect_len
            )

            # 計算損失
            loss = self.criterion(logits, labels)

            # 反向傳播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['grad_clip']
            )

            self.optimizer.step()

            # 記錄指標
            running_metrics.update(loss=loss.item())

            # 收集預測和標籤
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新進度條
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 計算 epoch 指標
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = self.metrics_calculator.calculate_all(all_labels, all_preds)

        epoch_metrics = {
            'loss': running_metrics.get_average('loss'),
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1']
        }

        return epoch_metrics

    def validate(self, epoch: int) -> dict:
        """驗證"""
        self.model.eval()
        running_metrics = RunningMetrics()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch} [驗證]')

            for batch in pbar:
                # 移動到設備
                text_indices = batch['text_indices'].to(self.device)
                aspect_indices = batch['aspect_indices'].to(self.device)
                text_len = batch['text_len'].to(self.device)
                aspect_len = batch['aspect_len'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 前向傳播
                logits, _ = self.model(
                    text_indices=text_indices,
                    aspect_indices=aspect_indices,
                    text_len=text_len,
                    aspect_len=aspect_len
                )

                # 計算損失
                loss = self.criterion(logits, labels)

                # 記錄指標
                running_metrics.update(loss=loss.item())

                # 收集預測和標籤
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 更新進度條
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # 計算 epoch 指標
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = self.metrics_calculator.calculate_all(all_labels, all_preds)

        epoch_metrics = {
            'loss': running_metrics.get_average('loss'),
            'accuracy': metrics['accuracy'],
            'macro_f1': metrics['macro_f1']
        }

        return epoch_metrics

    def train(self, num_epochs: int):
        """完整訓練流程"""
        self.logger.info("開始訓練...")
        self.logger.info(f"訓練輪數: {num_epochs}")

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
                f"訓練損失: {train_metrics['loss']:.4f}, "
                f"訓練 F1: {train_metrics['macro_f1']:.4f}, "
                f"驗證損失: {val_metrics['loss']:.4f}, "
                f"驗證 F1: {val_metrics['macro_f1']:.4f}"
            )

            # 學習率調整
            self.scheduler.step(val_metrics['macro_f1'])
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"當前學習率: {current_lr}")

            # 保存最佳模型
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.patience_counter = 0

                # 保存模型
                save_path = Path(self.config['checkpoint']['save_dir']) / \
                            f"hmac_net_best_f1_{self.best_val_f1:.4f}.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)

                self.model.save_model(
                    str(save_path),
                    epoch=epoch,
                    val_f1=self.best_val_f1,
                    config=self.config
                )

                self.logger.info(f"保存最佳模型: {save_path}")

            else:
                self.patience_counter += 1
                self.logger.info(
                    f"驗證 F1 沒有提升 ({self.patience_counter}/{self.patience})"
                )

            # Early Stopping
            if self.patience_counter >= self.patience:
                self.logger.info("Early Stopping 觸發，停止訓練")
                break

        self.logger.info(f"訓練完成！最佳驗證 F1: {self.best_val_f1:.4f}")

        # 繪製訓練曲線
        self.plot_training_curves()

    def plot_training_curves(self):
        """繪製訓練曲線"""
        visualizer = TrainingVisualizer(save_dir='results/visualizations')
        visualizer.plot_training_curves(
            self.history,
            save_name='hmac_net_training_curves.png'
        )


def main():
    # 解析命令列參數
    parser = argparse.ArgumentParser(description='訓練 HMAC-Net')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml',
                        help='實驗配置檔案路徑')
    parser.add_argument('--model_config', type=str, default='configs/model_config.yaml',
                        help='模型配置檔案路徑')
    parser.add_argument('--data_config', type=str, default='configs/data_config.yaml',
                        help='數據配置檔案路徑')
    args = parser.parse_args()

    # 載入配置
    with open(args.config, 'r', encoding='utf-8') as f:
        exp_config = yaml.safe_load(f)

    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)

    with open(args.data_config, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    # 初始化日誌
    logger = get_logger(
        name='HMAC-Net-Training',
        log_dir=exp_config['logging']['log_dir'],
        use_tensorboard=exp_config['logging']['tensorboard']
    )

    logger.info("=" * 60)
    logger.info("HMAC-Net 訓練腳本")
    logger.info("=" * 60)

    # 設定設備
    device = torch.device('cuda' if torch.cuda.is_available() and model_config['device']['use_cuda'] else 'cpu')
    logger.info(f"使用設備: {device}")

    # 設定隨機種子
    seed = model_config['device']['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # 載入數據（這裡使用模擬數據，實際使用時替換為 SemEval 數據）
    logger.info("載入數據...")

    # TODO: 替換為實際的數據載入
    # preprocessor = SemEvalPreprocessor()
    # train_df, test_df = load_semeval_2014(
    #     data_dir='data/raw/semeval2014',
    #     domain='restaurant',
    #     preprocessor=preprocessor
    # )
    # train_df, val_df = split_train_val(train_df, val_ratio=0.2)

    # 暫時使用模擬數據進行示範
    logger.warning("使用模擬數據進行示範，請替換為實際數據！")
    import pandas as pd

    # 模擬數據
    dummy_data = []
    for i in range(100):
        dummy_data.append({
            'text_indices': [i % 1000 for _ in range(50)],
            'aspect_indices': [i % 1000 for _ in range(5)],
            'aspect_mask': [1 if j < 5 else 0 for j in range(50)],
            'label': i % 3
        })

    train_df = pd.DataFrame(dummy_data[:70])
    val_df = pd.DataFrame(dummy_data[70:])

    # 創建 DataLoader
    train_loader = create_data_loader(
        train_df,
        batch_size=exp_config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = create_data_loader(
        val_df,
        batch_size=exp_config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    logger.info(f"訓練集大小: {len(train_df)}")
    logger.info(f"驗證集大小: {len(val_df)}")

    # 創建模型
    logger.info("創建 HMAC-Net 模型...")

    model = HMACNet(
        vocab_size=1000,  # TODO: 從詞彙表獲取
        embedding_dim=model_config['model']['embedding_dim'],
        hidden_dim=model_config['model']['hidden_dim'],
        num_classes=model_config['classifier']['num_classes'],
        num_layers=model_config['model']['num_layers'],
        dropout=model_config['model']['dropout'],
        word_attention_dim=model_config['aaha']['word_attention_dim'],
        phrase_attention_dim=model_config['aaha']['phrase_attention_dim'],
        sentence_attention_dim=model_config['aaha']['sentence_attention_dim'],
        fusion_dim=model_config['pmac']['fusion_dim'],
        num_composition_layers=model_config['pmac']['composition_layers'],
        fusion_method=model_config['pmac']['fusion_method'],
        use_iarm=True,
        relation_type=model_config['iarm']['relation_type'],
        num_heads=model_config['iarm']['num_heads'],
        classifier_hidden_dims=model_config['classifier']['hidden_dims'],
        use_batch_norm=model_config['classifier']['use_batch_norm']
    ).to(device)

    model.print_model_summary()

    # 創建優化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=exp_config['training']['learning_rate'],
        weight_decay=exp_config['training']['weight_decay']
    )

    # 損失函數
    criterion = nn.CrossEntropyLoss()

    # 創建訓練器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        config=exp_config
    )

    # 開始訓練
    trainer.train(num_epochs=exp_config['training']['num_epochs'])

    logger.close()


if __name__ == "__main__":
    main()
