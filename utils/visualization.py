"""
視覺化模組
提供注意力視覺化、訓練曲線、混淆矩陣等圖表生成功能
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 設定中文字體（支援正體中文）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題


class AttentionVisualizer:
    """
    注意力權重視覺化器

    功能:
        - 繪製注意力熱圖
        - 支援多層注意力視覺化
        - 高亮顯示面向詞
    """

    def __init__(self, save_dir: str = "results/visualizations"):
        """
        初始化視覺化器

        參數:
            save_dir: 圖片保存目錄
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_attention(
        self,
        attention_weights: np.ndarray,
        words: List[str],
        aspect_words: List[str],
        title: str = "注意力權重",
        save_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """
        繪製注意力權重熱圖

        參數:
            attention_weights: 注意力權重 [seq_len] 或 [seq_len, seq_len]
            words: 詞列表
            aspect_words: 面向詞列表
            title: 圖表標題
            save_name: 保存檔案名稱
            figsize: 圖表大小
        """
        plt.figure(figsize=figsize)

        # 如果是一維注意力
        if attention_weights.ndim == 1:
            # 創建熱圖數據（轉為 2D）
            attention_2d = attention_weights.reshape(1, -1)

            # 繪製熱圖
            sns.heatmap(
                attention_2d,
                xticklabels=words,
                yticklabels=['注意力'],
                cmap='YlOrRd',
                cbar=True,
                fmt='.2f',
                square=False
            )

            # 高亮面向詞
            for i, word in enumerate(words):
                if word in aspect_words:
                    plt.gca().get_xticklabels()[i].set_color('red')
                    plt.gca().get_xticklabels()[i].set_weight('bold')

        # 如果是二維注意力（如 self-attention）
        else:
            sns.heatmap(
                attention_weights,
                xticklabels=words,
                yticklabels=words,
                cmap='YlOrRd',
                cbar=True,
                square=True
            )

        plt.title(title, fontsize=14, pad=15)
        plt.xlabel('詞', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # 保存圖片
        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"注意力圖已保存: {save_path}")

        plt.show()
        plt.close()

    def plot_hierarchical_attention(
        self,
        word_attention: np.ndarray,
        phrase_attention: np.ndarray,
        sentence_attention: np.ndarray,
        words: List[str],
        aspect: str,
        save_name: Optional[str] = None
    ):
        """
        繪製階層式注意力（AAHA 模組）

        參數:
            word_attention: 詞級注意力 [seq_len]
            phrase_attention: 片語級注意力 [seq_len]
            sentence_attention: 句子級注意力 [seq_len]
            words: 詞列表
            aspect: 面向詞
            save_name: 保存檔案名稱
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 8))

        # 詞級注意力
        sns.heatmap(
            word_attention.reshape(1, -1),
            xticklabels=words,
            yticklabels=['詞級'],
            cmap='Blues',
            cbar=True,
            ax=axes[0]
        )
        axes[0].set_title(f'詞級注意力 (面向: {aspect})', fontsize=12)

        # 片語級注意力
        sns.heatmap(
            phrase_attention.reshape(1, -1),
            xticklabels=words,
            yticklabels=['片語級'],
            cmap='Greens',
            cbar=True,
            ax=axes[1]
        )
        axes[1].set_title('片語級注意力', fontsize=12)

        # 句子級注意力
        sns.heatmap(
            sentence_attention.reshape(1, -1),
            xticklabels=words,
            yticklabels=['句子級'],
            cmap='Oranges',
            cbar=True,
            ax=axes[2]
        )
        axes[2].set_title('句子級注意力', fontsize=12)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"階層式注意力圖已保存: {save_path}")

        plt.show()
        plt.close()


class TrainingVisualizer:
    """
    訓練過程視覺化器

    功能:
        - 繪製訓練/驗證曲線
        - 損失和指標追蹤
        - 學習率變化
    """

    def __init__(self, save_dir: str = "results/visualizations"):
        """
        初始化視覺化器

        參數:
            save_dir: 圖片保存目錄
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_training_curves(
        self,
        history: Dict[str, List[float]],
        save_name: Optional[str] = None
    ):
        """
        繪製訓練曲線

        參數:
            history: 訓練歷史字典（包含 train_loss, val_loss, train_acc, val_acc 等）
            save_name: 保存檔案名稱
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 損失曲線
        if 'train_loss' in history and 'val_loss' in history:
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0].plot(epochs, history['train_loss'], 'b-', label='訓練損失', linewidth=2)
            axes[0].plot(epochs, history['val_loss'], 'r-', label='驗證損失', linewidth=2)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('損失', fontsize=12)
            axes[0].set_title('損失曲線', fontsize=14)
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3)

        # 準確率 / F1 曲線
        metric_key = 'train_macro_f1' if 'train_macro_f1' in history else 'train_accuracy'
        val_metric_key = 'val_macro_f1' if 'val_macro_f1' in history else 'val_accuracy'

        if metric_key in history and val_metric_key in history:
            epochs = range(1, len(history[metric_key]) + 1)
            metric_name = 'Macro F1' if 'f1' in metric_key else '準確率'

            axes[1].plot(epochs, history[metric_key], 'b-', label=f'訓練{metric_name}', linewidth=2)
            axes[1].plot(epochs, history[val_metric_key], 'r-', label=f'驗證{metric_name}', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel(metric_name, fontsize=12)
            axes[1].set_title(f'{metric_name}曲線', fontsize=14)
            axes[1].legend(fontsize=10)
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"訓練曲線已保存: {save_path}")

        plt.show()
        plt.close()

    def plot_learning_rate(
        self,
        lr_history: List[float],
        save_name: Optional[str] = None
    ):
        """
        繪製學習率變化曲線

        參數:
            lr_history: 學習率歷史
            save_name: 保存檔案名稱
        """
        plt.figure(figsize=(10, 5))

        epochs = range(1, len(lr_history) + 1)
        plt.plot(epochs, lr_history, 'g-', linewidth=2)

        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('學習率', fontsize=12)
        plt.title('學習率變化', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用對數刻度

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"學習率曲線已保存: {save_path}")

        plt.show()
        plt.close()


class MetricsVisualizer:
    """
    指標視覺化器

    功能:
        - 繪製混淆矩陣
        - 模型比較圖表
        - 消融實驗結果
    """

    def __init__(self, save_dir: str = "results/visualizations"):
        """
        初始化視覺化器

        參數:
            save_dir: 圖片保存目錄
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        normalize: bool = False,
        title: str = "混淆矩陣",
        save_name: Optional[str] = None
    ):
        """
        繪製混淆矩陣

        參數:
            cm: 混淆矩陣
            class_names: 類別名稱
            normalize: 是否正規化
            title: 標題
            save_name: 保存檔案名稱
        """
        plt.figure(figsize=(8, 6))

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            fmt = '.2f'
        else:
            fmt = 'd'

        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=True,
            square=True
        )

        plt.ylabel('真實標籤', fontsize=12)
        plt.xlabel('預測標籤', fontsize=12)
        plt.title(title, fontsize=14, pad=15)
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩陣已保存: {save_path}")

        plt.show()
        plt.close()

    def plot_model_comparison(
        self,
        model_scores: Dict[str, List[float]],
        metric_name: str = "Macro F1",
        title: str = "模型性能比較",
        save_name: Optional[str] = None
    ):
        """
        繪製模型比較圖（箱線圖）

        參數:
            model_scores: 模型名稱到分數列表的映射
            metric_name: 指標名稱
            title: 標題
            save_name: 保存檔案名稱
        """
        plt.figure(figsize=(10, 6))

        # 準備數據
        data = []
        labels = []
        for model_name, scores in model_scores.items():
            data.append(scores)
            labels.append(model_name)

        # 繪製箱線圖
        bp = plt.boxplot(
            data,
            labels=labels,
            patch_artist=True,
            notch=True,
            showmeans=True
        )

        # 設定顏色
        colors = plt.cm.Set3(range(len(data)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        plt.ylabel(metric_name, fontsize=12)
        plt.title(title, fontsize=14, pad=15)
        plt.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"模型比較圖已保存: {save_path}")

        plt.show()
        plt.close()

    def plot_ablation_study(
        self,
        variant_scores: Dict[str, float],
        title: str = "消融實驗結果",
        save_name: Optional[str] = None
    ):
        """
        繪製消融實驗結果（柱狀圖）

        參數:
            variant_scores: 變體名稱到分數的映射
            title: 標題
            save_name: 保存檔案名稱
        """
        plt.figure(figsize=(10, 6))

        variants = list(variant_scores.keys())
        scores = list(variant_scores.values())

        # 創建柱狀圖
        bars = plt.bar(range(len(variants)), scores, color=plt.cm.viridis(np.linspace(0, 1, len(variants))))

        # 添加數值標籤
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f'{score:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )

        plt.xticks(range(len(variants)), variants, rotation=45, ha='right')
        plt.ylabel('Macro F1', fontsize=12)
        plt.title(title, fontsize=14, pad=15)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        if save_name:
            save_path = self.save_dir / save_name
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"消融實驗圖已保存: {save_path}")

        plt.show()
        plt.close()


if __name__ == "__main__":
    # 測試視覺化功能
    print("測試視覺化模組...")

    # 測試注意力視覺化
    print("\n1. 測試注意力視覺化")
    visualizer = AttentionVisualizer()

    words = ['the', 'food', 'was', 'great', 'but', 'service', 'terrible']
    attention = np.array([0.05, 0.35, 0.05, 0.3, 0.05, 0.1, 0.1])
    visualizer.plot_attention(
        attention,
        words,
        ['food'],
        title="範例注意力權重",
        save_name="test_attention.png"
    )

    # 測試訓練曲線
    print("\n2. 測試訓練曲線")
    trainer_viz = TrainingVisualizer()

    history = {
        'train_loss': [1.0, 0.8, 0.6, 0.5, 0.4],
        'val_loss': [1.1, 0.9, 0.7, 0.6, 0.5],
        'train_macro_f1': [0.6, 0.7, 0.75, 0.8, 0.85],
        'val_macro_f1': [0.55, 0.65, 0.7, 0.75, 0.8]
    }
    trainer_viz.plot_training_curves(history, save_name="test_training_curves.png")

    # 測試混淆矩陣
    print("\n3. 測試混淆矩陣")
    metrics_viz = MetricsVisualizer()

    cm = np.array([[45, 3, 2], [5, 38, 7], [2, 6, 42]])
    metrics_viz.plot_confusion_matrix(
        cm,
        ['負面', '中性', '正面'],
        normalize=True,
        save_name="test_confusion_matrix.png"
    )

    print("\n視覺化模組測試完成！")
