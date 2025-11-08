"""
評估指標模組
提供情感分析任務的各種評估指標計算
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from scipy import stats


class MetricsCalculator:
    """
    指標計算器

    功能:
        - 計算準確率、精確率、召回率、F1 分數
        - 生成混淆矩陣
        - 支援多類別和二元分類
        - 提供詳細的分類報告
    """

    def __init__(self, num_classes: int = 3, class_names: Optional[List[str]] = None):
        """
        初始化指標計算器

        參數:
            num_classes: 類別數量
            class_names: 類別名稱列表（用於報告）
        """
        self.num_classes = num_classes

        if class_names is None:
            if num_classes == 3:
                self.class_names = ['負面', '中性', '正面']
            elif num_classes == 2:
                self.class_names = ['負面', '正面']
            else:
                self.class_names = [f'類別{i}' for i in range(num_classes)]
        else:
            self.class_names = class_names

    def calculate_all(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        計算所有指標

        參數:
            y_true: 真實標籤 [N]
            y_pred: 預測標籤 [N]

        返回:
            包含所有指標的字典
        """
        # 確保輸入是 numpy 陣列
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # 計算基本指標
        accuracy = accuracy_score(y_true, y_pred)

        # 計算每個類別的精確率、召回率、F1
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # 計算宏平均和微平均
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )

        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='micro', zero_division=0
        )

        # 計算加權平均
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # 組織結果
        metrics = {
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
        }

        # 添加每個類別的指標
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name}_precision'] = precision[i] if i < len(precision) else 0.0
            metrics[f'{class_name}_recall'] = recall[i] if i < len(recall) else 0.0
            metrics[f'{class_name}_f1'] = f1[i] if i < len(f1) else 0.0
            metrics[f'{class_name}_support'] = support[i] if i < len(support) else 0

        return metrics

    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        計算混淆矩陣

        參數:
            y_true: 真實標籤
            y_pred: 預測標籤
            normalize: 正規化方式 ('true', 'pred', 'all', None)

        返回:
            混淆矩陣
        """
        cm = confusion_matrix(y_true, y_pred)

        if normalize == 'true':
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        elif normalize == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0, keepdims=True)
        elif normalize == 'all':
            cm = cm.astype('float') / cm.sum()

        return cm

    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        生成分類報告

        參數:
            y_true: 真實標籤
            y_pred: 預測標籤

        返回:
            分類報告字串
        """
        return classification_report(
            y_true,
            y_pred,
            target_names=self.class_names,
            zero_division=0
        )

    def format_metrics(self, metrics: Dict[str, float]) -> str:
        """
        格式化指標輸出

        參數:
            metrics: 指標字典

        返回:
            格式化的字串
        """
        lines = []
        lines.append("=" * 60)
        lines.append("評估指標")
        lines.append("=" * 60)

        # 整體指標
        lines.append(f"準確率 (Accuracy):         {metrics['accuracy']:.4f}")
        lines.append(f"宏平均 F1 (Macro F1):      {metrics['macro_f1']:.4f}")
        lines.append(f"微平均 F1 (Micro F1):      {metrics['micro_f1']:.4f}")
        lines.append(f"加權平均 F1 (Weighted F1): {metrics['weighted_f1']:.4f}")
        lines.append("-" * 60)

        # 每個類別的指標
        lines.append("各類別指標:")
        for class_name in self.class_names:
            p = metrics.get(f'{class_name}_precision', 0)
            r = metrics.get(f'{class_name}_recall', 0)
            f = metrics.get(f'{class_name}_f1', 0)
            s = metrics.get(f'{class_name}_support', 0)

            lines.append(f"  {class_name}:")
            lines.append(f"    精確率: {p:.4f} | 召回率: {r:.4f} | F1: {f:.4f} | 樣本數: {int(s)}")

        lines.append("=" * 60)
        return "\n".join(lines)


class StatisticalTester:
    """
    統計顯著性檢驗器

    功能:
        - 成對 t 檢驗
        - Wilcoxon 符號秩檢驗
        - Bootstrap 置信區間
    """

    @staticmethod
    def paired_t_test(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        成對 t 檢驗

        參數:
            scores1: 第一組分數
            scores2: 第二組分數
            alpha: 顯著性水平

        返回:
            包含統計量和 p 值的字典
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)

        t_stat, p_value = stats.ttest_rel(scores1, scores2)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha
        }

    @staticmethod
    def wilcoxon_test(
        scores1: List[float],
        scores2: List[float],
        alpha: float = 0.05
    ) -> Dict[str, float]:
        """
        Wilcoxon 符號秩檢驗（非參數檢驗）

        參數:
            scores1: 第一組分數
            scores2: 第二組分數
            alpha: 顯著性水平

        返回:
            包含統計量和 p 值的字典
        """
        scores1 = np.array(scores1)
        scores2 = np.array(scores2)

        # 使用雙側檢驗
        statistic, p_value = stats.wilcoxon(scores1, scores2, alternative='two-sided')

        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < alpha,
            'alpha': alpha
        }

    @staticmethod
    def bootstrap_confidence_interval(
        scores: List[float],
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Bootstrap 置信區間

        參數:
            scores: 分數列表
            n_bootstrap: Bootstrap 樣本數
            confidence_level: 置信水平

        返回:
            包含均值和置信區間的字典
        """
        scores = np.array(scores)
        n = len(scores)

        # Bootstrap 重抽樣
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(np.mean(sample))

        bootstrap_means = np.array(bootstrap_means)

        # 計算置信區間
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)

        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }


class RunningMetrics:
    """
    運行時指標追蹤器
    用於在訓練過程中累積和計算平均指標
    """

    def __init__(self):
        """初始化運行時指標"""
        self.metrics = defaultdict(list)
        self.counts = defaultdict(int)

    def update(self, **kwargs):
        """
        更新指標

        參數:
            **kwargs: 指標名稱和值的鍵值對
        """
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)):
                self.metrics[key].extend(value)
                self.counts[key] += len(value)
            else:
                self.metrics[key].append(value)
                self.counts[key] += 1

    def get_average(self, key: str) -> float:
        """
        獲取某個指標的平均值

        參數:
            key: 指標名稱

        返回:
            平均值
        """
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return sum(self.metrics[key]) / len(self.metrics[key])

    def get_all_averages(self) -> Dict[str, float]:
        """
        獲取所有指標的平均值

        返回:
            包含所有平均值的字典
        """
        return {
            key: self.get_average(key)
            for key in self.metrics.keys()
        }

    def reset(self):
        """重置所有指標"""
        self.metrics.clear()
        self.counts.clear()

    def __str__(self) -> str:
        """字串表示"""
        avg_metrics = self.get_all_averages()
        return " | ".join([
            f"{key}: {value:.4f}"
            for key, value in avg_metrics.items()
        ])


def compare_models(
    model_scores: Dict[str, List[float]],
    reference_model: str,
    alpha: float = 0.05
) -> Dict[str, Dict]:
    """
    比較多個模型的性能

    參數:
        model_scores: 模型名稱到分數列表的映射
        reference_model: 參考模型名稱
        alpha: 顯著性水平

    返回:
        比較結果字典
    """
    results = {}
    reference_scores = model_scores[reference_model]

    tester = StatisticalTester()

    for model_name, scores in model_scores.items():
        if model_name == reference_model:
            continue

        # 執行成對 t 檢驗
        t_test_result = tester.paired_t_test(scores, reference_scores, alpha)

        # 執行 Wilcoxon 檢驗
        wilcoxon_result = tester.wilcoxon_test(scores, reference_scores, alpha)

        # 計算平均差異
        mean_diff = np.mean(scores) - np.mean(reference_scores)

        results[model_name] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'mean_diff': mean_diff,
            't_test': t_test_result,
            'wilcoxon_test': wilcoxon_result
        }

    return results


if __name__ == "__main__":
    # 測試指標計算器
    print("測試指標計算器...")

    # 模擬預測結果
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    y_pred = np.array([0, 1, 2, 0, 2, 2, 0, 1, 1, 0])

    calculator = MetricsCalculator(num_classes=3)

    # 計算所有指標
    metrics = calculator.calculate_all(y_true, y_pred)
    print(calculator.format_metrics(metrics))

    # 測試混淆矩陣
    print("\n混淆矩陣:")
    cm = calculator.get_confusion_matrix(y_true, y_pred)
    print(cm)

    # 測試統計檢驗
    print("\n測試統計顯著性檢驗...")
    scores1 = [0.85, 0.87, 0.86, 0.88, 0.85]
    scores2 = [0.80, 0.82, 0.81, 0.83, 0.80]

    tester = StatisticalTester()
    t_result = tester.paired_t_test(scores1, scores2)
    print(f"成對 t 檢驗: p-value = {t_result['p_value']:.4f}, 顯著: {t_result['significant']}")

    bootstrap_result = tester.bootstrap_confidence_interval(scores1)
    print(f"Bootstrap 95% 置信區間: [{bootstrap_result['lower_bound']:.4f}, {bootstrap_result['upper_bound']:.4f}]")

    print("\n指標模組測試完成！")
