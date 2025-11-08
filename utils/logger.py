"""
日誌記錄模組
提供統一的日誌記錄介面，支援控制台輸出、檔案記錄和 TensorBoard
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("警告: TensorBoard 不可用，請安裝 tensorboard 套件")


class Logger:
    """
    統一的日誌記錄器

    功能:
        - 控制台輸出（帶顏色）
        - 檔案記錄
        - TensorBoard 整合
        - 不同日誌級別支援
    """

    # ANSI 顏色代碼
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 綠色
        'WARNING': '\033[33m',  # 黃色
        'ERROR': '\033[31m',    # 紅色
        'CRITICAL': '\033[35m', # 紫色
        'RESET': '\033[0m'      # 重置
    }

    def __init__(
        self,
        name: str = "HMAC-Net",
        log_dir: Optional[str] = None,
        use_tensorboard: bool = False,
        console_level: str = "INFO",
        file_level: str = "DEBUG"
    ):
        """
        初始化日誌記錄器

        參數:
            name: 日誌記錄器名稱
            log_dir: 日誌檔案目錄（None 則不記錄到檔案）
            use_tensorboard: 是否使用 TensorBoard
            console_level: 控制台日誌級別
            file_level: 檔案日誌級別
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # 清除現有的處理器

        # 設定控制台處理器
        self._setup_console_handler(console_level)

        # 設定檔案處理器
        if log_dir is not None:
            self._setup_file_handler(log_dir, file_level)

        # 設定 TensorBoard
        self.writer = None
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self._setup_tensorboard(log_dir or "results/logs")

    def _setup_console_handler(self, level: str):
        """設定控制台處理器"""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))

        # 使用彩色格式化器
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

    def _setup_file_handler(self, log_dir: str, level: str):
        """設定檔案處理器"""
        # 建立日誌目錄
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # 建立日誌檔案（帶時間戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{self.name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))

        # 檔案使用詳細格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.info(f"日誌檔案: {log_file}")

    def _setup_tensorboard(self, log_dir: str):
        """設定 TensorBoard"""
        tb_dir = Path(log_dir) / "tensorboard" / datetime.now().strftime("%Y%m%d_%H%M%S")
        tb_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(tb_dir))
        self.info(f"TensorBoard 日誌目錄: {tb_dir}")

    # 基本日誌方法
    def debug(self, message: str):
        """輸出 DEBUG 級別日誌"""
        self.logger.debug(message)

    def info(self, message: str):
        """輸出 INFO 級別日誌"""
        self.logger.info(message)

    def warning(self, message: str):
        """輸出 WARNING 級別日誌"""
        self.logger.warning(message)

    def error(self, message: str):
        """輸出 ERROR 級別日誌"""
        self.logger.error(message)

    def critical(self, message: str):
        """輸出 CRITICAL 級別日誌"""
        self.logger.critical(message)

    # TensorBoard 方法
    def log_scalar(self, tag: str, value: float, step: int):
        """記錄標量值到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_value_dict: dict, step: int):
        """記錄多個標量值到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalars(main_tag, tag_value_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """記錄直方圖到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_histogram(tag, values, step)

    def log_figure(self, tag: str, figure, step: int):
        """記錄圖表到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_figure(tag, figure, step)

    def log_text(self, tag: str, text: str, step: int):
        """記錄文本到 TensorBoard"""
        if self.writer is not None:
            self.writer.add_text(tag, text, step)

    def close(self):
        """關閉日誌記錄器"""
        if self.writer is not None:
            self.writer.close()

        # 關閉所有處理器
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


class ColoredFormatter(logging.Formatter):
    """
    彩色日誌格式化器
    為不同級別的日誌添加顏色
    """

    def format(self, record):
        # 保存原始級別名稱
        levelname = record.levelname

        # 添加顏色
        if levelname in Logger.COLORS:
            record.levelname = (
                f"{Logger.COLORS[levelname]}{levelname}{Logger.COLORS['RESET']}"
            )

        # 格式化
        result = super().format(record)

        # 恢復原始級別名稱
        record.levelname = levelname

        return result


class MetricLogger:
    """
    指標記錄器
    專門用於記錄訓練過程中的指標
    """

    def __init__(self, logger: Logger):
        """
        初始化指標記錄器

        參數:
            logger: Logger 實例
        """
        self.logger = logger
        self.metrics = {}

    def update(self, **kwargs):
        """
        更新指標

        參數:
            **kwargs: 指標名稱和值的鍵值對
        """
        for key, value in kwargs.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def log_epoch(self, epoch: int, prefix: str = ""):
        """
        記錄一個 epoch 的指標

        參數:
            epoch: 當前 epoch 數
            prefix: 指標前綴（例如 "train" 或 "val"）
        """
        # 計算平均值
        avg_metrics = {
            key: sum(values) / len(values)
            for key, values in self.metrics.items()
        }

        # 格式化輸出
        metrics_str = " | ".join([
            f"{key}: {value:.4f}"
            for key, value in avg_metrics.items()
        ])

        self.logger.info(f"Epoch {epoch} {prefix} - {metrics_str}")

        # 記錄到 TensorBoard
        for key, value in avg_metrics.items():
            tag = f"{prefix}/{key}" if prefix else key
            self.logger.log_scalar(tag, value, epoch)

        # 清空指標
        self.metrics.clear()

        return avg_metrics

    def reset(self):
        """重置所有指標"""
        self.metrics.clear()


def get_logger(
    name: str = "HMAC-Net",
    log_dir: Optional[str] = "results/logs",
    use_tensorboard: bool = False
) -> Logger:
    """
    獲取日誌記錄器的便捷函數

    參數:
        name: 日誌記錄器名稱
        log_dir: 日誌目錄
        use_tensorboard: 是否使用 TensorBoard

    返回:
        Logger 實例
    """
    return Logger(
        name=name,
        log_dir=log_dir,
        use_tensorboard=use_tensorboard
    )


if __name__ == "__main__":
    # 測試日誌記錄器
    logger = get_logger("Test", use_tensorboard=True)

    logger.debug("這是一條 DEBUG 訊息")
    logger.info("這是一條 INFO 訊息")
    logger.warning("這是一條 WARNING 訊息")
    logger.error("這是一條 ERROR 訊息")
    logger.critical("這是一條 CRITICAL 訊息")

    # 測試 TensorBoard
    for i in range(10):
        logger.log_scalar("test/loss", 1.0 / (i + 1), i)

    logger.close()
    print("\n日誌記錄器測試完成！")
