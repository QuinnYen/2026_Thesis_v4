"""
Domain-Adaptive Pre-training (DAPT) for BERT

用無標籤的 Amazon 筆電評論做 MLM 任務，
讓 BERT 學習筆電領域的語言模式。

使用方法:
    # Step 1: 預訓練 (約 1-2 小時，視數據量而定)
    python data/domain_pretrain.py --dataset laptops --epochs 3
    python data/domain_pretrain.py --dataset restaurants --epochs 3

    # Step 2: 訓練時使用 domain-adapted BERT
    # 在 config 中設定 bert_model: "saved_models/bert_laptop_dapt"
"""

import os
import json
import argparse
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)


class DomainTextDataset(Dataset):
    """領域文本數據集 for MLM"""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encodings = []

        print(f"Tokenizing {len(texts)} texts...")
        for text in tqdm(texts):
            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def load_domain_texts(data_path: Path, domain: str = 'laptop', max_samples: int = 50000) -> List[str]:
    """
    載入領域文本

    支援格式：
    1. 純文字檔 (.txt)：每行一筆評論
    2. JSON Array：[{"text": "..."}, ...]
    3. JSONL：每行一個 JSON

    Args:
        data_path: 數據文件路徑
        domain: 領域 ('laptop' 或 'restaurant')
        max_samples: 最大樣本數
    """
    if not data_path.exists():
        print(f"Error: Data not found at {data_path}")
        return []

    texts = []

    # 筆電相關關鍵詞
    laptop_keywords = [
        'laptop', 'notebook', 'computer', 'screen', 'display', 'battery',
        'keyboard', 'trackpad', 'touchpad', 'processor', 'cpu', 'ram',
        'memory', 'storage', 'ssd', 'hard drive', 'graphics', 'gpu',
        'webcam', 'speaker', 'fan', 'heat', 'weight', 'port', 'usb',
        'charger', 'power', 'boot', 'windows', 'mac', 'chrome'
    ]

    # 檢測文件格式
    file_ext = data_path.suffix.lower()

    # 純文字格式 (.txt)：每行一筆評論
    if file_ext == '.txt':
        print(f"偵測到純文字格式，載入最多 {max_samples} 筆...")
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="載入評論"):
                if len(texts) >= max_samples:
                    break
                text = line.strip()
                if text and len(text) >= 50:
                    texts.append(text)
        print(f"已載入 {len(texts)} 筆領域文本")
        return texts

    # JSON/JSONL 格式
    with open(data_path, 'r', encoding='utf-8') as f:
        first_char = f.read(1)

    if first_char == '[':
        # JSON Array 格式
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                text = item.get('text', '') or item.get('reviewText', '')
                if text and len(text) >= 50:
                    texts.append(text)
                if len(texts) >= max_samples:
                    break
    else:
        # JSONL 格式 (Amazon 5-core)
        print(f"Detected JSONL format, loading up to {max_samples} samples...")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading reviews"):
                if len(texts) >= max_samples:
                    break

                try:
                    item = json.loads(line.strip())
                    text = item.get('reviewText', '')

                    if not text or len(text) < 50:
                        continue

                    # 筆電領域過濾
                    if domain == 'laptop':
                        text_lower = text.lower()
                        if not any(kw in text_lower for kw in laptop_keywords):
                            continue

                    texts.append(text)

                except json.JSONDecodeError:
                    continue

    print(f"Loaded {len(texts)} domain texts")
    return texts


def run_dapt(
    texts: List[str],
    output_dir: Path,
    base_model: str = 'bert-base-uncased',
    epochs: int = 3,
    batch_size: int = 16,
    max_length: int = 128,
    mlm_probability: float = 0.15,
    learning_rate: float = 5e-5
):
    """
    執行 Domain-Adaptive Pre-training

    Args:
        texts: 領域文本列表
        output_dir: 輸出目錄
        base_model: 基礎 BERT 模型
        epochs: 訓練輪數
        batch_size: 批次大小
        max_length: 最大序列長度
        mlm_probability: MLM 遮蔽比例
        learning_rate: 學習率
    """
    print(f"\n{'='*60}")
    print(f"Domain-Adaptive Pre-training (DAPT)")
    print(f"{'='*60}")
    print(f"  Base model: {base_model}")
    print(f"  Texts: {len(texts)}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  MLM probability: {mlm_probability}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    # 載入 tokenizer 和模型
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = BertForMaskedLM.from_pretrained(base_model)

    # 建立數據集
    dataset = DomainTextDataset(texts, tokenizer, max_length)

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability
    )

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=str(output_dir / 'checkpoints'),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir=str(output_dir / 'logs'),
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,  # Windows 兼容
        report_to='none'
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )

    # 訓練
    print("\nStarting DAPT training...")
    trainer.train()

    # 保存模型
    final_output = output_dir / 'final'
    print(f"\nSaving domain-adapted BERT to {final_output}...")
    model.save_pretrained(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    print("\nDAPT completed!")

    return str(final_output)


def main():
    parser = argparse.ArgumentParser(description="Domain-Adaptive Pre-training for BERT")
    parser.add_argument('--dataset', type=str, default='laptops',
                       choices=['laptops', 'restaurants'],
                       help='Dataset domain')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--max-samples', type=int, default=50000,
                       help='Maximum number of texts to use')
    parser.add_argument('--base-model', type=str, default='bert-base-uncased',
                       help='Base BERT model')
    parser.add_argument('--unlabeled-path', type=str, default=None,
                       help='Path to unlabeled data')

    args = parser.parse_args()

    # 路徑設置
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / 'data'

    # 確定數據路徑
    if args.dataset == 'laptops':
        domain = 'laptop'
        # 優先使用 Amazon 5-core
        default_path = data_dir / 'unlabeled' / 'Electronics_5.json'
        if not default_path.exists():
            default_path = data_dir / 'unlabeled' / 'amazon_laptops.json'
        output_name = 'laptop_dapt'
    else:
        domain = 'restaurant'
        # 優先使用已提取的純文字檔
        default_path = data_dir / 'unlabeled' / 'Yelp' / 'yelp_restaurant_corpus.txt'
        if not default_path.exists():
            default_path = data_dir / 'unlabeled' / 'yelp_restaurants.json'
        output_name = 'restaurant_dapt'

    data_path = Path(args.unlabeled_path) if args.unlabeled_path else default_path
    output_dir = data_dir / 'dapt' / output_name

    # 載入文本
    print(f"\nLoading domain texts from {data_path}...")
    texts = load_domain_texts(data_path, domain, args.max_samples)

    if not texts:
        print("No texts loaded. Please check the data path.")
        return

    # 執行 DAPT
    saved_path = run_dapt(
        texts=texts,
        output_dir=output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print(f"\n{'='*60}")
    print(f"Domain-Adapted BERT saved to: {saved_path}")
    print(f"{'='*60}")
    print(f"\nTo use in HKGAN, update your config:")
    print(f"  model:")
    print(f"    bert_model: \"{saved_path}\"")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
