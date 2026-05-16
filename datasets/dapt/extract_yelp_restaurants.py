"""
Yelp 餐廳評論提取腳本

從 Yelp 資料集中提取餐廳評論，用於 DAPT 訓練。

步驟：
1. 從 business.json 篩選餐廳 ID
2. 從 review.json 提取對應評論（Line-by-Line 避免記憶體爆炸）
3. 隨機抽樣 50-100 萬筆

使用方法：
    python data/extract_yelp_restaurants.py --max_samples 500000
"""

import json
import random
import argparse
from pathlib import Path
from tqdm import tqdm


def load_restaurant_ids(business_path: Path) -> set:
    """
    步驟一：從 business.json 載入餐廳 ID

    篩選條件：categories 包含 "Restaurants" 或 "Food"
    """
    restaurant_ids = set()

    print(f"正在載入商家資料: {business_path}")
    print("篩選條件: categories 包含 'Restaurants' 或 'Food'")

    with open(business_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="讀取商家"):
            try:
                business = json.loads(line.strip())
                categories = business.get('categories', '') or ''

                # 檢查是否為餐廳
                if 'Restaurants' in categories or 'Food' in categories:
                    restaurant_ids.add(business['business_id'])

            except json.JSONDecodeError:
                continue

    print(f"找到 {len(restaurant_ids):,} 家餐廳")
    return restaurant_ids


def extract_reviews(
    review_path: Path,
    restaurant_ids: set,
    output_path: Path,
    max_samples: int = 500000,
    min_length: int = 50
) -> int:
    """
    步驟二 & 三：提取餐廳評論並隨機抽樣

    使用 reservoir sampling 進行隨機抽樣，避免需要先計算總數
    """
    print(f"\n正在提取評論: {review_path}")
    print(f"目標數量: {max_samples:,} 筆")
    print(f"最小長度: {min_length} 字元")

    # Reservoir sampling
    reservoir = []
    total_restaurant_reviews = 0

    with open(review_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="讀取評論"):
            try:
                review = json.loads(line.strip())

                # 檢查是否為餐廳評論
                if review.get('business_id') not in restaurant_ids:
                    continue

                text = review.get('text', '').strip()

                # 過濾太短的評論
                if len(text) < min_length:
                    continue

                total_restaurant_reviews += 1

                # Reservoir sampling
                if len(reservoir) < max_samples:
                    reservoir.append(text)
                else:
                    # 隨機替換
                    idx = random.randint(0, total_restaurant_reviews - 1)
                    if idx < max_samples:
                        reservoir[idx] = text

            except json.JSONDecodeError:
                continue

    print(f"\n餐廳評論總數: {total_restaurant_reviews:,}")
    print(f"抽樣數量: {len(reservoir):,}")

    # 打亂順序
    random.shuffle(reservoir)

    # 寫入輸出檔案
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for text in tqdm(reservoir, desc="寫入檔案"):
            # 移除換行符，確保每行一筆
            text_clean = text.replace('\n', ' ').replace('\r', ' ')
            f.write(text_clean + '\n')

    print(f"\n已儲存至: {output_path}")

    # 顯示統計
    total_chars = sum(len(t) for t in reservoir)
    avg_length = total_chars / len(reservoir) if reservoir else 0

    print(f"\n=== 統計 ===")
    print(f"評論數量: {len(reservoir):,}")
    print(f"總字元數: {total_chars:,}")
    print(f"平均長度: {avg_length:.1f} 字元")

    # 顯示範例
    print(f"\n=== 範例評論 ===")
    for i, text in enumerate(reservoir[:3]):
        print(f"\n[{i+1}] {text[:200]}...")

    return len(reservoir)


def main():
    parser = argparse.ArgumentParser(description="提取 Yelp 餐廳評論")
    parser.add_argument('--max_samples', type=int, default=500000,
                        help='最大抽樣數量 (預設: 500000)')
    parser.add_argument('--min_length', type=int, default=50,
                        help='評論最小長度 (預設: 50)')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子 (預設: 42)')

    args = parser.parse_args()

    # 設定隨機種子
    random.seed(args.seed)

    # 路徑設定
    data_dir = Path(__file__).parent.parent.parent / 'data'
    yelp_dir = data_dir / 'unlabeled' / 'Yelp'

    business_path = yelp_dir / 'yelp_academic_dataset_business.json'
    review_path = yelp_dir / 'yelp_academic_dataset_review.json'
    output_path = yelp_dir / 'yelp_restaurant_corpus.txt'

    # 檢查檔案是否存在
    if not business_path.exists():
        print(f"錯誤: 找不到商家資料 {business_path}")
        return

    if not review_path.exists():
        print(f"錯誤: 找不到評論資料 {review_path}")
        return

    print("=" * 60)
    print("Yelp 餐廳評論提取")
    print("=" * 60)
    print(f"商家資料: {business_path}")
    print(f"評論資料: {review_path}")
    print(f"輸出路徑: {output_path}")
    print(f"最大抽樣: {args.max_samples:,}")
    print("=" * 60)

    # 步驟一：載入餐廳 ID
    restaurant_ids = load_restaurant_ids(business_path)

    if not restaurant_ids:
        print("錯誤: 沒有找到餐廳")
        return

    # 步驟二 & 三：提取評論並抽樣
    count = extract_reviews(
        review_path=review_path,
        restaurant_ids=restaurant_ids,
        output_path=output_path,
        max_samples=args.max_samples,
        min_length=args.min_length
    )

    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)
    print(f"\n下一步：執行 DAPT 訓練")
    print(f"python data/domain_pretrain.py --dataset restaurants \\")
    print(f"       --unlabeled-path {output_path}")


if __name__ == "__main__":
    main()
