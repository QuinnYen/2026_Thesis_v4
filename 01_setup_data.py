"""
setup_data.py — 第一次啟動專案時執行此腳本

功能：
  Step 1: 從 HuggingFace 下載 raw/, SenticNet_5.0/, laptop_dapt, restaurant_dapt
  Step 2: 提示用戶手動下載 Amazon/Yelp 原始語料（版權限制，無法自動下載）
  Step 3: 確認語料存在後，執行 extract_yelp_restaurants.py 提取 Yelp 餐廳評論
  Step 4: 執行 domain_pretrain.py 進行 DAPT 訓練（laptops + restaurants）

使用方法：
  python setup_data.py              # 完整流程（若 HF 已有 DAPT 模型則跳過 Step 3-4）
  python setup_data.py --skip-dapt  # 只下載資料，不跑 DAPT
  python setup_data.py --only-dapt  # 假設資料已存在，只跑 DAPT
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# ─────────────────────────────────────────
# 設定
# ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

HF_DATASET_REPO = "MatchaCat4477/2026-thesis-data"
HF_LAPTOP_MODEL = "MatchaCat4477/bert-laptop-dapt"
HF_RESTAURANT_MODEL = "MatchaCat4477/bert-restaurant-dapt"

# 需要用戶手動下載的檔案（版權限制）
MANUAL_DOWNLOAD_INSTRUCTIONS = """
以下語料因版權限制無法自動下載，請手動取得後放到指定位置：

[1] Amazon Electronics 5-core (Electronics_5.json, ~1.4GB)
    下載網址: https://nijianmo.github.io/amazon/index.html
    選擇: Electronics -> 5-core
    放置位置: data/dapt/unlabeled/Electronics_5.json

[2] Yelp Academic Dataset (~5GB)
    下載網址: https://www.yelp.com/dataset
    需要填寫申請表格
    放置位置:
      data/dapt/unlabeled/Yelp/yelp_academic_dataset_business.json
      data/dapt/unlabeled/Yelp/yelp_academic_dataset_review.json

完成後請重新執行: python setup_data.py
"""


def check_huggingface_hub():
    try:
        import huggingface_hub
        return True
    except ImportError:
        print("[ERROR] 缺少套件 huggingface_hub，請執行：pip install huggingface_hub")
        sys.exit(1)


def step1_download_from_hf():
    print("\n" + "="*60)
    print("Step 1: 從 HuggingFace 下載資料集與模型")
    print("="*60)

    from huggingface_hub import snapshot_download

    # 下載 raw/ 和 SenticNet_5.0/
    dataset_dest = DATA_DIR
    dataset_dest.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] 下載資料集 ({HF_DATASET_REPO})...")
    snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        local_dir=str(dataset_dest),
        local_dir_use_symlinks=False,
    )
    print("      資料集下載完成")

    # 下載 laptop_dapt 模型
    laptop_dest = DATA_DIR / "dapt" / "laptop_dapt" / "final"
    laptop_dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[2/3] 下載 laptop DAPT 模型 ({HF_LAPTOP_MODEL})...")
    snapshot_download(
        repo_id=HF_LAPTOP_MODEL,
        repo_type="model",
        local_dir=str(laptop_dest),
        local_dir_use_symlinks=False,
    )
    print("      laptop DAPT 模型下載完成")

    # 下載 restaurant_dapt 模型
    restaurant_dest = DATA_DIR / "dapt" / "restaurant_dapt" / "final"
    restaurant_dest.mkdir(parents=True, exist_ok=True)
    print(f"\n[3/3] 下載 restaurant DAPT 模型 ({HF_RESTAURANT_MODEL})...")
    snapshot_download(
        repo_id=HF_RESTAURANT_MODEL,
        repo_type="model",
        local_dir=str(restaurant_dest),
        local_dir_use_symlinks=False,
    )
    print("      restaurant DAPT 模型下載完成")


def step2_check_manual_downloads():
    print("\n" + "="*60)
    print("Step 2: 確認手動下載的原始語料")
    print("="*60)

    amazon_path = DATA_DIR / "dapt" / "unlabeled" / "Electronics_5.json"
    yelp_business = DATA_DIR / "dapt" / "unlabeled" / "Yelp" / "yelp_academic_dataset_business.json"
    yelp_review = DATA_DIR / "dapt" / "unlabeled" / "Yelp" / "yelp_academic_dataset_review.json"

    missing = []
    if not amazon_path.exists():
        missing.append(str(amazon_path))
    if not yelp_business.exists():
        missing.append(str(yelp_business))
    if not yelp_review.exists():
        missing.append(str(yelp_review))

    if missing:
        print("\n[WARNING] 以下檔案缺失：")
        for f in missing:
            print(f"  - {f}")
        print(MANUAL_DOWNLOAD_INSTRUCTIONS)
        return False

    print("\n[OK] 所有原始語料確認存在")
    return True


def step3_extract_yelp():
    print("\n" + "="*60)
    print("Step 3: 提取 Yelp 餐廳評論語料")
    print("="*60)

    corpus_path = DATA_DIR / "dapt" / "unlabeled" / "Yelp" / "yelp_restaurant_corpus.txt"
    if corpus_path.exists():
        print(f"\n[SKIP] 已存在 {corpus_path}，跳過提取步驟")
        return True

    script = PROJECT_ROOT / "datasets" / "dapt" / "extract_yelp_restaurants.py"
    print(f"\n執行: {script}")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT)
    )
    if result.returncode != 0:
        print("[ERROR] Yelp 語料提取失敗")
        return False

    print("[OK] Yelp 語料提取完成")
    return True


def step4_run_dapt():
    print("\n" + "="*60)
    print("Step 4: 執行 Domain-Adaptive Pre-training (DAPT)")
    print("="*60)

    script = PROJECT_ROOT / "datasets" / "dapt" / "domain_pretrain.py"

    for domain in ["laptops", "restaurants"]:
        dest = DATA_DIR / "dapt" / f"{'laptop' if domain == 'laptops' else 'restaurant'}_dapt" / "final"
        if (dest / "model.safetensors").exists():
            print(f"\n[SKIP] {domain} DAPT 模型已存在，跳過訓練")
            continue

        print(f"\n執行 DAPT: {domain}")
        result = subprocess.run(
            [sys.executable, str(script), "--dataset", domain],
            cwd=str(PROJECT_ROOT)
        )
        if result.returncode != 0:
            print(f"[ERROR] {domain} DAPT 訓練失敗")
            return False

    print("\n[OK] DAPT 訓練完成")
    return True


def main():
    parser = argparse.ArgumentParser(description="專案資料初始化腳本")
    parser.add_argument("--skip-dapt", action="store_true",
                        help="跳過 DAPT 訓練（只下載資料）")
    parser.add_argument("--only-dapt", action="store_true",
                        help="跳過下載，只執行 DAPT 訓練")
    args = parser.parse_args()

    check_huggingface_hub()

    print("\n" + "="*60)
    print("  2026 Thesis — 資料初始化")
    print("="*60)

    if not args.only_dapt:
        step1_download_from_hf()

    if args.skip_dapt:
        print("\n[--skip-dapt] 跳過 DAPT 相關步驟，初始化完成。")
        return

    unlabeled_ok = step2_check_manual_downloads()
    if not unlabeled_ok:
        print("\n請完成手動下載後重新執行此腳本。")
        print("若你選擇使用 HuggingFace 上已有的 DAPT 模型（不重新訓練），")
        print("可以直接執行: python setup_data.py --skip-dapt")
        sys.exit(0)

    yelp_ok = step3_extract_yelp()
    if not yelp_ok:
        sys.exit(1)

    dapt_ok = step4_run_dapt()
    if not dapt_ok:
        sys.exit(1)

    print("\n" + "="*60)
    print("  初始化完成！專案已可正常執行。")
    print("="*60)


if __name__ == "__main__":
    main()
