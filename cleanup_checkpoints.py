#!/usr/bin/env python3
"""
checkpoint 清理腳本
====================
保留每個實驗的最終最佳 checkpoint（F1 最高），刪除其餘中間檔案。
同時清理 DAPT 中間 checkpoint 的 optimizer.pt。

用法：
    python cleanup_checkpoints.py           # dry-run（只列出，不刪除）
    python cleanup_checkpoints.py --execute # 實際執行刪除

選項：
    --results-dirs  指定要清理的 results 子目錄（預設: ablation baseline improved）
    --skip-dapt     跳過 DAPT optimizer.pt 清理
    --skip-nul      跳過根目錄 nul 檔案清理
"""

import argparse
import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# 設定
# ─────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
RESULTS_ROOT = PROJECT_ROOT / "results"
DATA_ROOT    = PROJECT_ROOT / "data"

# F1 score 從檔名解析，例如 best_model_epoch17_f1_0.7747.pt → 0.7747
F1_PATTERN = re.compile(r"_f1_([\d.]+)\.pt$", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# 輔助函式
# ─────────────────────────────────────────────────────────────────────────────
def parse_f1(filename: str) -> float:
    """從檔名解析 F1 值，解析失敗回傳 -1.0。"""
    m = F1_PATTERN.search(filename)
    return float(m.group(1)) if m else -1.0


def fmt_size(size_bytes: int) -> str:
    """格式化位元組為易讀字串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def collect_size(paths: list[Path]) -> int:
    """計算多個路徑的總大小（跳過不存在的）。"""
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total


# ─────────────────────────────────────────────────────────────────────────────
# 掃描：results/ 中間 checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def scan_intermediate_checkpoints(result_dirs: list[str]) -> tuple[list[Path], list[Path], int]:
    """
    掃描 results/{category}/{dataset}/{exp}/checkpoints/ 下的 checkpoint 檔案。
    回傳 (要刪除的路徑列表, 要保留的路徑列表, 可回收位元組數)。
    """
    to_delete: list[Path] = []
    to_keep:   list[Path] = []
    no_pattern: list[Path] = []

    for category in result_dirs:
        cat_dir = RESULTS_ROOT / category
        if not cat_dir.exists():
            print(f"  [跳過] {cat_dir} 不存在")
            continue

        # 掃描每個實驗目錄下的 checkpoints/
        ckpt_dirs = sorted(cat_dir.glob("*/*/checkpoints"))
        if not ckpt_dirs:
            # 有些目錄只有一層（dataset 直接在 category 下）
            ckpt_dirs = sorted(cat_dir.glob("*/checkpoints"))

        for ckpt_dir in ckpt_dirs:
            pts = [f for f in ckpt_dir.glob("*.pt") if F1_PATTERN.search(f.name)]
            non_pts = [f for f in ckpt_dir.glob("*.pt") if not F1_PATTERN.search(f.name)]

            if non_pts:
                no_pattern.extend(non_pts)

            if not pts:
                continue

            # 找 F1 最高的
            best = max(pts, key=lambda p: parse_f1(p.name))
            to_keep.append(best)
            for p in pts:
                if p != best:
                    to_delete.append(p)

    if no_pattern:
        print(f"\n  [注意] 以下 .pt 檔無法解析 F1 值，不會自動刪除（請手動確認）：")
        for p in no_pattern:
            print(f"    {p.relative_to(PROJECT_ROOT)}")

    total_bytes = collect_size(to_delete)
    return to_delete, to_keep, total_bytes


# ─────────────────────────────────────────────────────────────────────────────
# 掃描：DAPT optimizer.pt
# ─────────────────────────────────────────────────────────────────────────────
def scan_dapt_optimizers() -> tuple[list[Path], int]:
    """
    找出 data/dapt/*/checkpoints/*/optimizer.pt（中間 checkpoint 的 optimizer）。
    final/ 目錄下的不動。
    """
    to_delete: list[Path] = []
    # 只掃 data/dapt/*/checkpoints/（不掃 final/）
    for optimizer_pt in DATA_ROOT.glob("dapt/*/checkpoints/*/optimizer.pt"):
        to_delete.append(optimizer_pt)
    total_bytes = collect_size(to_delete)
    return to_delete, total_bytes


# ─────────────────────────────────────────────────────────────────────────────
# 掃描：根目錄 nul 檔案
# ─────────────────────────────────────────────────────────────────────────────
def scan_nul_file() -> list[Path]:
    nul = PROJECT_ROOT / "nul"
    return [nul] if nul.exists() else []


# ─────────────────────────────────────────────────────────────────────────────
# 列印報告
# ─────────────────────────────────────────────────────────────────────────────
def print_report(
    ckpt_delete: list[Path],
    ckpt_keep:   list[Path],
    ckpt_bytes:  int,
    dapt_delete: list[Path],
    dapt_bytes:  int,
    nul_files:   list[Path],
    execute:     bool,
) -> None:
    total_bytes = ckpt_bytes + dapt_bytes + collect_size(nul_files)

    print("\n" + "=" * 70)
    print(f"  Checkpoint 清理{'（執行模式）' if execute else '（Dry-Run 模式）'}")
    print("=" * 70)

    # ── 中間 checkpoint ──
    print(f"\n[1] Results 中間 checkpoint（保留最高 F1，刪除其餘）")
    print(f"    保留: {len(ckpt_keep):>4} 個最終 checkpoint")
    print(f"    刪除: {len(ckpt_delete):>4} 個中間 checkpoint  →  可回收 {fmt_size(ckpt_bytes)}")
    if ckpt_delete:
        # 每個實驗只展示一行摘要
        exp_map: dict[str, list[Path]] = {}
        for p in ckpt_delete:
            exp_key = str(p.parent.parent.relative_to(PROJECT_ROOT))
            exp_map.setdefault(exp_key, []).append(p)
        print(f"\n    刪除清單（共 {len(exp_map)} 個實驗，每行=1 實驗）：")
        for exp_key, files in sorted(exp_map.items()):
            size = fmt_size(collect_size(files))
            print(f"      {exp_key}  [{len(files)} 個，{size}]")

    # ── DAPT optimizer ──
    print(f"\n[2] DAPT 中間 optimizer.pt")
    if dapt_delete:
        print(f"    刪除: {len(dapt_delete)} 個 optimizer.pt  →  可回收 {fmt_size(dapt_bytes)}")
        for p in dapt_delete:
            print(f"      {p.relative_to(PROJECT_ROOT)}")
    else:
        print("    無需清理")

    # ── nul ──
    print(f"\n[3] 根目錄 nul 檔案")
    if nul_files:
        print(f"    刪除: {[str(p.name) for p in nul_files]}")
    else:
        print("    無需清理")

    # ── 總計 ──
    print(f"\n{'─' * 70}")
    print(f"  預計可回收：{fmt_size(total_bytes)}")
    if not execute:
        print(f"\n  [!] 這是 Dry-Run，尚未刪除任何檔案。")
        print(f"     確認後執行：python cleanup_checkpoints.py --execute")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 實際刪除
# ─────────────────────────────────────────────────────────────────────────────
def execute_delete(files: list[Path], label: str) -> int:
    deleted = 0
    failed  = 0
    for f in files:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"  [錯誤] 無法刪除 {f.name}: {e}")
            failed += 1
    print(f"  {label}: 刪除 {deleted} 個，失敗 {failed} 個")
    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="清理中間 checkpoint 以回收磁碟空間")
    parser.add_argument(
        "--execute", action="store_true",
        help="實際執行刪除（預設為 dry-run 模式）"
    )
    parser.add_argument(
        "--results-dirs", nargs="+",
        default=["ablation", "baseline", "improved"],
        metavar="DIR",
        help="要清理的 results 子目錄（預設: ablation baseline improved）"
    )
    parser.add_argument(
        "--skip-dapt", action="store_true",
        help="跳過 DAPT optimizer.pt 清理"
    )
    parser.add_argument(
        "--skip-nul", action="store_true",
        help="跳過根目錄 nul 檔案清理"
    )
    args = parser.parse_args()

    print(f"\n掃描中... 根目錄：{PROJECT_ROOT}")

    ckpt_delete, ckpt_keep, ckpt_bytes = scan_intermediate_checkpoints(args.results_dirs)
    dapt_delete, dapt_bytes = ([], 0) if args.skip_dapt else scan_dapt_optimizers()
    nul_files = [] if args.skip_nul else scan_nul_file()

    print_report(ckpt_delete, ckpt_keep, ckpt_bytes, dapt_delete, dapt_bytes, nul_files, args.execute)

    if args.execute:
        print("開始刪除...")
        execute_delete(ckpt_delete, "中間 checkpoint")
        execute_delete(dapt_delete, "DAPT optimizer.pt")
        execute_delete(nul_files,   "nul 檔案")
        print("\n清理完成。")
    else:
        print("（Dry-Run 完成，未刪除任何檔案）\n")


if __name__ == "__main__":
    main()
