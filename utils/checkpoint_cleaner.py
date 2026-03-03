"""
checkpoint 清理工具
====================
提供可程式化呼叫的 checkpoint 清理函式。
保留每個實驗的最終最佳 checkpoint（F1 最高），刪除其餘中間檔案。
同時清理 DAPT 中間 checkpoint 的 optimizer.pt。

對外 API：
    run_cleanup(execute=False, result_dirs=None, skip_dapt=False, skip_nul=False)
        → 執行清理並回傳摘要字典

    print_cleanup_summary(summary)
        → 印出人類可讀的清理報告
"""

import re
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# F1 score 從檔名解析，例如 best_model_epoch17_f1_0.7747.pt → 0.7747
# ─────────────────────────────────────────────────────────────────────────────
_F1_PATTERN = re.compile(r"_f1_([\d.]+)\.pt$", re.IGNORECASE)

_DEFAULT_RESULT_DIRS = ["ablation", "baseline", "improved"]


def _parse_f1(filename: str) -> float:
    """從檔名解析 F1 值，解析失敗回傳 -1.0。"""
    m = _F1_PATTERN.search(filename)
    return float(m.group(1)) if m else -1.0


def _fmt_size(size_bytes: int) -> str:
    """格式化位元組為易讀字串。"""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _collect_size(paths: list) -> int:
    """計算多個路徑的總大小（跳過不存在的）。"""
    total = 0
    for p in paths:
        try:
            total += p.stat().st_size
        except FileNotFoundError:
            pass
    return total


def _scan_intermediate_checkpoints(
    results_root: Path, result_dirs: list
) -> tuple:
    """
    掃描 results/{category}/{dataset}/{exp}/checkpoints/ 下的 checkpoint 檔案。
    回傳 (要刪除的路徑列表, 要保留的路徑列表, 無法解析的路徑列表, 可回收位元組數)。
    """
    to_delete: list = []
    to_keep:   list = []
    no_pattern: list = []

    for category in result_dirs:
        cat_dir = results_root / category
        if not cat_dir.exists():
            continue

        # 掃描每個實驗目錄下的 checkpoints/
        ckpt_dirs = sorted(cat_dir.glob("*/*/checkpoints"))
        if not ckpt_dirs:
            # 有些目錄只有一層（dataset 直接在 category 下）
            ckpt_dirs = sorted(cat_dir.glob("*/checkpoints"))

        for ckpt_dir in ckpt_dirs:
            pts = [f for f in ckpt_dir.glob("*.pt") if _F1_PATTERN.search(f.name)]
            non_pts = [f for f in ckpt_dir.glob("*.pt") if not _F1_PATTERN.search(f.name)]

            if non_pts:
                no_pattern.extend(non_pts)

            if not pts:
                continue

            # 找 F1 最高的
            best = max(pts, key=lambda p: _parse_f1(p.name))
            to_keep.append(best)
            for p in pts:
                if p != best:
                    to_delete.append(p)

    total_bytes = _collect_size(to_delete)
    return to_delete, to_keep, no_pattern, total_bytes


def _scan_dapt_optimizers(data_root: Path) -> tuple:
    """
    找出 data/dapt/*/checkpoints/*/optimizer.pt（中間 checkpoint 的 optimizer）。
    final/ 目錄下的不動。
    """
    to_delete: list = []
    for optimizer_pt in data_root.glob("dapt/*/checkpoints/*/optimizer.pt"):
        to_delete.append(optimizer_pt)
    total_bytes = _collect_size(to_delete)
    return to_delete, total_bytes


def _scan_nul_file(project_root: Path) -> list:
    nul = project_root / "nul"
    return [nul] if nul.exists() else []


def _do_delete(files: list, label: str) -> dict:
    """實際刪除檔案，回傳刪除統計。"""
    deleted = 0
    failed  = 0
    for f in files:
        try:
            f.unlink()
            deleted += 1
        except Exception as e:
            print(f"  [錯誤] 無法刪除 {f.name}: {e}")
            failed += 1
    return {"label": label, "deleted": deleted, "failed": failed}


# ─────────────────────────────────────────────────────────────────────────────
# 對外公開 API
# ─────────────────────────────────────────────────────────────────────────────

def run_cleanup(
    project_root: Path = None,
    execute: bool = False,
    result_dirs: list = None,
    skip_dapt: bool = False,
    skip_nul: bool = False,
) -> dict:
    """
    執行 checkpoint 清理。

    Args:
        project_root: 專案根目錄（預設為此檔案的上層目錄）
        execute:      True 才實際刪除，False 為 dry-run
        result_dirs:  要清理的 results 子目錄名稱列表
        skip_dapt:    是否跳過 DAPT optimizer.pt 清理
        skip_nul:     是否跳過根目錄 nul 檔案清理

    Returns:
        摘要字典，包含以下欄位：
            execute, ckpt_delete, ckpt_keep, ckpt_bytes,
            dapt_delete, dapt_bytes, nul_files, no_pattern,
            total_bytes, delete_results（execute=True 時才有）
    """
    if project_root is None:
        # 此檔案位於 utils/，上一層即為專案根目錄
        project_root = Path(__file__).parent.parent

    results_root = project_root / "results"
    data_root    = project_root / "data"

    if result_dirs is None:
        result_dirs = _DEFAULT_RESULT_DIRS

    ckpt_delete, ckpt_keep, no_pattern, ckpt_bytes = _scan_intermediate_checkpoints(
        results_root, result_dirs
    )
    dapt_delete, dapt_bytes = ([], 0) if skip_dapt else _scan_dapt_optimizers(data_root)
    nul_files = [] if skip_nul else _scan_nul_file(project_root)

    total_bytes = ckpt_bytes + dapt_bytes + _collect_size(nul_files)

    summary = {
        "execute":     execute,
        "ckpt_delete": ckpt_delete,
        "ckpt_keep":   ckpt_keep,
        "ckpt_bytes":  ckpt_bytes,
        "dapt_delete": dapt_delete,
        "dapt_bytes":  dapt_bytes,
        "nul_files":   nul_files,
        "no_pattern":  no_pattern,
        "total_bytes": total_bytes,
        "project_root": project_root,
    }

    if execute:
        results = []
        results.append(_do_delete(ckpt_delete, "中間 checkpoint"))
        results.append(_do_delete(dapt_delete, "DAPT optimizer.pt"))
        results.append(_do_delete(nul_files,   "nul 檔案"))
        summary["delete_results"] = results

    return summary


def print_cleanup_summary(summary: dict) -> None:
    """印出清理摘要報告。"""
    execute      = summary["execute"]
    ckpt_delete  = summary["ckpt_delete"]
    ckpt_keep    = summary["ckpt_keep"]
    ckpt_bytes   = summary["ckpt_bytes"]
    dapt_delete  = summary["dapt_delete"]
    dapt_bytes   = summary["dapt_bytes"]
    nul_files    = summary["nul_files"]
    no_pattern   = summary["no_pattern"]
    total_bytes  = summary["total_bytes"]
    project_root = summary["project_root"]

    print("\n" + "=" * 70)
    print(f"  Checkpoint 清理{'（執行模式）' if execute else '（Dry-Run 模式）'}")
    print("=" * 70)

    # ── 無法解析的 .pt ──
    if no_pattern:
        print(f"\n  [注意] 以下 .pt 檔無法解析 F1 值，不會自動刪除（請手動確認）：")
        for p in no_pattern:
            try:
                rel = p.relative_to(project_root)
            except ValueError:
                rel = p
            print(f"    {rel}")

    # ── 中間 checkpoint ──
    print(f"\n[1] Results 中間 checkpoint（保留最高 F1，刪除其餘）")
    print(f"    保留: {len(ckpt_keep):>4} 個最終 checkpoint")
    print(f"    刪除: {len(ckpt_delete):>4} 個中間 checkpoint  →  可回收 {_fmt_size(ckpt_bytes)}")
    if ckpt_delete:
        exp_map: dict = {}
        for p in ckpt_delete:
            exp_key = str(p.parent.parent.relative_to(project_root))
            exp_map.setdefault(exp_key, []).append(p)
        print(f"\n    刪除清單（共 {len(exp_map)} 個實驗）：")
        for exp_key, files in sorted(exp_map.items()):
            size = _fmt_size(_collect_size(files))
            print(f"      {exp_key}  [{len(files)} 個，{size}]")

    # ── DAPT optimizer ──
    print(f"\n[2] DAPT 中間 optimizer.pt")
    if dapt_delete:
        print(f"    刪除: {len(dapt_delete)} 個 optimizer.pt  →  可回收 {_fmt_size(dapt_bytes)}")
        for p in dapt_delete:
            try:
                rel = p.relative_to(project_root)
            except ValueError:
                rel = p
            print(f"      {rel}")
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
    print(f"  預計可回收：{_fmt_size(total_bytes)}")

    if execute and "delete_results" in summary:
        print(f"\n  刪除結果：")
        for r in summary["delete_results"]:
            print(f"    {r['label']}: 刪除 {r['deleted']} 個，失敗 {r['failed']} 個")
    elif not execute:
        print(f"\n  [!] 這是 Dry-Run，尚未刪除任何檔案。")
        print(f"      確認後執行：python tests/cleanup_checkpoints.py --execute")

    print("=" * 70 + "\n")
