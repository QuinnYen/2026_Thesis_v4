"""
Ensemble 推理工具模組

對指定資料集的多 seed checkpoint 執行等重 Ensemble 及 Logit Adjustment，
計算三種策略並儲存報告至 results/HKGAN_Ensemble_{dataset}.txt。

用法（程式內呼叫）：
    from utils.ensemble_runner import run_ensemble
    result = run_ensemble('laptops', save=True)

用法（run_experiments.py 自動呼叫）：
    訓練完 multi-seed 後由 run_ensemble_test(dataset) 自動觸發。
"""

import json
import sys
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm

# 專案根目錄（utils/ 的上一層）
_PROJECT_ROOT = Path(__file__).parent.parent

# 確保 project_root 和 experiments/ 在 sys.path
def _ensure_path():
    for p in [str(_PROJECT_ROOT), str(_PROJECT_ROOT / 'experiments')]:
        if p not in sys.path:
            sys.path.insert(0, p)

DATASET_MAP = {
    'restaurants': 'improved/restaurants',
    'laptops':     'improved/laptops',
    'mams':        'improved/mams',
    'rest16':      'improved/rest16',
    'lap16':       'improved/lap16',
}

DISPLAY_NAMES = {
    'restaurants': 'REST14',
    'laptops':     'LAP14',
    'mams':        'MAMS',
    'rest16':      'REST16',
    'lap16':       'LAP16',
}


# ─────────────────────────────────────────────────────────────
# 私有工具函式
# ─────────────────────────────────────────────────────────────

def _get_best_checkpoint(exp_dir: Path) -> Path:
    """從 checkpoints/ 目錄中選取 F1 最高的 .pt 檔。"""
    pts = sorted(
        (exp_dir / 'checkpoints').glob('*.pt'),
        key=lambda p: float(p.stem.split('f1_')[1])
    )
    return pts[-1]


def _resolve_bert_path(bert_model: str, project_root: Path) -> str:
    """將相對路徑解析為絕對路徑（若存在），否則原樣回傳（供 HuggingFace 遠端下載）。"""
    if not Path(bert_model).is_absolute():
        candidate = project_root / bert_model
        if candidate.exists():
            return candidate.as_posix()
    return bert_model


def _load_seeds_info(exp_root: Path) -> list:
    """
    掃描 exp_root 底下所有完成的實驗目錄，回傳 checkpoint 資訊列表。

    Returns:
        [{dir, pt, args, solo_f1}, ...]  按目錄名排序
    """
    seeds_info = []
    for d in sorted(exp_root.glob('*')):
        j = d / 'reports' / 'experiment_results.json'
        if not j.exists():
            continue
        try:
            data = json.load(open(j, encoding='utf-8'))
            ckpt = _get_best_checkpoint(d)
            seeds_info.append({
                'dir':      d,
                'pt':       ckpt,
                'args':     data.get('args', {}),
                'solo_f1':  data.get('test_metrics', {}).get('f1_macro', 0.0),
            })
        except (IndexError, FileNotFoundError, KeyError, json.JSONDecodeError):
            continue
    return seeds_info


def _build_data_loaders(dataset: str, ref_args: dict, project_root: Path):
    """
    根據資料集名稱建立 val/test DataLoader。

    Args:
        dataset:      資料集名稱（'restaurants'|'laptops'|'mams'|'rest16'|'lap16'）
        ref_args:     實驗 args 字典（從 experiment_results.json['args'] 讀取）
        project_root: 專案根目錄

    Returns:
        (val_loader, test_loader, n_val, n_test)
    """
    _ensure_path()
    from transformers import AutoTokenizer
    from datasets.multiaspect_dataset import create_multiaspect_dataloaders
    from datasets.loader_semeval14    import load_multiaspect_data
    from datasets.loader_semeval16    import load_semeval2016_data
    from datasets.loader_mams        import load_mams_data

    raw_dir = project_root / 'data' / 'raw'

    bert_model_path = _resolve_bert_path(ref_args.get('bert_model', 'bert-base-uncased'), project_root)
    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)

    if dataset == 'restaurants':
        train_all, test_s = load_multiaspect_data(
            str(raw_dir / 'semeval2014' / 'Restaurants_Train_v2.xml'),
            str(raw_dir / 'semeval2014' / 'Restaurants_Test_Gold.xml'),
            min_aspects=ref_args.get('min_aspects', 2),
            max_aspects=ref_args.get('max_aspects', 10),
            include_single_aspect=ref_args.get('include_single_aspect', False),
            virtual_aspect_mode=ref_args.get('virtual_aspect_mode', 'none'),
        )
        val_size = int(0.1 * len(train_all))
        val_s, train_s = train_all[:val_size], train_all[val_size:]

    elif dataset == 'laptops':
        train_all, test_s = load_multiaspect_data(
            str(raw_dir / 'semeval2014' / 'Laptop_Train_v2.xml'),
            str(raw_dir / 'semeval2014' / 'Laptops_Test_Gold.xml'),
            min_aspects=ref_args.get('min_aspects', 2),
            max_aspects=ref_args.get('max_aspects', 10),
            include_single_aspect=ref_args.get('include_single_aspect', False),
            virtual_aspect_mode=ref_args.get('virtual_aspect_mode', 'none'),
        )
        val_size = int(0.1 * len(train_all))
        val_s, train_s = train_all[:val_size], train_all[val_size:]

    elif dataset == 'mams':
        mams_dir = raw_dir / 'MAMS-ATSA'
        train_s, val_s, test_s = load_mams_data(
            str(mams_dir / 'train.xml'),
            str(mams_dir / 'val.xml'),
            str(mams_dir / 'test.xml'),
            max_aspects=ref_args.get('max_aspects', 10),
        )

    elif dataset == 'rest16':
        train_s, val_s, test_s = load_semeval2016_data(
            str(raw_dir / 'semeval2016' / 'restaurants16_train_sb1.xml'),
            str(raw_dir / 'semeval2016' / 'restaurants16_test_sb1.xml'),
            min_aspects=ref_args.get('min_aspects', 2),
            max_aspects=ref_args.get('max_aspects', 10),
            include_single_aspect=ref_args.get('include_single_aspect', True),
            virtual_aspect_mode=ref_args.get('virtual_aspect_mode', 'overall'),
        )

    elif dataset == 'lap16':
        train_s, val_s, test_s = load_semeval2016_data(
            str(raw_dir / 'semeval2016' / 'Laptops16_train_sb1.xml'),
            str(raw_dir / 'semeval2016' / 'Laptops16_test_sb1.xml'),
            min_aspects=ref_args.get('min_aspects', 2),
            max_aspects=ref_args.get('max_aspects', 10),
            include_single_aspect=ref_args.get('include_single_aspect', True),
            virtual_aspect_mode=ref_args.get('virtual_aspect_mode', 'overall'),
        )

    else:
        raise NotImplementedError(f'dataset {dataset!r} 尚未支援')

    _, val_loader, test_loader = create_multiaspect_dataloaders(
        train_samples=train_s,
        val_samples=val_s,
        test_samples=test_s,
        tokenizer=tokenizer,
        batch_size=16,
        max_text_len=ref_args.get('max_text_len', 128),
        max_aspect_len=ref_args.get('max_aspect_len', 32),
        max_num_aspects=ref_args.get('max_aspects', 10),
    )
    return val_loader, test_loader, len(val_s), len(test_s)


def _collect_logits(model, loader, device):
    """收集 raw logits 和 labels（展平為有效 aspect）。"""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='  推理', ascii=True, ncols=70, file=sys.stderr):
            pair_ids      = batch['pair_input_ids'].to(device)
            pair_mask     = batch['pair_attention_mask'].to(device)
            pair_type_ids = batch['pair_token_type_ids'].to(device)
            labels        = batch['labels']       # [B, A]  cpu
            aspect_mask   = batch['aspect_mask']  # [B, A]  cpu

            logits, _ = model(pair_ids, pair_mask, pair_type_ids, aspect_mask.to(device))
            logits = logits.cpu()                 # [B, A, 3]

            valid = aspect_mask & (labels != -100)
            if valid.any():
                all_logits.append(logits[valid])
                all_labels.append(labels[valid])

    return torch.cat(all_logits), torch.cat(all_labels)


def _evaluate_logits(logits, labels, gs=None):
    """
    套用 Logit Adjustment 後計算 Acc / Macro-F1 / AUC。

    Args:
        gs: {'neutral_boost': float, 'neg_suppress': float, 'pos_suppress': float}
    Returns:
        (acc, f1, auc)
    """
    logits = logits.clone()
    if gs:
        if gs.get('neutral_boost'): logits[:, 1] += gs['neutral_boost']
        if gs.get('neg_suppress'):  logits[:, 0] -= gs['neg_suppress']
        if gs.get('pos_suppress'):  logits[:, 2] -= gs['pos_suppress']

    preds     = torch.argmax(logits, dim=-1).numpy()
    labels_np = labels.numpy()
    acc = accuracy_score(labels_np, preds)
    f1  = f1_score(labels_np, preds, average='macro', zero_division=0)
    probs = F.softmax(logits, dim=-1).numpy()
    try:
        auc = roc_auc_score(labels_np, probs, multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    return acc, f1, auc


def _grid_search(val_logits, val_labels, gs_grid):
    """在 val logit 上 grid search 最佳 Logit Adjustment 值。"""
    best_f1, best_nb, best_ns, best_ps = -1.0, 0.0, 0.0, 0.0
    for nb in gs_grid:
        for ns in gs_grid:
            for ps in gs_grid:
                adj = val_logits.clone()
                if nb: adj[:, 1] += nb
                if ns: adj[:, 0] -= ns
                if ps: adj[:, 2] -= ps
                f1_v = f1_score(
                    val_labels.numpy(),
                    torch.argmax(adj, dim=-1).numpy(),
                    average='macro', zero_division=0,
                )
                if f1_v > best_f1:
                    best_f1, best_nb, best_ns, best_ps = f1_v, nb, ns, ps
    return {'neutral_boost': best_nb, 'neg_suppress': best_ns,
            'pos_suppress': best_ps, 'val_f1': best_f1}


def _build_report_text(dataset: str, seeds_info: list, results: dict) -> str:
    """生成 Ensemble 報告文字。"""
    disp   = DISPLAY_NAMES.get(dataset, dataset.upper())
    ts     = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    n      = results['n_seeds']
    solo   = results['solo_mean_f1']
    ew     = results['equal_weight']
    ea     = results['ensemble_adj']
    ps     = results['per_seed_adj']
    best_f1 = results['best_f1']
    best_st = results['best_strategy']

    def row(label, m, ref_f1):
        diff = (m['f1'] - ref_f1) * 100
        sign = '+' if diff >= 0 else ''
        return (f"  {label:<36} {m['acc']*100:>6.2f}%  "
                f"{m['f1']*100:>7.2f}%  {m['auc']*100:>7.2f}%  "
                f"{sign}{diff:.2f}%")

    gs     = ea.get('gs', {})
    gs_str = (f"nb={gs.get('neutral_boost',0):.1f} "
              f"ns={gs.get('neg_suppress',0):.1f} "
              f"ps={gs.get('pos_suppress',0):.1f}  "
              f"val_F1={gs.get('val_f1',0)*100:.2f}%")

    lines = [
        "=" * 80,
        f"HKGAN Ensemble 報告 - {disp} Dataset",
        "=" * 80,
        f"生成時間: {ts}",
        f"Seeds: {n} 個 checkpoint",
        "",
        f"Solo Mean F1: {solo*100:.2f}%",
        "-" * 80,
        f"  {'策略':<36} {'Acc':>7}  {'Macro-F1':>8}  {'AUC':>7}  {'vs Solo':>8}",
        "-" * 80,
        row("等重 Ensemble",               ew, solo),
        row("Ensemble + Logit Adj",        ea, solo),
        f"    (gs: {gs_str})",
        row("[方案十一] Per-Seed Adj → 等重", ps, solo),
        "-" * 80,
        f"  最佳策略: {best_st}",
        f"  最佳 F1:  {best_f1*100:.2f}%  (vs Solo {(best_f1-solo)*100:+.2f}%)",
        "=" * 80,
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# 公開函式
# ─────────────────────────────────────────────────────────────

def run_ensemble(
    dataset: str,
    results_root: Optional[Path] = None,
    save: bool = True,
    verbose: bool = True,
) -> dict:
    """
    對 results/improved/{dataset}/ 下所有 seed checkpoint 執行 Ensemble 推理。

    Args:
        dataset:      資料集名稱（'restaurants'|'laptops'|'mams'|'rest16'|'lap16'）
        results_root: results/ 根目錄（預設為 project_root/results）
        save:         是否存檔至 results/HKGAN_Ensemble_{dataset}.txt
        verbose:      是否印出過程訊息

    Returns:
        {
          'dataset', 'n_seeds', 'solo_mean_f1',
          'equal_weight':  {'acc', 'f1', 'auc'},
          'ensemble_adj':  {'acc', 'f1', 'auc', 'gs'},
          'per_seed_adj':  {'acc', 'f1', 'auc'},
          'best_f1', 'best_strategy', 'output_path'
        }
        失敗時回傳 {}，不拋例外。
    """
    try:
        return _run_ensemble_impl(dataset, results_root, save, verbose)
    except Exception as e:
        print(f"\n[WARN] Ensemble 執行失敗 ({dataset}): {e}\n")
        return {}


def _run_ensemble_impl(dataset, results_root, save, verbose):
    _ensure_path()
    from experiments.improved_models import create_improved_model
    import argparse as _ap

    project_root = _PROJECT_ROOT
    if results_root is None:
        results_root = project_root / 'results'
    results_root = Path(results_root)

    if dataset not in DATASET_MAP:
        raise ValueError(f"不支援的資料集: {dataset!r}，請選擇 {list(DATASET_MAP)}")

    exp_root = results_root / DATASET_MAP[dataset]
    if not exp_root.exists():
        print(f"[WARN] 找不到實驗目錄: {exp_root}")
        return {}

    seeds_info = _load_seeds_info(exp_root)
    if not seeds_info:
        print(f"[WARN] {dataset}: 找不到任何完成的 checkpoint")
        return {}

    if verbose:
        disp = DISPLAY_NAMES.get(dataset, dataset.upper())
        print(f"\n=== Ensemble: {disp} ({len(seeds_info)} seeds) ===")
        for i, s in enumerate(seeds_info):
            print(f"  seed{i}: {s['pt'].name}  solo_f1={s['solo_f1']*100:.2f}%")

    ref_args = seeds_info[0]['args']
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"\nDevice: {device}")

    val_loader, test_loader, n_val, n_test = _build_data_loaders(
        dataset, ref_args, project_root
    )
    if verbose:
        print(f"Val: {n_val} samples | Test: {n_test} samples\n")

    # 逐 seed 收集 logits
    all_val_logits_list  = []
    all_test_logits_list = []

    for i, s in enumerate(seeds_info):
        if verbose:
            print(f"[{i+1}/{len(seeds_info)}] 載入 {s['pt'].name}")
        model_args = _ap.Namespace(**s['args'])
        model_args.bert_model = _resolve_bert_path(
            getattr(model_args, 'bert_model', 'bert-base-uncased'), project_root
        )
        model = create_improved_model(
            model_type=getattr(model_args, 'improved', 'hkgan'),
            args=model_args,
            num_classes=3,
        ).to(device)
        model.load_state_dict(
            torch.load(s['pt'], map_location=device), strict=False
        )
        val_logits,  val_labels  = _collect_logits(model, val_loader,  device)
        test_logits, test_labels = _collect_logits(model, test_loader, device)
        all_val_logits_list.append(val_logits)
        all_test_logits_list.append(test_logits)
        del model
        torch.cuda.empty_cache()

    # 等重平均
    avg_val_logits  = torch.stack(all_val_logits_list).mean(dim=0)
    avg_test_logits = torch.stack(all_test_logits_list).mean(dim=0)

    gs_grid = np.arange(0.0, 1.3, 0.1).round(1).tolist()

    # 策略 1：等重 Ensemble
    a1, f1_1, u1 = _evaluate_logits(avg_test_logits.clone(), test_labels)

    # 策略 2：Ensemble + Ensemble-level Logit Adj
    if verbose:
        print('\n[Ensemble Logit Adj] 在 ensemble val logit 上搜尋最佳調整值 ...')
    best_ens_gs = _grid_search(avg_val_logits, val_labels, gs_grid)
    if verbose:
        print(f"  最佳 gs: nb={best_ens_gs['neutral_boost']} "
              f"ns={best_ens_gs['neg_suppress']} "
              f"ps={best_ens_gs['pos_suppress']}  "
              f"val_F1={best_ens_gs['val_f1']*100:.2f}%")
    a2, f1_2, u2 = _evaluate_logits(avg_test_logits.clone(), test_labels, gs=best_ens_gs)

    # 策略 3：Per-Seed Logit Adj → 等重平均（方案十一）
    if verbose:
        print('\n[Per-Seed Logit Adj] 對每個 seed 獨立搜尋調整值 ...')
    per_seed_adj_list = []
    for i, (vl, tl) in enumerate(zip(all_val_logits_list, all_test_logits_list)):
        seed_gs = _grid_search(vl, val_labels, gs_grid)
        tl_adj = tl.clone()
        if seed_gs['neutral_boost']: tl_adj[:, 1] += seed_gs['neutral_boost']
        if seed_gs['neg_suppress']:  tl_adj[:, 0] -= seed_gs['neg_suppress']
        if seed_gs['pos_suppress']:  tl_adj[:, 2] -= seed_gs['pos_suppress']
        per_seed_adj_list.append(tl_adj)
        if verbose:
            print(f"  seed{i}: nb={seed_gs['neutral_boost']} "
                  f"ns={seed_gs['neg_suppress']} "
                  f"ps={seed_gs['pos_suppress']}  "
                  f"val_F1={seed_gs['val_f1']*100:.2f}%")
    per_seed_avg = torch.stack(per_seed_adj_list).mean(dim=0)
    a3, f1_3, u3 = _evaluate_logits(per_seed_avg.clone(), test_labels)

    # 彙總
    solo_mean = float(np.mean([s['solo_f1'] for s in seeds_info]))

    strategies = {
        '等重 Ensemble':               (a1, f1_1, u1),
        'Ensemble + Logit Adj':        (a2, f1_2, u2),
        '[方案十一] Per-Seed Adj → 等重': (a3, f1_3, u3),
    }
    best_strategy = max(strategies, key=lambda k: strategies[k][1])
    best_f1 = strategies[best_strategy][1]

    results = {
        'dataset':      dataset,
        'n_seeds':      len(seeds_info),
        'solo_mean_f1': solo_mean,
        'equal_weight': {'acc': a1, 'f1': f1_1, 'auc': u1},
        'ensemble_adj': {'acc': a2, 'f1': f1_2, 'auc': u2, 'gs': best_ens_gs},
        'per_seed_adj': {'acc': a3, 'f1': f1_3, 'auc': u3},
        'best_f1':      best_f1,
        'best_strategy': best_strategy,
        'output_path':  None,
    }

    # 印出彙總
    if verbose:
        print('\n' + '=' * 65)
        print(f"{'方法':<38} {'Acc':>7} {'F1':>8} {'AUC':>8}")
        print('-' * 65)
        print(f"  {'各 seed 單獨（平均）':<36} {'—':>7} {solo_mean*100:>7.2f}% {'—':>8}")
        for name, (a, f, u) in strategies.items():
            diff = (f - solo_mean) * 100
            sign = '+' if diff >= 0 else ''
            print(f"  {name:<36} {a*100:>6.2f}% {f*100:>7.2f}% {u*100:>7.2f}%  ({sign}{diff:.2f}%)")
        print('=' * 65)
        print(f"  最佳 F1: {best_f1*100:.2f}%  vs 單 seed 平均: {solo_mean*100:.2f}%  "
              f"提升: {(best_f1-solo_mean)*100:+.2f}%")

    # 存檔
    if save:
        report_text = _build_report_text(dataset, seeds_info, results)
        output_path = results_root / f"HKGAN_Ensemble_{dataset}.txt"
        output_path.write_text(report_text, encoding='utf-8')
        results['output_path'] = output_path
        if verbose:
            print(f"\n報告已保存至: {output_path}")

    return results


def run_ensemble_and_save(
    dataset: str,
    results_root: Optional[Path] = None,
) -> Optional[Path]:
    """
    `run_ensemble()` 的薄包裝，固定 save=True，回傳輸出路徑（或 None）。
    供 run_experiments.py 一行呼叫用。
    """
    result = run_ensemble(dataset, results_root=results_root, save=True, verbose=True)
    return result.get('output_path') if result else None
