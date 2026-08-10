"""
XAI transfer diagnostic for into-SPEDIA zero-shot cells (Contribution 2).

Trains DaySupervisedGNN (A3 / full components) on CERT source only, then
attributes the positive-class logit w.r.t. the aligned user-day feature vector
x via Integrated Gradients, with frozen per-user-day memory s_u:

    f(x) = head(proj(cat([x, s_u])))

Compares normalized |IG| attribution profiles on SOURCE (in-distribution) vs
SPEDIA TARGET user-days. No target labels enter attributions.

Cells (default): cert52->spedia (primary), cert42->spedia (secondary).
Seeds: SEEDS_DEFAULT = (0, 1, 2, 3, 4). Reuses results/cache/transfer_5d/.

Outputs:
  results/xai_transfer_diagnostic.csv
    columns: cell, feature, alpha_src, alpha_tgt, delta

Does NOT modify training, the transfer matrix runner, or loaders.

Usage (from repo root):
  python -m scripts.run_xai_transfer
  python -m scripts.run_xai_transfer --cells "cert52->spedia" --seeds 0,1
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.features import FEATURE_COLUMNS
from src.train.day_supervised_gnn import DaySupervisedGNN
from src.train.multiseed_transfer import gnn_kw_for_source
from src.utils.seed import set_seed
from scripts.run_transfer_matrix_5domain import (
    SEEDS_DEFAULT,
    cache_events,
    cache_features,
    load_events,
    load_feat,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("xai_transfer")

OUT = ROOT / "results"
OUT_CSV = OUT / "xai_transfer_diagnostic.csv"

CELLS_DEFAULT: Tuple[Tuple[str, str], ...] = (
    ("cert52", "spedia"),  # primary
    ("cert42", "spedia"),  # secondary
)

IG_STEPS = 50
IG_BATCH = 4096


def _parse_cells(spec: str) -> List[Tuple[str, str]]:
    cells: List[Tuple[str, str]] = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "->" not in part:
            raise SystemExit(f"--cells entry must be src->tgt, got {part!r}")
        s, t = (x.strip() for x in part.split("->", 1))
        cells.append((s, t))
    return cells


def _parse_seeds(spec: Optional[str]) -> Tuple[int, ...]:
    if not spec:
        return tuple(SEEDS_DEFAULT)
    return tuple(int(x.strip()) for x in spec.split(",") if x.strip())


def _try_captum_ig():
    try:
        from captum.attr import IntegratedGradients  # type: ignore
        return IntegratedGradients
    except Exception:
        return None


def collect_x_su(
    det: DaySupervisedGNN,
    df: pd.DataFrame,
    feat: pd.DataFrame,
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """Stream events day-by-day; return aligned (x, s_u) for every user-day.

    s_u is detached frozen encoder memory at end-of-day (same as scoring).
    x is scaler-transformed FEATURE_COLUMNS (same as training).
    """
    torch = det._torch()
    det.set_full_features(feat)
    feat_ud = det._features_for(df)
    ts = det._tensors(df, feat=feat_ud)
    det.model.set_graph(ts["n_users"], ts["n_hosts"])
    det.model.reset_memory()
    det.model.eval()
    det.head.eval()
    if det.proj is not None:
        det.proj.eval()

    xs: List = []
    sus: List = []
    with torch.no_grad():
        for day_val in pd.unique(ts["day"]):
            day_idx = np.flatnonzero(ts["day"] == day_val)
            if day_idx.size == 0:
                continue
            det._maybe_reset_memory_for_day()
            det._stream_day(ts, day_idx)
            ud = det._ud_for_day(ts, day_val)
            if ud is None or ud["x"] is None:
                continue
            s_u = det.model.user_mem.memory[ud["u"]].detach()
            xs.append(ud["x"].detach().cpu())
            sus.append(s_u.cpu())

    if not xs:
        raise RuntimeError("collect_x_su: no user-day rows collected")
    return torch.cat(xs, dim=0), torch.cat(sus, dim=0)


def _manual_ig(
    proj,
    head,
    x: "torch.Tensor",
    s_u: "torch.Tensor",
    *,
    steps: int = IG_STEPS,
    batch_size: int = IG_BATCH,
) -> np.ndarray:
    """50-step Riemann IG of positive-class logit w.r.t. x; baseline = 0."""
    import torch

    n = x.shape[0]
    baseline = torch.zeros_like(x)
    attrs = torch.zeros_like(x)
    delta = x - baseline

    proj.eval()
    head.eval()
    # Attribute w.r.t. x only; do not accumulate grads into frozen weights.
    req = [p.requires_grad for p in list(proj.parameters()) + list(head.parameters())]
    for p in list(proj.parameters()) + list(head.parameters()):
        p.requires_grad_(False)
    try:
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x_b = x[start:end]
            s_b = s_u[start:end]
            base_b = baseline[start:end]
            delta_b = delta[start:end]
            grad_acc = torch.zeros_like(x_b)
            for k in range(1, steps + 1):
                alpha = float(k) / float(steps)
                x_k = (base_b + alpha * delta_b).detach().requires_grad_(True)
                logits = head(proj(torch.cat([x_k, s_b], dim=-1)))
                g = torch.autograd.grad(logits.sum(), x_k)[0]
                grad_acc = grad_acc + g.detach()
            avg_grad = grad_acc / float(steps)
            attrs[start:end] = delta_b * avg_grad
    finally:
        for p, r in zip(list(proj.parameters()) + list(head.parameters()), req):
            p.requires_grad_(r)
    return attrs.detach().cpu().numpy()


def _captum_ig(
    IntegratedGradients,
    proj,
    head,
    x: "torch.Tensor",
    s_u: "torch.Tensor",
    *,
    steps: int = IG_STEPS,
    batch_size: int = IG_BATCH,
) -> np.ndarray:
    import torch
    import torch.nn as nn

    class _F(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = proj
            self.head = head
            self._s: Optional[torch.Tensor] = None

        def forward(self, x_in: torch.Tensor) -> torch.Tensor:
            assert self._s is not None
            return self.head(self.proj(torch.cat([x_in, self._s], dim=-1)))

    f = _F()
    f.eval()
    ig = IntegratedGradients(f)
    n = x.shape[0]
    out = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_b = x[start:end]
        f._s = s_u[start:end]
        bas = torch.zeros_like(x_b)
        attr = ig.attribute(x_b, baselines=bas, n_steps=steps)
        out.append(attr.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def attribute_domain(
    det: DaySupervisedGNN,
    df: pd.DataFrame,
    feat: pd.DataFrame,
    *,
    steps: int = IG_STEPS,
    max_rows: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return attr matrix [N, K] for one domain under a frozen source-trained model."""
    torch = det._torch()
    x_cpu, s_cpu = collect_x_su(det, df, feat)
    if max_rows is not None and len(x_cpu) > max_rows:
        n0 = len(x_cpu)
        rng = np.random.default_rng(seed)
        idx = rng.choice(n0, size=max_rows, replace=False)
        idx.sort()
        x_cpu = x_cpu[idx]
        s_cpu = s_cpu[idx]
        log.info("IG subsample %d -> %d (seed=%d)", n0, max_rows, seed)

    device = det._device
    x = x_cpu.to(device)
    s_u = s_cpu.to(device).detach()  # frozen; no grad through memory

    IGCls = _try_captum_ig()
    if IGCls is not None:
        log.info("IG via captum (steps=%d, n=%d)", steps, x.shape[0])
        attr = _captum_ig(
            IGCls, det.proj, det.head, x, s_u, steps=steps, batch_size=IG_BATCH,
        )
    else:
        log.info("IG via manual Riemann (steps=%d, n=%d; captum unavailable)",
                 steps, x.shape[0])
        attr = _manual_ig(
            det.proj, det.head, x, s_u, steps=steps, batch_size=IG_BATCH,
        )
    del x, s_u, x_cpu, s_cpu
    return attr


def alpha_profile(attr: np.ndarray) -> np.ndarray:
    """alpha_k = sum_n |attr_{n,k}| / sum_{n,l} |attr_{n,l}|."""
    abs_a = np.abs(attr.astype(np.float64))
    col = abs_a.sum(axis=0)
    tot = float(col.sum())
    if tot <= 0:
        return np.full(attr.shape[1], 1.0 / attr.shape[1], dtype=np.float64)
    return col / tot


def spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        rho, _ = spearmanr(a, b)
        return float(rho)
    except Exception:
        # rank-correlation fallback without scipy
        ra = pd.Series(a).rank(method="average").to_numpy()
        rb = pd.Series(b).rank(method="average").to_numpy()
        ra = ra - ra.mean()
        rb = rb - rb.mean()
        den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
        return float((ra * rb).sum() / den) if den > 0 else float("nan")


def topk_overlap(alpha_src: np.ndarray, alpha_tgt: np.ndarray, k: int = 10) -> int:
    top_s = set(np.argsort(-alpha_src)[:k])
    top_t = set(np.argsort(-alpha_tgt)[:k])
    return int(len(top_s & top_t))


def transferable_scores(alpha_src: np.ndarray, alpha_tgt: np.ndarray) -> np.ndarray:
    """High in BOTH domains with similar rank."""
    k = len(alpha_src)
    # ranks: 1 = highest alpha
    r_s = pd.Series(alpha_src).rank(ascending=False, method="average").to_numpy()
    r_t = pd.Series(alpha_tgt).rank(ascending=False, method="average").to_numpy()
    rank_sim = 1.0 - np.abs(r_s - r_t) / max(k - 1, 1)
    return np.minimum(alpha_src, alpha_tgt) * rank_sim


def run_cell(
    source: str,
    target: str,
    seeds: Sequence[int],
    *,
    ig_steps: int = IG_STEPS,
    max_src_ig: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    cell = f"{source}->{target}"
    log.info("==== XAI cell %s seeds=%s ====", cell, list(seeds))

    for name in (source, target):
        cache_features(name, force=False)
        cache_events(name, force=False)

    feat_s = load_feat(source)
    feat_t = load_feat(target)
    df_s = load_events(source)
    df_t = load_events(target)
    log.info(
        "%s feat rows=%d / events=%d | %s feat rows=%d / events=%d",
        source, len(feat_s), len(df_s), target, len(feat_t), len(df_t),
    )

    alphas_src: List[np.ndarray] = []
    alphas_tgt: List[np.ndarray] = []

    for seed in seeds:
        log.info("--- %s seed=%d ---", cell, seed)
        set_seed(seed)
        kw = gnn_kw_for_source(source, seed, use_dann=False)
        kw["variant"] = "A3"  # ≡ BC concat; all component toggles default ON
        det = DaySupervisedGNN(**kw)
        det.set_full_features(feat_s)
        det.fit(df_s)  # zero-shot: source only

        # (a) SOURCE in-distribution attributions
        attr_s = attribute_domain(
            det, df_s, feat_s, steps=ig_steps, max_rows=max_src_ig, seed=seed,
        )
        alphas_src.append(alpha_profile(attr_s))
        del attr_s
        gc.collect()

        # (b) TARGET (SPEDIA) attributions — same frozen model; no target labels
        attr_t = attribute_domain(
            det, df_t, feat_t, steps=ig_steps, max_rows=None, seed=seed,
        )
        alphas_tgt.append(alpha_profile(attr_t))
        del attr_t, det
        gc.collect()

    alpha_src = np.mean(np.stack(alphas_src, axis=0), axis=0)
    alpha_tgt = np.mean(np.stack(alphas_tgt, axis=0), axis=0)
    delta = alpha_src - alpha_tgt

    t_xai = float(1.0 - 0.5 * np.abs(alpha_src - alpha_tgt).sum())
    rho = spearman_rho(alpha_src, alpha_tgt)
    overlap10 = topk_overlap(alpha_src, alpha_tgt, k=10)

    names = list(FEATURE_COLUMNS)
    rows = pd.DataFrame({
        "cell": cell,
        "feature": names,
        "alpha_src": alpha_src,
        "alpha_tgt": alpha_tgt,
        "delta": delta,
    })

    xfer = transferable_scores(alpha_src, alpha_tgt)
    top_xfer_idx = np.argsort(-xfer)[:5]
    top_src_idx = np.argsort(-delta)[:5]  # SOURCE-SPECIFIC: high src, collapses on tgt

    summary = {
        "cell": cell,
        "T_XAI": t_xai,
        "spearman_rho": rho,
        "top10_overlap": overlap10,
        "transferable": [names[i] for i in top_xfer_idx],
        "source_specific": [names[i] for i in top_src_idx],
        "transferable_detail": [
            (names[i], float(alpha_src[i]), float(alpha_tgt[i]), float(xfer[i]))
            for i in top_xfer_idx
        ],
        "source_specific_detail": [
            (names[i], float(alpha_src[i]), float(alpha_tgt[i]), float(delta[i]))
            for i in top_src_idx
        ],
    }
    return rows, summary


def print_summary(s: dict) -> None:
    print()
    print("=" * 72)
    print(f"Cell {s['cell']}")
    print(f"  T_XAI          = {s['T_XAI']:.4f}")
    print(f"  Spearman rho   = {s['spearman_rho']:.4f}")
    print(f"  Top-10 overlap = {s['top10_overlap']}/10")
    print("  Top-5 TRANSFERABLE (high both + similar rank):")
    for name, a_s, a_t, sc in s["transferable_detail"]:
        print(f"    {name:22s}  α_src={a_s:.4f}  α_tgt={a_t:.4f}  score={sc:.4f}")
    print("  Top-5 SOURCE-SPECIFIC (α_src − α_tgt large):")
    for name, a_s, a_t, d in s["source_specific_detail"]:
        print(f"    {name:22s}  α_src={a_s:.4f}  α_tgt={a_t:.4f}  Δ={d:+.4f}")
    print("=" * 72)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="XAI transfer diagnostic (CERT→SPEDIA)")
    p.add_argument(
        "--cells",
        default="cert52->spedia,cert42->spedia",
        help='Comma-separated src->tgt (default: cert52->spedia,cert42->spedia)',
    )
    p.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds (default: 0,1,2,3,4)",
    )
    p.add_argument("--ig-steps", type=int, default=IG_STEPS)
    p.add_argument(
        "--max-src-ig",
        type=int,
        default=None,
        help="Optional subsample of SOURCE user-days for IG (default: all)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_CSV,
        help=f"Output CSV (default: {OUT_CSV})",
    )
    args = p.parse_args(argv)

    cells = _parse_cells(args.cells)
    seeds = _parse_seeds(args.seeds)
    OUT.mkdir(parents=True, exist_ok=True)

    log.info(
        "XAI transfer diagnostic | cells=%s seeds=%s ig_steps=%d captum=%s",
        cells, list(seeds), args.ig_steps,
        "yes" if _try_captum_ig() is not None else "no (manual Riemann)",
    )

    all_rows: List[pd.DataFrame] = []
    for source, target in cells:
        rows, summary = run_cell(
            source, target, seeds,
            ig_steps=args.ig_steps,
            max_src_ig=args.max_src_ig,
        )
        all_rows.append(rows)
        print_summary(summary)

    out = pd.concat(all_rows, ignore_index=True)
    out = out[["cell", "feature", "alpha_src", "alpha_tgt", "delta"]]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    log.info("Wrote %s (%d rows)", args.out, len(out))
    print(f"\nWrote {args.out} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
