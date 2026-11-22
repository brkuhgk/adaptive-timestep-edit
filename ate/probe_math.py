"""Statistics of the prompt-disagreement curve d(t) and rules that turn it into a timestep.

Intuition (see report/report.md, Section 3):
  ChordEdit's transport step moves the source latent by u(t) ~ E_eps[ x0(z_t, c_tgt) - x0(z_t, c_src) ].
  At low noise both predictions stay near x_src, so ||u(t)|| is small: the edit is "not reachable" yet.
  As t grows, the model is free to move low-frequency content and ||u(t)|| rises, then saturates once the
  requested change is expressible; extra noise past that point mostly buys unrelated drift (background loss).
  The "knee" rule therefore picks the smallest t whose disagreement reaches a fraction tau of its maximum:
  the minimal-noise (lowest-energy) timestep at which the edit is already expressible.
"""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def curve_statistics(deltas: torch.Tensor, lowpass: int = 4) -> Dict[str, np.ndarray]:
    """deltas: [T, K, C, H, W] -> per-t statistics.

    s     : RMS magnitude of the disagreement (mean over noises)
    rho   : mean pairwise cosine similarity of the disagreement across noises (direction consistency)
    lf    : share of the disagreement energy that survives a lowpass (avg-pool) filter (coarse vs fine)
    s_bar : RMS of the noise-averaged disagreement (what ChordEdit actually applies)
    """
    d = deltas.float()
    T, K = d.shape[:2]
    flat = d.reshape(T, K, -1)
    rms = flat.pow(2).mean(-1).sqrt()                       # [T, K]
    s = rms.mean(1)
    s_bar = flat.mean(1).pow(2).mean(-1).sqrt()
    if K >= 2:
        unit = F.normalize(flat, dim=-1)
        sims = unit @ unit.transpose(1, 2)                  # [T, K, K]
        off = sims.sum((1, 2)) - sims.diagonal(dim1=1, dim2=2).sum(-1)
        rho = off / (K * (K - 1))
    else:
        rho = torch.full((T,), float("nan"))
    k = max(1, int(lowpass))
    pooled = F.avg_pool2d(d.reshape(T * K, *d.shape[2:]), k) if min(d.shape[-2:]) >= k else d.reshape(T * K, *d.shape[2:])
    lf_energy = pooled.pow(2).mean((1, 2, 3)).reshape(T, K)
    tot_energy = d.reshape(T * K, -1).pow(2).mean(-1).reshape(T, K).clamp(min=1e-12)
    lf = (lf_energy / tot_energy).mean(1)
    return {
        "s": s.numpy(),
        "s_bar": s_bar.numpy(),
        "rho": rho.numpy(),
        "lf": lf.numpy(),
    }


def knee_select(ts: Sequence[float], s: Sequence[float], tau: float) -> float:
    """Smallest t (in increasing order) whose s(t)/max(s) >= tau."""
    ts = np.asarray(ts, dtype=float)
    s = np.asarray(s, dtype=float)
    order = np.argsort(ts)
    ts, s = ts[order], s[order]
    peak = np.nanmax(s) if np.isfinite(s).any() else 0.0
    if peak <= 0:
        return float(ts[-1])
    ok = np.where(s / peak >= tau)[0]
    return float(ts[ok[0]]) if ok.size else float(ts[-1])


def consistency_select(ts: Sequence[float], s: Sequence[float], rho: Sequence[float]) -> float:
    """argmax_t s(t) * max(rho(t), 0): strong *and* noise-consistent disagreement."""
    ts = np.asarray(ts, dtype=float)
    score = np.asarray(s, dtype=float) * np.clip(np.nan_to_num(np.asarray(rho, dtype=float), nan=1.0), 0, None)
    return float(ts[int(np.nanargmax(score))])


RULES = ("knee", "consistency")
