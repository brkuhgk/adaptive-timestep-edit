"""Oracles, timestep predictors, and their evaluation. Everything is tuned on dev and reported on test."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import EditConfig
from .objective import EVAL_METRICS, Standardizer
from .probe_math import consistency_select, knee_select


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------
def paired_bootstrap(diff: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    diff = np.asarray(diff, dtype=float)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, diff.size, size=(n, diff.size))
    means = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# --------------------------------------------------------------------------------------
# Data container
# --------------------------------------------------------------------------------------
@dataclass
class Study:
    metrics_long: pd.DataFrame          # sample_id, config_id, seed, <EVAL_METRICS>
    split: Dict[str, str]               # sample_id -> dev | test
    category: Dict[str, int]            # sample_id -> PIE edit type
    reference_id: str
    edit_weight: float = 1.0
    preserve_weight: float = 1.0
    standardizer: Standardizer = field(init=False)
    J_long: pd.DataFrame = field(init=False)
    J: pd.DataFrame = field(init=False)          # sample x config, seed-averaged score
    M: pd.DataFrame = field(init=False)          # (sample, config) -> seed-averaged metrics

    def __post_init__(self) -> None:
        df = self.metrics_long.copy()
        df["split"] = df["sample_id"].map(self.split)
        dev_rows = df[df["split"] == "dev"]
        self.standardizer = Standardizer.fit(dev_rows if len(dev_rows) else df)
        df["J"] = self.standardizer.score(df, self.edit_weight, self.preserve_weight).to_numpy()
        df["J_edit"] = self.standardizer.edit_score(df).to_numpy()
        df["J_preserve"] = self.standardizer.preserve_score(df).to_numpy()
        self.J_long = df
        cols = [c for c in EVAL_METRICS if c in df.columns] + ["J", "J_edit", "J_preserve"]
        self.M = df.groupby(["sample_id", "config_id"])[cols].mean()
        self.J = self.M["J"].unstack("config_id")

    # ----- helpers -----
    def ids(self, which: str) -> List[str]:
        return sorted(s for s in self.J.index if self.split.get(s) == which)

    @property
    def configs(self) -> List[str]:
        return list(self.J.columns)

    def zero_delta_configs(self) -> List[str]:
        return [c for c in self.configs if EditConfig.from_id(c).t_delta == 0.0]

    # ----- selection rules fitted on dev -----
    def global_best(self, candidates: Optional[Sequence[str]] = None) -> str:
        cand = list(candidates or self.configs)
        return str(self.J.loc[self.ids("dev"), cand].mean().idxmax())

    def category_table(self, candidates: Optional[Sequence[str]] = None) -> Dict[int, str]:
        cand = list(candidates or self.configs)
        dev = self.ids("dev")
        fallback = self.global_best(cand)
        table: Dict[int, str] = {}
        for cat in sorted(set(self.category.values())):
            ids = [s for s in dev if self.category.get(s) == cat]
            table[cat] = str(self.J.loc[ids, cand].mean().idxmax()) if ids else fallback
        return table

    def t_end_given_t_start(self) -> Dict[float, str]:
        """For each t_start, the dev-best (t_end) among delta=0 configs -> config id."""
        dev = self.ids("dev")
        out: Dict[float, str] = {}
        by_ts: Dict[float, List[str]] = {}
        for c in self.zero_delta_configs():
            by_ts.setdefault(EditConfig.from_id(c).t_start, []).append(c)
        for ts, cands in by_ts.items():
            out[ts] = str(self.J.loc[dev, cands].mean().idxmax())
        return out

    # ----- oracles on any id set -----
    def per_image_oracle(self, ids: Sequence[str]) -> Dict[str, str]:
        return {s: str(self.J.loc[s].idxmax()) for s in ids}

    def held_out_seed_oracle(self, ids: Sequence[str]) -> Optional[pd.Series]:
        """Pick the best config on one seed and score it on the other seeds (removes winner's-curse optimism)."""
        seeds = sorted(self.J_long["seed"].unique())
        if len(seeds) < 2:
            return None
        df = self.J_long[self.J_long["sample_id"].isin(ids)]
        sel = df[df["seed"] == seeds[0]].pivot_table(index="sample_id", columns="config_id", values="J")
        ev = df[df["seed"] != seeds[0]].pivot_table(index="sample_id", columns="config_id", values="J", aggfunc="mean")
        choice = sel.idxmax(axis=1)
        return pd.Series({s: ev.loc[s, c] for s, c in choice.items() if s in ev.index})

    # ----- evaluation of any choice -----
    def score_choice(self, choice: Dict[str, str]) -> pd.DataFrame:
        rows = [self.M.loc[(s, c)].to_dict() | {"sample_id": s, "config_id": c} for s, c in choice.items()]
        return pd.DataFrame(rows).set_index("sample_id")


# --------------------------------------------------------------------------------------
# Predictors
# --------------------------------------------------------------------------------------
def probe_choices(study: Study, probe_long: pd.DataFrame, rule: str, tau: Optional[float],
                  ids: Sequence[str]) -> Dict[str, str]:
    te_map = study.t_end_given_t_start()
    available = sorted(te_map)
    out: Dict[str, str] = {}
    for sid, g in probe_long[probe_long["sample_id"].isin(ids)].groupby("sample_id"):
        g = g.sort_values("t")
        if rule == "knee":
            t = knee_select(g["t"], g["s_bar"], float(tau))
        elif rule == "consistency":
            t = consistency_select(g["t"], g["s"], g["rho"])
        else:
            raise ValueError(rule)
        nearest = min(available, key=lambda a: abs(a - t))
        out[sid] = te_map[nearest]
    return out


def tune_probe(study: Study, probe_long: pd.DataFrame, taus: Sequence[float]) -> tuple[str, Optional[float], pd.DataFrame]:
    dev = study.ids("dev")
    rows = []
    for tau in taus:
        ch = probe_choices(study, probe_long, "knee", tau, dev)
        rows.append({"rule": "knee", "tau": tau, "dev_J": float(np.mean([study.J.loc[s, c] for s, c in ch.items()]))})
    if probe_long["rho"].notna().any():
        ch = probe_choices(study, probe_long, "consistency", None, dev)
        rows.append({"rule": "consistency", "tau": np.nan, "dev_J": float(np.mean([study.J.loc[s, c] for s, c in ch.items()]))})
    table = pd.DataFrame(rows).sort_values("dev_J", ascending=False)
    best = table.iloc[0]
    return str(best["rule"]), (None if pd.isna(best["tau"]) else float(best["tau"])), table


def classifier_choices(study: Study, predicted_category: Dict[str, int], ids: Sequence[str]) -> Dict[str, str]:
    table = study.category_table()
    fallback = study.global_best()
    return {s: table.get(int(predicted_category[s]), fallback) if s in predicted_category else fallback for s in ids}


# --------------------------------------------------------------------------------------
# Main comparison
# --------------------------------------------------------------------------------------
@dataclass
class MethodResult:
    name: str
    choice: Dict[str, str]
    extra_unet_calls: float
    note: str = ""


def compare(study: Study, methods: Sequence[MethodResult], n_boot: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, Dict]:
    test = study.ids("test")
    ref = {s: study.reference_id for s in test}
    J_ref = study.J.loc[test, study.reference_id]
    oracle_in = study.per_image_oracle(test)
    J_or_in = pd.Series({s: study.J.loc[s, c] for s, c in oracle_in.items()})
    J_or_ho = study.held_out_seed_oracle(test)
    gb = study.global_best()
    J_gb = study.J.loc[test, gb]

    rows = []
    all_methods = [
        MethodResult("Fixed: paper default", ref, 0, study.reference_id),
        MethodResult("Fixed: dev global best", {s: gb for s in test}, 0, gb),
        *methods,
        MethodResult("Oracle: per image (upper bound)", oracle_in, float("nan"), "in-sample, optimistic"),
    ]
    for m in all_methods:
        sc = study.score_choice({s: m.choice[s] for s in test if s in m.choice})
        diff = (sc["J"] - J_ref.loc[sc.index]).to_numpy()
        mean, lo, hi = paired_bootstrap(diff, n_boot, seed)
        denom_in = float(J_or_in.mean() - J_ref.mean())
        denom_gb = float(J_or_in.mean() - J_gb.mean())
        denom_in = denom_in if abs(denom_in) > 1e-3 else 0.0   # no meaningful gap to close
        denom_gb = denom_gb if abs(denom_gb) > 1e-3 else 0.0
        row = {
            "method": m.name,
            "J": float(sc["J"].mean()),
            "dJ_vs_default": mean,
            "dJ_lo": lo,
            "dJ_hi": hi,
            "gap_closed_vs_default_%": 100.0 * (float(sc["J"].mean()) - float(J_ref.mean())) / denom_in if denom_in else np.nan,
            "gap_closed_vs_globalbest_%": 100.0 * (float(sc["J"].mean()) - float(J_gb.mean())) / denom_gb if denom_gb else np.nan,
            "extra_unet_calls": m.extra_unet_calls,
            "note": m.note,
        }
        for col in ["clip_similarity_target_image", "clip_similarity_target_image_edit_part", "structure_distance",
                    "psnr_unedit_part", "lpips_unedit_part", "ssim_unedit_part", "J_edit", "J_preserve"]:
            if col in sc.columns:
                row[col] = float(sc[col].mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    info = {
        "n_test": len(test),
        "n_dev": len(study.ids("dev")),
        "reference_config": study.reference_id,
        "global_best_config": gb,
        "oracle_in_sample_J": float(J_or_in.mean()),
        "oracle_held_out_seed_J": None if J_or_ho is None else float(J_or_ho.mean()),
        "reference_J": float(J_ref.mean()),
    }
    if J_or_ho is not None:
        denom = info["oracle_held_out_seed_J"] - info["reference_J"]
        table["gap_closed_vs_default_heldout_oracle_%"] = 100.0 * (table["J"] - info["reference_J"]) / denom if denom else np.nan
    return table, info


def category_best_tstart(study: Study, ids: Sequence[str]) -> pd.DataFrame:
    """Mean J by (category, t_start) using each t_start's dev-best t_end; used for the heatmap."""
    te_map = study.t_end_given_t_start()
    rows = []
    for ts, cfg in sorted(te_map.items()):
        for s in ids:
            rows.append({"category": study.category[s], "t_start": ts, "J": study.J.loc[s, cfg]})
    df = pd.DataFrame(rows)
    return df.groupby(["category", "t_start"])["J"].mean().unstack("t_start")
