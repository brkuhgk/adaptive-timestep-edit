"""Figures for the report: category x t_start heatmap, gap-closed bars, probe curves by category."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .pie import CATEGORY_NAMES  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def heatmap_category_tstart(table: pd.DataFrame, path: Path) -> None:
    """Rows: edit category. Columns: t_start. Cell: mean J (row-centered so the best t per category pops)."""
    centered = table.sub(table.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    vmax = float(np.nanmax(np.abs(centered.to_numpy()))) or 1.0
    im = ax.imshow(centered.to_numpy(), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(table.columns)), [f"{c:.2f}" for c in table.columns])
    ax.set_yticks(range(len(table.index)), [CATEGORY_NAMES.get(int(i), str(i)) for i in table.index])
    best = np.nanargmax(table.to_numpy(), axis=1)
    for r, c in enumerate(best):
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=1.6, ec="black"))
    ax.set_xlabel("transport timestep t_start (higher = more noise)")
    ax.set_title("Edit score by category and t_start (row-centered; box = best)")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="ΔJ vs row mean")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def gap_closed_bars(table: pd.DataFrame, path: Path) -> None:
    df = table[~table["method"].str.startswith("Fixed: paper")].copy()
    fig, ax = plt.subplots(figsize=(6.4, 0.45 * len(df) + 1.2))
    y = np.arange(len(df))
    err = np.vstack([df["dJ_vs_default"] - df["dJ_lo"], df["dJ_hi"] - df["dJ_vs_default"]])
    colors = ["#9aa5b1" if m.startswith(("Fixed", "Oracle")) else "#2b6cb0" for m in df["method"]]
    ax.barh(y, df["dJ_vs_default"], xerr=err, color=colors, capsize=2, height=0.6)
    span = float(np.nanmax(np.abs(df[["dJ_lo", "dJ_hi"]].to_numpy()))) or 1.0
    for yi, (hi, g) in enumerate(zip(df["dJ_hi"], df["gap_closed_vs_default_%"])):
        if np.isfinite(g):
            ax.text(max(hi, 0) + 0.03 * span, yi, f"{g:.0f}% of gap", va="center", fontsize=7.5, color="#333")
    ax.set_xlim(min(0, float(np.nanmin(df["dJ_lo"]))) - 0.05 * span, 1.35 * span)
    ax.set_yticks(y, df["method"])
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Improvement in edit score J over the paper default\n(test split, 95% paired bootstrap CI)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def probe_curves(probe_long: pd.DataFrame, category: dict, path: Path) -> None:
    df = probe_long.copy()
    df["category"] = df["sample_id"].map(category)
    df["s_norm"] = df["s_bar"] / df.groupby("sample_id")["s_bar"].transform("max")
    mean = df.groupby(["category", "t"])["s_norm"].mean().unstack("t")
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    cmap = plt.get_cmap("tab10")
    for i, (cat, row) in enumerate(mean.iterrows()):
        ax.plot(row.index, row.values, marker="o", ms=3, lw=1.2, color=cmap(i % 10),
                label=CATEGORY_NAMES.get(int(cat), str(cat)))
    ax.set_xlabel("timestep t")
    ax.set_ylabel("normalized prompt disagreement")
    ax.set_title("Probe: where target and source predictions diverge")
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
