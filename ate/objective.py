"""Scalar edit-quality score J from PIE-Bench metrics (z-scored on the dev split only)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd

# PIE-Bench metric names exactly as written by PnPInversion/evaluation/evaluate.py
EVAL_METRICS = [
    "structure_distance",
    "psnr_unedit_part",
    "lpips_unedit_part",
    "mse_unedit_part",
    "ssim_unedit_part",
    "clip_similarity_source_image",
    "clip_similarity_target_image",
    "clip_similarity_target_image_edit_part",
]

EDIT_TERMS: Dict[str, int] = {
    "clip_similarity_target_image": +1,
    "clip_similarity_target_image_edit_part": +1,
}
PRESERVE_TERMS: Dict[str, int] = {
    "psnr_unedit_part": +1,
    "lpips_unedit_part": -1,
    "structure_distance": -1,
}

# Display names / directions for tables
DISPLAY = {
    "structure_distance": ("Structure dist. x10^3", -1, 1e3),
    "psnr_unedit_part": ("BG PSNR", +1, 1.0),
    "lpips_unedit_part": ("BG LPIPS x10^3", -1, 1e3),
    "ssim_unedit_part": ("BG SSIM x10^2", +1, 1e2),
    "clip_similarity_target_image": ("CLIP whole", +1, 1.0),
    "clip_similarity_target_image_edit_part": ("CLIP edited", +1, 1.0),
}


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in EVAL_METRICS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


@dataclass
class Standardizer:
    mean: Dict[str, float]
    std: Dict[str, float]

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "Standardizer":
        cols = list(EDIT_TERMS) + list(PRESERVE_TERMS)
        mean = {c: float(np.nanmean(df[c].to_numpy(dtype=float))) for c in cols}
        std = {c: float(np.nanstd(df[c].to_numpy(dtype=float))) or 1.0 for c in cols}
        return cls(mean, std)

    def z(self, df: pd.DataFrame, col: str) -> pd.Series:
        return (df[col].astype(float) - self.mean[col]) / self.std[col]

    def edit_score(self, df: pd.DataFrame) -> pd.Series:
        parts = pd.concat([sign * self.z(df, c) for c, sign in EDIT_TERMS.items()], axis=1)
        return parts.mean(axis=1, skipna=True).fillna(0.0)

    def preserve_score(self, df: pd.DataFrame) -> pd.Series:
        parts = pd.concat([sign * self.z(df, c) for c, sign in PRESERVE_TERMS.items()], axis=1)
        return parts.mean(axis=1, skipna=True).fillna(0.0)

    def score(self, df: pd.DataFrame, edit_weight: float = 1.0, preserve_weight: float = 1.0) -> pd.Series:
        return edit_weight * self.edit_score(df) + preserve_weight * self.preserve_score(df)
