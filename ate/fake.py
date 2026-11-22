"""Synthetic mini PIE-Bench + fake metrics so the whole kit can be smoke-tested on CPU with no downloads."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .pie import CATEGORY_NAMES

_PAIRS = {
    0: ("a photo of a bench in a park", "a photo of a [red] bench in a [snowy] park"),
    1: ("a cat sitting on a sofa", "a [dog] sitting on a sofa"),
    2: ("a table in a kitchen", "a table with a [vase] in a kitchen"),
    3: ("a man with a hat on a street", "a man on a street"),
    4: ("a bowl of apples", "a bowl of [oranges]"),
    5: ("a dog standing on grass", "a dog [lying] on grass"),
    6: ("a blue car on a road", "a [red] car on a road"),
    7: ("a wooden chair in a room", "a [glass] chair in a room"),
    8: ("a horse in a field", "a horse on the [beach]"),
    9: ("a photo of a lighthouse", "a [watercolor painting] of a lighthouse"),
}


def _rle(mask: np.ndarray) -> list:
    flat = mask.reshape(-1).astype(np.uint8)
    out, i, n = [], 0, flat.size
    while i < n:
        if flat[i]:
            j = i
            while j < n and flat[j]:
                j += 1
            out += [i, j - i]
            i = j
        else:
            i += 1
    return out


def make_fake_pie(root: Path, per_category: int = 3, seed: int = 0) -> Path:
    root = Path(root)
    if (root / "mapping_file.json").exists():
        return root
    rng = np.random.RandomState(seed)
    mapping = {}
    for cat in CATEGORY_NAMES:
        for k in range(per_category):
            sid = f"{cat}{k:011d}"
            rel = f"{cat}_{CATEGORY_NAMES[cat].replace(' ', '_')}/{sid}.jpg"
            arr = (rng.rand(512, 512, 3) * 60 + np.linspace(0, 180, 512)[None, :, None]).astype(np.uint8)
            path = root / "annotation_images" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr).save(path)
            mask = np.zeros((512, 512), dtype=np.uint8)
            y0, x0 = rng.randint(64, 256, size=2)
            mask[y0 : y0 + 192, x0 : x0 + 192] = 1
            src, tgt = _PAIRS[cat]
            mapping[sid] = {
                "image_path": rel,
                "original_prompt": src,
                "editing_prompt": tgt,
                "editing_instruction": "",
                "editing_type_id": str(cat),
                "mask": _rle(mask),
            }
    with (root / "mapping_file.json").open("w", encoding="utf-8") as handle:
        json.dump(mapping, handle)
    return root


class FakeMetricsCalculator:
    """Same method names as PnPInversion's MetricsCalculator; cheap numpy stand-ins (plumbing tests only)."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    @staticmethod
    def _arr(img, mask=None):
        a = np.asarray(img.convert("RGB").resize((512, 512)), dtype=np.float32) / 255.0
        return a if mask is None else a * mask

    def calculate_psnr(self, a, b, ma, mb):
        mse = self.calculate_mse(a, b, ma, mb)
        return float(10 * np.log10(1.0 / max(mse, 1e-10)))

    def calculate_mse(self, a, b, ma, mb):
        x, y = self._arr(a, ma), self._arr(b, mb)
        return float(((x - y) ** 2).mean())

    def calculate_lpips(self, a, b, ma, mb):
        return float(np.abs(self._arr(a, ma) - self._arr(b, mb)).mean())

    def calculate_ssim(self, a, b, ma, mb):
        return float(1.0 - self.calculate_lpips(a, b, ma, mb))

    def calculate_structure_distance(self, a, b, ma, mb):
        return float(self.calculate_mse(a, b, ma, mb) * 0.1)

    def calculate_clip_similarity(self, img, text, mask):
        x = self._arr(img, mask)
        h = (sum(ord(c) for c in text) % 97) / 97.0
        return float(20 + 10 * (x.mean() * h + x.std()))


def compute_metrics_local(calc, metrics, src_image, tgt_image, mask, src_prompt, tgt_prompt) -> list:
    """Mirror of PnPInversion's calculate_metric for the 8 PIE metrics (used only in --fake mode)."""
    out = []
    for m in metrics:
        if m == "structure_distance":
            out.append(calc.calculate_structure_distance(src_image, tgt_image, None, None))
        elif m.endswith("_unedit_part"):
            if (1 - mask).sum() == 0:
                out.append("nan")
                continue
            base = m.replace("_unedit_part", "")
            out.append(getattr(calc, f"calculate_{base}")(src_image, tgt_image, 1 - mask, 1 - mask))
        elif m == "clip_similarity_source_image":
            out.append(calc.calculate_clip_similarity(src_image, src_prompt, None))
        elif m == "clip_similarity_target_image":
            out.append(calc.calculate_clip_similarity(tgt_image, tgt_prompt, None))
        elif m == "clip_similarity_target_image_edit_part":
            out.append("nan" if mask.sum() == 0 else calc.calculate_clip_similarity(tgt_image, tgt_prompt, mask))
        else:
            raise ValueError(m)
    return out
