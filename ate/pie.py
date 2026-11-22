"""PIE-Bench loading, edit categories, dev/test split, and source-image preparation."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from PIL import Image, ImageOps

CATEGORY_NAMES: Dict[int, str] = {
    0: "random",
    1: "change object",
    2: "add object",
    3: "delete object",
    4: "change content",
    5: "change pose",
    6: "change color",
    7: "change material",
    8: "change background",
    9: "change style",
}

CENTER_CROP_SUBDIR = "annotation_images_center_crop_resize512"


@dataclass(frozen=True)
class PieItem:
    sample_id: str
    rel_path: str
    source_prompt_raw: str
    target_prompt_raw: str
    instruction: str
    category: int
    mask_rle: Sequence[int]

    @property
    def source_prompt(self) -> str:
        return clean_prompt(self.source_prompt_raw)

    @property
    def target_prompt(self) -> str:
        return clean_prompt(self.target_prompt_raw)

    def prompts(self, strip_brackets: bool) -> tuple[str, str]:
        if strip_brackets:
            return self.source_prompt, self.target_prompt
        return self.source_prompt_raw, self.target_prompt_raw


def clean_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.replace("[", "").replace("]", "")).strip()


def _category_from(meta: dict, rel_path: str) -> int:
    value = meta.get("editing_type_id")
    if value is not None:
        return int(value)
    # Fallback: PIE folders are named like "6_change_attribute_color_80/..."
    head = Path(rel_path).parts[0]
    match = re.match(r"(\d+)_", head)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot infer edit category for {rel_path}")


def load_items(pie_root: Path, mapping_file: str = "mapping_file.json") -> List[PieItem]:
    mapping_path = Path(pie_root) / mapping_file
    with mapping_path.open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    items: List[PieItem] = []
    for sample_id in sorted(mapping):
        meta = mapping[sample_id]
        rel = meta.get("image_path")
        if rel is None:
            continue
        src = meta.get("original_prompt") or meta.get("source_prompt") or ""
        tgt = meta.get("editing_prompt") or meta.get("edited_prompt") or meta.get("target_prompt") or ""
        items.append(
            PieItem(
                sample_id=str(sample_id),
                rel_path=str(rel),
                source_prompt_raw=src,
                target_prompt_raw=tgt,
                instruction=meta.get("editing_instruction") or "",
                category=_category_from(meta, rel),
                mask_rle=tuple(meta.get("mask") or ()),
            )
        )
    if not items:
        raise FileNotFoundError(f"No PIE items found in {mapping_path}")
    return items


def mask_decode(encoded_mask: Sequence[int], shape=(512, 512), border: bool = True) -> np.ndarray:
    """RLE decode identical to PnPInversion/evaluation/evaluate.py (pairs of start, length)."""
    length = shape[0] * shape[1]
    flat = np.zeros((length,), dtype=np.float32)
    for i in range(0, len(encoded_mask) - 1, 2):
        start, run = int(encoded_mask[i]), int(encoded_mask[i + 1])
        flat[start : start + min(run, max(0, length - start))] = 1.0
    mask = flat.reshape(shape)
    if border:
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = 1.0
    return mask


def make_split(items: Sequence[PieItem], dev_fraction: float, seed: int) -> Dict[str, str]:
    """Stratified (by category) deterministic dev/test split."""
    rng = np.random.RandomState(seed)
    split: Dict[str, str] = {}
    by_cat: Dict[int, List[str]] = {}
    for item in items:
        by_cat.setdefault(item.category, []).append(item.sample_id)
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        order = rng.permutation(len(ids))
        n_dev = int(round(dev_fraction * len(ids)))
        if len(ids) >= 2:
            n_dev = min(max(n_dev, 1), len(ids) - 1)
        for rank, idx in enumerate(order):
            split[ids[idx]] = "dev" if rank < n_dev else "test"
    return split


def save_split(split: Dict[str, str], items: Sequence[PieItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cats = {it.sample_id: it.category for it in items}
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "category", "split"])
        for sid in sorted(split):
            writer.writerow([sid, cats[sid], split[sid]])


def load_split(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as handle:
        return {row["sample_id"]: row["split"] for row in csv.DictReader(handle)}


def center_crop_resize(image: Image.Image, size: int = 512) -> Image.Image:
    """Same preprocessing as ChordEditPipeline (center square crop, LANCZOS resize)."""
    w, h = image.size
    if w != h:
        side = min(w, h)
        image = ImageOps.fit(image, (side, side), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def prepare_center_crop_folder(pie_root: Path, items: Sequence[PieItem], size: int = 512,
                               image_subdir: str = "annotation_images", out_subdir: Optional[str] = None) -> Path:
    """Create the source folder the PIE evaluator compares against (skips existing files)."""
    out_root = Path(pie_root) / (out_subdir or CENTER_CROP_SUBDIR)
    for item in items:
        dst = out_root / item.rel_path
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(Path(pie_root) / image_subdir / item.rel_path) as img:
            center_crop_resize(img.convert("RGB"), size).save(dst)
    return out_root
