"""Shared command-line handling for the numbered scripts."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from .config import Settings, load_settings
from .pie import PieItem, load_items, load_split

LOG = logging.getLogger("ate")


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    p.add_argument("--fake", action="store_true", help="CPU smoke test with a toy pipeline and synthetic data")
    p.add_argument("--quick", action="store_true", help="Only the first seed")
    p.add_argument("--max-samples", type=int, default=None, help="Only the first N PIE samples")
    p.add_argument("--device", default=None)
    return p


def setup(args) -> Tuple[Settings, List[PieItem]]:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    overrides = {}
    if args.fake:
        overrides["paths.work_dir"] = "work_fake"
    settings = load_settings(args.config, overrides)
    if args.fake:
        from .fake import make_fake_pie

        fake_root = settings.work / "fake_pie"
        make_fake_pie(fake_root)
        settings.raw["paths"]["pie_root"] = str(fake_root)
        settings.raw["grid"]["t_start"] = [0.5, 0.75, 1.0]
        settings.raw["grid"]["t_end"] = [0.2, 0.4]
        settings.raw["grid"]["seeds"] = [42, 43]
        settings.raw["probe"]["t_candidates"] = [0.5, 0.75, 1.0]
        settings.raw["classifier"]["use_llm"] = False
        settings.raw["bootstrap"]["n"] = 200
        settings.named.clear()
        settings.grid()
    items = load_items(settings.path("pie_root"))
    if args.max_samples:
        items = items[: args.max_samples]
    return settings, items


def split_or_die(settings: Settings) -> dict:
    path = settings.results / "split.csv"
    if not path.exists():
        raise SystemExit("Run scripts/01_prepare.py first (no split.csv).")
    return load_split(path)


def grid_image_path(settings: Settings, config_id: str, seed: int, rel_path: str) -> Path:
    return settings.work / "grid" / config_id / f"seed{seed}" / rel_path
