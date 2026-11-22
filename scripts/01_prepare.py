"""Stage 1: dev/test split + the center-cropped source folder the PIE evaluator compares against."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ate.cli import LOG, base_parser, setup  # noqa: E402
from ate.pie import CATEGORY_NAMES, make_split, prepare_center_crop_folder, save_split  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    settings, items = setup(args)
    split = make_split(items, settings.raw["split"]["dev_fraction"], settings.raw["split"]["seed"])
    save_split(split, items, settings.results / "split.csv")
    out = prepare_center_crop_folder(settings.path("pie_root"), items)
    n_dev = sum(v == "dev" for v in split.values())
    LOG.info("Items: %d (dev %d / test %d). Source folder for evaluation: %s", len(items), n_dev, len(items) - n_dev, out)
    counts = {}
    for it in items:
        counts[it.category] = counts.get(it.category, 0) + 1
    for cat, n in sorted(counts.items()):
        LOG.info("  category %d %-18s %d", cat, CATEGORY_NAMES[cat], n)
    grid = settings.grid()
    seeds = settings.seeds(args.quick)
    LOG.info("Grid: %d configs x %d seed(s) x %d images = %d edits", len(grid), len(seeds), len(items),
             len(grid) * len(seeds) * len(items))


if __name__ == "__main__":
    main()
