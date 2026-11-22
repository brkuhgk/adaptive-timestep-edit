"""Stage 2: run the lab's ChordEdit pipeline for every (config, seed, image). Resumable: existing files are skipped."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from ate.cli import LOG, base_parser, grid_image_path, setup  # noqa: E402
from ate.editor import edit_image, load_pipeline  # noqa: E402


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--only-config", default=None, help="Run a single config id (e.g. for a quick check)")
    args = parser.parse_args()
    settings, items = setup(args)
    configs = settings.grid()
    if args.only_config:
        configs = [c for c in configs if c.id == args.only_config]
    seeds = settings.seeds(args.quick)
    strip = bool(settings.raw["grid"].get("strip_brackets", False))
    base = settings.base_edit_config()
    pipe = load_pipeline(settings, fake=args.fake, device=args.device)
    pie_root = settings.path("pie_root")

    timing_path = settings.results / "timing_grid.csv"
    new_file = not timing_path.exists()
    with timing_path.open("a", newline="") as tf:
        tw = csv.writer(tf)
        if new_file:
            tw.writerow(["config_id", "seed", "n_new", "sec_per_edit"])
        total = len(configs) * len(seeds) * len(items)
        done = 0
        for cfg in configs:
            for seed in seeds:
                t0, n_new = time.time(), 0
                for it in items:
                    done += 1
                    out = grid_image_path(settings, cfg.id, seed, it.rel_path)
                    if out.exists():
                        continue
                    with Image.open(pie_root / "annotation_images" / it.rel_path) as img:
                        src_img = img.convert("RGB")
                    src_p, tgt_p = it.prompts(strip)
                    edited = edit_image(pipe, src_img, src_p, tgt_p, cfg, base, seed)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    edited.save(out)
                    n_new += 1
                sec = (time.time() - t0) / n_new if n_new else 0.0
                tw.writerow([cfg.id, seed, n_new, f"{sec:.4f}"])
                tf.flush()
                LOG.info("[%d/%d] %s seed %d: %d new edits, %.3f s/edit", done, total, cfg.id, seed, n_new, sec)


if __name__ == "__main__":
    main()
