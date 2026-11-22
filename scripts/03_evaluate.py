"""Stage 3: PIE-Bench metrics for every generated image, using the lab's own evaluator (PnPInversion fork).

Writes work/results/metrics_long.csv with one row per (sample_id, config_id, seed). Resumable.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from ate.cli import LOG, base_parser, grid_image_path, parse_shard, setup, shard_suffix  # noqa: E402
from ate.objective import EVAL_METRICS  # noqa: E402
from ate.pie import CENTER_CROP_SUBDIR, mask_decode  # noqa: E402


def _to_float(value):
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        return float(value.item())
    return float(np.asarray(value))


def load_calculator(settings, fake: bool, device):
    if fake:
        from ate.fake import FakeMetricsCalculator, compute_metrics_local

        return FakeMetricsCalculator(), compute_metrics_local
    eval_repo = str(settings.path("eval_repo").resolve())
    if eval_repo not in sys.path:
        sys.path.insert(0, eval_repo)
    from evaluation.evaluate import calculate_metric  # type: ignore
    from evaluation.matrics_calculator import MetricsCalculator  # type: ignore

    calc = MetricsCalculator(device or "cuda")

    def compute(calc_, metrics, src, tgt, mask, sp, tp):
        return [calculate_metric(calc_, m, src, tgt, mask, mask, sp, tp) for m in metrics]

    return calc, compute


def main() -> None:
    args = base_parser(__doc__).parse_args()
    settings, items = setup(args)
    si, sn = parse_shard(args.shard)
    configs = settings.grid()[si::sn]
    seeds = settings.seeds(args.quick)
    calc, compute = load_calculator(settings, args.fake, args.device)
    src_root = settings.path("pie_root") / CENTER_CROP_SUBDIR

    out_path = settings.results / f"metrics_long{shard_suffix(args.shard)}.csv"
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            for row in csv.DictReader(f):
                done.add((row["sample_id"], row["config_id"], row["seed"]))
    new_file = not out_path.exists()
    n_new = 0
    with out_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["sample_id", "config_id", "seed"] + EVAL_METRICS)
        for it in items:
            mask = mask_decode(it.mask_rle)[:, :, None].repeat(3, axis=2)
            with Image.open(src_root / it.rel_path) as img:
                src_img = img.convert("RGB")
            for cfg in configs:
                for seed in seeds:
                    key = (it.sample_id, cfg.id, str(seed))
                    if key in done:
                        continue
                    tgt_path = grid_image_path(settings, cfg.id, seed, it.rel_path)
                    if not tgt_path.exists():
                        continue
                    with Image.open(tgt_path) as timg:
                        tgt_img = timg.convert("RGB")
                    values = compute(calc, EVAL_METRICS, src_img, tgt_img, mask, it.source_prompt, it.target_prompt)
                    w.writerow([it.sample_id, cfg.id, seed] + [_to_float(v) for v in values])
                    n_new += 1
            f.flush()
    LOG.info("Wrote %d new metric rows to %s", n_new, out_path)


if __name__ == "__main__":
    main()
