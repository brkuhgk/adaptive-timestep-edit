"""Stage 4: the training-free probe. For each image, measure how far the target- and source-prompt predictions
disagree at each candidate timestep (same noise the edit will use). Writes work/results/probe_long.csv."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from ate.cli import LOG, base_parser, parse_shard, setup, shard_suffix  # noqa: E402
from ate.editor import compute_prompt_deltas, load_pipeline  # noqa: E402
from ate.probe_math import curve_statistics  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    settings, items = setup(args)
    si, sn = parse_shard(args.shard)
    items = items[si::sn]
    pcfg = settings.raw["probe"]
    ts = [float(t) for t in pcfg["t_candidates"]]
    seed = settings.seeds()[0]
    strip = bool(settings.raw["grid"].get("strip_brackets", False))
    pipe = load_pipeline(settings, fake=args.fake, device=args.device)
    pie_root = settings.path("pie_root")

    out_path = settings.results / f"probe_long{shard_suffix(args.shard)}.csv"
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {row["sample_id"] for row in csv.DictReader(f)}
    new_file = not out_path.exists()
    with out_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["sample_id", "t", "s", "s_bar", "rho", "lf", "unet_calls", "seconds"])
        for i, it in enumerate(items, 1):
            if it.sample_id in done:
                continue
            with Image.open(pie_root / "annotation_images" / it.rel_path) as img:
                src_img = img.convert("RGB")
            src_p, tgt_p = it.prompts(strip)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.time()
            deltas, calls = compute_prompt_deltas(pipe, src_img, src_p, tgt_p, ts, int(pcfg["noise_samples"]), seed,
                                                  chunk=int(pcfg.get("chunk", 14)))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            sec = time.time() - t0
            stats = curve_statistics(deltas)
            for j, t in enumerate(ts):
                w.writerow([it.sample_id, t, stats["s"][j], stats["s_bar"][j], stats["rho"][j], stats["lf"][j], calls, f"{sec:.4f}"])
            if i % 50 == 0:
                f.flush()
                LOG.info("probe %d/%d (%.2f s/image, %d UNet evals)", i, len(items), sec, calls)
    LOG.info("Probe written to %s", out_path)


if __name__ == "__main__":
    main()
