"""Stage 9: held-out-seed check of the per-image oracle.

The in-sample oracle picks, for each test image, the config that looks best on seed 42. Part of that
win can be seed noise. Here we re-edit each test image with a new seed (43) using (a) the paper default
and (b) the oracle's seed-42 choices, score them with the lab evaluator, and report the paired change.
If the gain survives the seed change, per-image headroom is real.

    python scripts/09_heldout_seed.py --shard 0/2      # edit + score (resumable)
    python scripts/09_heldout_seed.py --report         # paired results -> work/results/heldout_seed.json
"""
from __future__ import annotations

import csv
import glob
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from ate.cli import LOG, base_parser, grid_image_path, parse_shard, setup  # noqa: E402
from ate.config import EditConfig  # noqa: E402
from ate.objective import EVAL_METRICS  # noqa: E402
from ate.pie import CENTER_CROP_SUBDIR, mask_decode  # noqa: E402

E, W = "clip_similarity_target_image_edit_part", "clip_similarity_target_image"
PS, LP, SD = "psnr_unedit_part", "lpips_unedit_part", "structure_distance"
NEW_SEED = 43


def oracle_choices(results: Path, ref: str) -> dict:
    """Per test image: in-sample oracle choice under A (same edit, cleaner BG) and B (same BG, stronger edit)."""
    split = pd.read_csv(results / "split.csv", dtype={"sample_id": str}).set_index("sample_id")["split"].to_dict()
    m = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(str(results / "metrics_long*.csv"))])
    m = m[m.seed.astype(str) == "42"].drop_duplicates(["sample_id", "config_id"])
    piv = {c: m.pivot_table(index="sample_id", columns="config_id", values=c) for c in [E, W, PS, LP, SD]}
    ids = sorted(piv[E].index)
    piv = {k: v.reindex(index=ids) for k, v in piv.items()}
    dev = [s for s in ids if split.get(s) == "dev"]
    test = [s for s in ids if split.get(s) == "test"]
    mu = {c: np.nanmean(piv[c].loc[dev].values) for c in [PS, LP, SD]}
    sd = {c: np.nanstd(piv[c].loc[dev].values) for c in [PS, LP, SD]}
    pres = ((piv[PS] - mu[PS]) / sd[PS] - (piv[LP] - mu[LP]) / sd[LP] - (piv[SD] - mu[SD]) / sd[SD]) / 3.0
    cfgs = list(piv[E].columns)
    out = {}
    for s in test:
        ok_a = [c for c in cfgs if piv[E].loc[s, c] >= piv[E].loc[s, ref] and piv[W].loc[s, c] >= piv[W].loc[s, ref]]
        a = max(ok_a, key=lambda c: np.nan_to_num(pres.loc[s, c], nan=-1e9)) if ok_a else ref
        ok_b = [c for c in cfgs if piv[PS].loc[s, c] >= piv[PS].loc[s, ref] and piv[LP].loc[s, c] <= piv[LP].loc[s, ref]]
        b = max(ok_b, key=lambda c: piv[E].loc[s, c]) if ok_b else ref
        out[s] = {"A": a, "B": b}
    return out, piv


def load_calc(settings, device):
    # The editor repo and the evaluator repo both ship a top-level "utils" module; drop the editor's copy.
    for k in [k for k, v in list(sys.modules.items()) if "ChordEdit-Reproduction" in str(getattr(v, "__file__", "") or "")]:
        del sys.modules[k]
    sys.modules.pop("utils", None)
    sys.path[:] = [q for q in sys.path if "ChordEdit-Reproduction" not in q]
    spec = importlib.util.spec_from_file_location("ev", Path(__file__).resolve().parent / "03_evaluate.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    return ev.load_calculator(settings, False, device), ev._to_float


def main() -> None:
    parser = base_parser(__doc__)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()
    settings, items = setup(args)
    ref = settings.reference_id()
    choices, piv42 = oracle_choices(settings.results, ref)

    if args.report:
        report(settings, choices, piv42, ref)
        return

    by_id = {it.sample_id: it for it in items}
    pairs = sorted({(s, c) for s, ch in choices.items() for c in (ref, ch["A"], ch["B"])})
    si, sn = parse_shard(args.shard)
    pairs = pairs[si::sn]
    LOG.info("Held-out seed %d: %d (image, config) pairs in this shard", NEW_SEED, len(pairs))

    from ate.editor import edit_image, load_pipeline

    strip = bool(settings.raw["grid"].get("strip_brackets", False))
    base = settings.base_edit_config()
    todo = any(not grid_image_path(settings, c, NEW_SEED, by_id[s].rel_path).exists() for s, c in pairs)
    pipe = load_pipeline(settings, fake=args.fake, device=args.device) if todo else None
    pie_root = settings.path("pie_root")
    for k, (s, cid) in enumerate(pairs):
        it = by_id[s]
        out = grid_image_path(settings, cid, NEW_SEED, it.rel_path)
        if out.exists():
            continue
        with Image.open(pie_root / "annotation_images" / it.rel_path) as img:
            src_img = img.convert("RGB")
        src_p, tgt_p = it.prompts(strip)
        edited = edit_image(pipe, src_img, src_p, tgt_p, EditConfig.from_id(cid), base, NEW_SEED)
        out.parent.mkdir(parents=True, exist_ok=True)
        edited.save(out)
        if k % 100 == 0:
            LOG.info("edit %d/%d", k, len(pairs))
    del pipe

    (calc, compute), to_float = load_calc(settings, args.device)
    src_root = pie_root / CENTER_CROP_SUBDIR
    out_path = settings.results / f"metrics_heldout.shard{si}of{sn}.csv"
    done = set()
    if out_path.exists():
        with out_path.open() as f:
            done = {(r["sample_id"], r["config_id"]) for r in csv.DictReader(f)}
    new_file = not out_path.exists()
    with out_path.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["sample_id", "config_id", "seed"] + EVAL_METRICS)
        for k, (s, cid) in enumerate(pairs):
            if (s, cid) in done:
                continue
            it = by_id[s]
            mask = mask_decode(it.mask_rle)[:, :, None].repeat(3, axis=2)
            with Image.open(src_root / it.rel_path) as img:
                src_img = img.convert("RGB")
            with Image.open(grid_image_path(settings, cid, NEW_SEED, it.rel_path)) as timg:
                tgt_img = timg.convert("RGB")
            vals = compute(calc, EVAL_METRICS, src_img, tgt_img, mask, it.source_prompt, it.target_prompt)
            w.writerow([s, cid, NEW_SEED] + [to_float(v) for v in vals])
            if k % 50 == 0:
                f.flush()
                LOG.info("score %d/%d", k, len(pairs))
    LOG.info("Done: %s", out_path)


def boot(d, n=2000, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    r = np.random.RandomState(seed)
    mm = d[r.randint(0, len(d), (n, len(d)))].mean(1)
    return [float(d.mean()), float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]


def report(settings, choices, piv42, ref) -> None:
    h = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob(str(settings.results / "metrics_heldout*.csv"))])
    h = h.drop_duplicates(["sample_id", "config_id"]).set_index(["sample_id", "config_id"])
    res = {"seed_select": 42, "seed_eval": NEW_SEED, "reference": ref}
    for key in ["A", "B"]:
        rows = {"n": 0}
        for lab, col in [("CLIP_edited", E), ("CLIP_whole", W), ("BG_PSNR", PS), ("BG_LPIPS", LP), ("Struct", SD)]:
            d42, d43 = [], []
            for s, ch in choices.items():
                c = ch[key]
                if (s, c) in h.index and (s, ref) in h.index:
                    d43.append(float(h.loc[(s, c), col]) - float(h.loc[(s, ref), col]))
                    d42.append(float(piv42[col].loc[s, c]) - float(piv42[col].loc[s, ref]))
            rows[lab + "_in_sample_seed42"] = boot(d42)
            rows[lab + "_held_out_seed43"] = boot(d43)
            rows["n"] = len(d43)
        res["oracle_" + key] = rows
    (settings.results / "heldout_seed.json").write_text(json.dumps(res, indent=1))
    for key in ["A", "B"]:
        r = res["oracle_" + key]
        print(f"== oracle {key} (n={r['n']})")
        for lab in ["CLIP_edited", "CLIP_whole", "BG_PSNR", "BG_LPIPS", "Struct"]:
            a, b = r[lab + "_in_sample_seed42"], r[lab + "_held_out_seed43"]
            print(f"  {lab:12s} seed42 {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}]   seed43 {b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}]")


if __name__ == "__main__":
    main()
