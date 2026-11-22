"""Stage 6: oracles, predictors, statistics, figures, and the numbers for the report and the email.

Outputs in work/results/:
  main_table.csv / .md     methods compared on the TEST split (tuned on DEV)
  sensitivity.csv          gap closed under different preserve weights in the objective
  category_tstart.csv      mean J per (category, t_start)  -> fig_heatmap.png
  probe_tuning.csv         dev score of each probe rule / tau
  summary.json             every headline number (used to fill the email and report)
  fig_heatmap.png, fig_gap_closed.png, fig_probe_curves.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ate.analysis import (MethodResult, Study, category_best_tstart, classifier_choices, compare,  # noqa: E402
                          probe_choices, tune_probe)
from ate.cli import LOG, base_parser, setup, split_or_die  # noqa: E402
from ate.config import EditConfig  # noqa: E402
from ate.objective import coerce_numeric  # noqa: E402
from ate.pie import CATEGORY_NAMES  # noqa: E402
from ate.plots import gap_closed_bars, heatmap_category_tstart, probe_curves  # noqa: E402

PAPER_CLIP_EDITED = 22.96      # ChordEdit (original paper) CLIP-edited on PIE-Bench
REPRO_CLIP_EDITED = 22.1554    # Lab reproduction, README of ChordEdit-Reproduction


def build_methods(study: Study, items, probe_long, clf, taus):
    test = study.ids("test")
    true_cat = {it.sample_id: it.category for it in items}
    table = study.category_table()
    methods = [MethodResult("Rule: category table (true label)", {s: table[true_cat[s]] for s in test}, 0,
                            "upper bound for any prompt classifier")]
    if clf is not None:
        kw = dict(zip(clf["sample_id"], clf["keyword_category"]))
        methods.append(MethodResult("Prompt keywords -> category table", classifier_choices(study, kw, test), 0, "zero cost"))
        if "llm_parsed" in clf and clf["llm_parsed"].astype(str).str.lower().eq("true").any():
            llm = dict(zip(clf["sample_id"], clf["llm_category"]))
            methods.append(MethodResult("LLM classifier -> category table", classifier_choices(study, llm, test), 0,
                                        "1 small-LLM call, no UNet"))
    tuning = None
    if probe_long is not None:
        rule, tau, tuning = tune_probe(study, probe_long, taus)
        calls = float(probe_long.groupby("sample_id")["unet_calls"].first().mean())
        name = f"Velocity probe ({rule}{'' if tau is None else f', tau={tau:g}'})"
        methods.append(MethodResult(name, probe_choices(study, probe_long, rule, tau, test), calls, "training-free"))
    return methods, tuning


def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            cells.append(format(v, floatfmt) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    args = base_parser(__doc__).parse_args()
    settings, items = setup(args)
    res = settings.results
    split = split_or_die(settings)
    category = {it.sample_id: it.category for it in items}
    metric_files = sorted(res.glob("metrics_long*.csv"))
    if not metric_files:
        raise SystemExit("No metrics_long*.csv found; run scripts/03_evaluate.py first.")
    metrics = coerce_numeric(pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in metric_files], ignore_index=True))
    metrics = metrics.drop_duplicates(["sample_id", "config_id", "seed"])
    metrics = metrics[metrics["sample_id"].isin(category)]
    if args.quick:
        metrics = metrics[metrics["seed"] == settings.seeds()[0]]
    probe_files = sorted(res.glob("probe_long*.csv"))
    probe_long = (pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in probe_files], ignore_index=True)
                  .drop_duplicates(["sample_id", "t"]) if probe_files else None)
    clf = pd.read_csv(res / "classifier.csv", dtype={"sample_id": str}) if (res / "classifier.csv").exists() else None
    obj = settings.raw["objective"]
    boot = settings.raw["bootstrap"]
    taus = [float(t) for t in settings.raw["probe"]["taus"]]

    study = Study(metrics, split, category, settings.reference_id(), obj["edit_weight"], obj["preserve_weight"])
    methods, tuning = build_methods(study, items, probe_long, clf, taus)
    table, info = compare(study, methods, int(boot["n"]), int(boot["seed"]))
    table.to_csv(res / "main_table.csv", index=False)
    show = table[["method", "J", "dJ_vs_default", "dJ_lo", "dJ_hi", "gap_closed_vs_default_%",
                  "gap_closed_vs_globalbest_%", "clip_similarity_target_image_edit_part", "psnr_unedit_part",
                  "lpips_unedit_part", "structure_distance", "extra_unet_calls"]].copy()
    show.columns = ["Method", "J", "dJ", "dJ lo", "dJ hi", "Gap closed % (vs default)", "Gap closed % (vs global best)",
                    "CLIP edited", "BG PSNR", "BG LPIPS", "Struct. dist.", "Extra UNet calls"]
    (res / "main_table.md").write_text(to_markdown(show) + "\n")
    if tuning is not None:
        tuning.to_csv(res / "probe_tuning.csv", index=False)

    # Sensitivity of conclusions to the objective's preserve weight
    sens_rows = []
    for w in obj.get("sensitivity", [1.0]):
        st = Study(metrics, split, category, settings.reference_id(), obj["edit_weight"], float(w))
        ms, _ = build_methods(st, items, probe_long, clf, taus)
        tb, _ = compare(st, ms, 200, int(boot["seed"]))
        for _, r in tb.iterrows():
            sens_rows.append({"preserve_weight": w, "method": r["method"], "gap_closed_%": r["gap_closed_vs_default_%"],
                              "dJ": r["dJ_vs_default"], "dJ_lo": r["dJ_lo"], "dJ_hi": r["dJ_hi"]})
    pd.DataFrame(sens_rows).to_csv(res / "sensitivity.csv", index=False)

    # Category x t_start (all samples, using dev-chosen t_end per t_start)
    cat_t = category_best_tstart(study, study.ids("dev") + study.ids("test"))
    cat_t.to_csv(res / "category_tstart.csv")
    heatmap_category_tstart(cat_t, res / "fig_heatmap.png")
    gap_closed_bars(table, res / "fig_gap_closed.png")
    if probe_long is not None:
        probe_curves(probe_long, category, res / "fig_probe_curves.png")

    # Headline numbers
    best_t = {int(c): float(cat_t.columns[int(np.nanargmax(r.to_numpy()))]) for c, r in cat_t.iterrows()}
    lo_cat = min(best_t, key=best_t.get)
    hi_cat = max(best_t, key=best_t.get)
    learned = table[~table["method"].str.startswith(("Fixed", "Oracle", "Rule"))]
    best_row = learned.sort_values("J", ascending=False).iloc[0] if len(learned) else table.iloc[0]
    summary = {
        **info,
        "n_configs": len(study.configs),
        "seeds": sorted(int(s) for s in metrics["seed"].unique()),
        "best_practical_method": best_row["method"],
        "best_practical_gap_closed_%": float(best_row["gap_closed_vs_default_%"]),
        "best_practical_dJ_ci": [float(best_row["dJ_vs_default"]), float(best_row["dJ_lo"]), float(best_row["dJ_hi"])],
        "best_practical_extra_unet_calls": float(best_row["extra_unet_calls"]),
        "best_practical_clip_edited": float(best_row.get("clip_similarity_target_image_edit_part", np.nan)),
        "paper_clip_edited": PAPER_CLIP_EDITED,
        "lab_repro_clip_edited": REPRO_CLIP_EDITED,
        "best_t_start_by_category": {CATEGORY_NAMES[c]: t for c, t in best_t.items()},
        "lowest_t_category": [CATEGORY_NAMES[lo_cat], best_t[lo_cat]],
        "highest_t_category": [CATEGORY_NAMES[hi_cat], best_t[hi_cat]],
        "global_best_config_parsed": EditConfig.from_id(info["global_best_config"]).__dict__,
    }
    if clf is not None:
        summary["keyword_classifier_acc_%"] = float(100 * (clf["keyword_category"] == clf["true_category"]).mean())
        summary["llm_classifier_acc_%"] = float(100 * (clf["llm_category"] == clf["true_category"]).mean())
    if probe_long is not None:
        summary["probe_seconds_per_image"] = float(probe_long.groupby("sample_id")["seconds"].first().astype(float).mean())
        knee = probe_long.groupby("sample_id").apply(
            lambda g: g.sort_values("t")["t"].iloc[int(np.argmax((g.sort_values("t")["s_bar"] / g["s_bar"].max()).to_numpy() >= 0.8))],
            include_groups=False)
        per_cat_knee = knee.groupby(knee.index.map(category)).mean()
        common = [c for c in per_cat_knee.index if c in best_t]
        if len(common) >= 3:
            summary["spearman_knee_vs_best_t_by_category"] = float(
                pd.Series([per_cat_knee[c] for c in common]).rank().corr(pd.Series([best_t[c] for c in common]).rank()))
    (res / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    LOG.info("Main table:\n%s", (res / "main_table.md").read_text())
    LOG.info("Summary: %s", json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
