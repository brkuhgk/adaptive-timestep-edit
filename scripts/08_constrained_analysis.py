"""Matched-edit-strength analysis: compare configs at equal edit strength (A) or equal preservation (B)."""
import glob, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from ate.pie import load_items, CATEGORY_NAMES
from ate.config import EditConfig
from ate.probe_math import knee_select

items = load_items("datasets/PIE-Bench_v1"); cat = {i.sample_id: i.category for i in items}
split = pd.read_csv("work/results/split.csv", dtype={"sample_id": str}).set_index("sample_id")["split"].to_dict()
m = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob("work/results/metrics_long*.csv")])
m = m.drop_duplicates(["sample_id", "config_id", "seed"])
E, W = "clip_similarity_target_image_edit_part", "clip_similarity_target_image"
PS, LP, SD = "psnr_unedit_part", "lpips_unedit_part", "structure_distance"
for c in [E, W, PS, LP, SD]:
    m[c] = pd.to_numeric(m[c], errors="coerce")
REF = "ts0.90_te0.30_d0.15_c1"
piv = {c: m.pivot_table(index="sample_id", columns="config_id", values=c) for c in [E, W, PS, LP, SD]}
ids = sorted(piv[E].index)
piv = {k: v.reindex(index=ids) for k, v in piv.items()}
dev = [s for s in ids if split.get(s) == "dev"]; test = [s for s in ids if split.get(s) == "test"]
cfgs = list(piv[E].columns)
# preservation score, z-scored on dev over all configs
mu = {c: np.nanmean(piv[c].loc[dev].values) for c in [PS, LP, SD]}
sd = {c: np.nanstd(piv[c].loc[dev].values) for c in [PS, LP, SD]}
PRES = ((piv[PS] - mu[PS]) / sd[PS] - (piv[LP] - mu[LP]) / sd[LP] - (piv[SD] - mu[SD]) / sd[SD]) / 3.0

def pick_A(group):  # same edit strength, cleanest background
    e, w, p = piv[E].loc[group].mean(), piv[W].loc[group].mean(), PRES.loc[group].mean()
    ok = [c for c in cfgs if e[c] >= e[REF] and w[c] >= w[REF]]
    return max(ok, key=lambda c: np.nan_to_num(p[c], nan=-1e9)) if ok else REF

def pick_B(group):  # same background quality, strongest edit
    e, ps, lp = piv[E].loc[group].mean(), piv[PS].loc[group].mean(), piv[LP].loc[group].mean()
    ok = [c for c in cfgs if ps[c] >= ps[REF] and lp[c] <= lp[REF]]
    return max(ok, key=lambda c: e[c]) if ok else REF

def boot(diff, n=2000, seed=0):
    d = np.asarray(diff, float); d = d[np.isfinite(d)]
    r = np.random.RandomState(seed); idx = r.randint(0, len(d), (n, len(d)))
    mm = d[idx].mean(1); return [float(d.mean()), float(np.percentile(mm, 2.5)), float(np.percentile(mm, 97.5))]

def evaluate(name, choice):
    rows = {k: np.array([piv[k].loc[s, choice[s]] for s in test]) for k in [E, W, PS, LP, SD]}
    refv = {k: np.array([piv[k].loc[s, REF] for s in test]) for k in [E, W, PS, LP, SD]}
    out = {"method": name}
    for k, lab in [(E, "CLIP_edited"), (W, "CLIP_whole"), (PS, "BG_PSNR"), (LP, "BG_LPIPS"), (SD, "Struct")]:
        out[lab] = float(np.nanmean(rows[k]))
        out["d_" + lab] = boot(rows[k] - refv[k])
    out["configs_used"] = len(set(choice.values()))
    return out

clf = pd.read_csv("work/results/classifier.csv", dtype={"sample_id": str}).set_index("sample_id")
res = {"A_same_edit_cleaner_bg": [], "B_same_bg_stronger_edit": []}
for key, pick in [("A_same_edit_cleaner_bg", pick_A), ("B_same_bg_stronger_edit", pick_B)]:
    g = pick(dev)
    table_true = {c: pick([s for s in dev if cat[s] == c]) for c in CATEGORY_NAMES}
    res[key].append(evaluate("Fixed: paper default", {s: REF for s in test}))
    res[key].append(evaluate("Fixed: dev best under constraint (" + g + ")", {s: g for s in test}))
    res[key].append(evaluate("Category table (true label)", {s: table_true[cat[s]] for s in test}))
    res[key].append(evaluate("Keyword classifier -> table", {s: table_true[int(clf.loc[s, "keyword_category"])] for s in test}))
    res[key].append(evaluate("LLM classifier -> table", {s: table_true[int(clf.loc[s, "llm_category"])] for s in test}))
    # per-image oracle under the same constraint (in-sample, optimistic)
    orc = {}
    for s in test:
        if key.startswith("A"):
            ok = [c for c in cfgs if piv[E].loc[s, c] >= piv[E].loc[s, REF] and piv[W].loc[s, c] >= piv[W].loc[s, REF]]
            orc[s] = max(ok, key=lambda c: np.nan_to_num(PRES.loc[s, c], nan=-1e9)) if ok else REF
        else:
            ok = [c for c in cfgs if piv[PS].loc[s, c] >= piv[PS].loc[s, REF] and piv[LP].loc[s, c] <= piv[LP].loc[s, REF]]
            orc[s] = max(ok, key=lambda c: piv[E].loc[s, c]) if ok else REF
    res[key].append(evaluate("Oracle per image (in-sample)", orc))
    res[key + "_table"] = {CATEGORY_NAMES[c]: v for c, v in table_true.items()}

# probe under constraint A: knee tau and t_end per t_start chosen on dev
pl = pd.concat([pd.read_csv(f, dtype={"sample_id": str}) for f in glob.glob("work/results/probe_long*.csv")]).drop_duplicates(["sample_id", "t"])
by_ts = {}
for c in cfgs:
    ec = EditConfig.from_id(c)
    if ec.t_delta == 0.0:
        by_ts.setdefault(ec.t_start, []).append(c)
def probe_choice(group, tau, te_map):
    out = {}
    for sid, gdf in pl[pl.sample_id.isin(group)].groupby("sample_id"):
        t = knee_select(gdf["t"], gdf["s_bar"], tau); ts = min(by_ts, key=lambda a: abs(a - t)); out[sid] = te_map[ts]
    return out
te_map = {}
for ts, cs in by_ts.items():
    e, p = piv[E].loc[dev, cs].mean(), PRES.loc[dev, cs].mean()
    te_map[ts] = max(cs, key=lambda c: (e[c] >= piv[E].loc[dev, REF].mean(), p[c]))
best = None
for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
    ch = probe_choice(dev, tau, te_map)
    e = np.mean([piv[E].loc[s, c] for s, c in ch.items()]); p = np.mean([PRES.loc[s, c] for s, c in ch.items()])
    ok = e >= piv[E].loc[dev, REF].mean()
    if best is None or (ok, p) > (best[1], best[2]):
        best = (tau, ok, p)
res["A_same_edit_cleaner_bg"].insert(5, evaluate(f"Velocity probe (tau={best[0]}, meets constraint on dev: {best[1]})", probe_choice(test, best[0], te_map)))

# knee vs category: mean knee t (tau 0.5) per category, and which share of images any config dominates the default
knee = pl.groupby("sample_id").apply(lambda g: knee_select(g["t"], g["s_bar"], 0.5), include_groups=False)
res["mean_knee_t_by_category"] = {CATEGORY_NAMES[c]: float(knee[[s for s in knee.index if cat[s] == c]].mean()) for c in CATEGORY_NAMES}
dom = 0
for s in test:
    r = {k: piv[k].loc[s, REF] for k in [E, W, PS, LP, SD]}
    if any(piv[E].loc[s, c] >= r[E] and piv[W].loc[s, c] >= r[W] and piv[PS].loc[s, c] >= r[PS] and piv[LP].loc[s, c] <= r[LP] and piv[SD].loc[s, c] <= r[SD] for c in cfgs if c != REF):
        dom += 1
res["share_test_images_with_a_dominating_config_%"] = 100.0 * dom / len(test)
json.dump(res, open("work/results/constrained_results.json", "w"), indent=1)
def fmt(x):
    return f"{x[0]:+.2f} [{x[1]:+.2f}, {x[2]:+.2f}]"
for key in ["A_same_edit_cleaner_bg", "B_same_bg_stronger_edit"]:
    print("==", key)
    for r in res[key]:
        print(f"{r['method'][:62]:62s} CLIPed {r['CLIP_edited']:.2f} {fmt(r['d_CLIP_edited'])} | PSNR {r['BG_PSNR']:.2f} {fmt(r['d_BG_PSNR'])} | LPIPS {r['BG_LPIPS']:.3f} | cfgs {r['configs_used']}")
    print("table:", res[key + "_table"])
print("knee:", {k: round(v, 3) for k, v in res["mean_knee_t_by_category"].items()})
print("dominating %:", round(res["share_test_images_with_a_dominating_config_%"], 1))
