import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ate.analysis import Study, paired_bootstrap  # noqa: E402
from ate.config import EditConfig, load_settings  # noqa: E402
from ate.editor import FakeChordPipeline, compute_prompt_deltas  # noqa: E402
from ate.pie import PieItem, clean_prompt, make_split, mask_decode  # noqa: E402
from ate.probe_math import consistency_select, curve_statistics, knee_select  # noqa: E402
from ate.prompt_classifier import keyword_category  # noqa: E402


def _reference_mask_decode(encoded_mask, image_shape=(512, 512)):
    """Verbatim logic of PnPInversion/evaluation/evaluate.py::mask_decode."""
    length = image_shape[0] * image_shape[1]
    mask_array = np.zeros((length,))
    for i in range(0, len(encoded_mask), 2):
        splice_len = min(encoded_mask[i + 1], length - encoded_mask[i])
        for j in range(splice_len):
            mask_array[encoded_mask[i] + j] = 1
    mask_array = mask_array.reshape(image_shape[0], image_shape[1])
    mask_array[0, :] = 1
    mask_array[-1, :] = 1
    mask_array[:, 0] = 1
    mask_array[:, -1] = 1
    return mask_array


def test_mask_decode_matches_lab_evaluator():
    rng = np.random.RandomState(1)
    starts = np.sort(rng.choice(512 * 512 - 600, 12, replace=False))
    rle = []
    for s in starts:
        rle += [int(s), int(rng.randint(1, 500))]
    rle += [512 * 512 - 10, 50]  # run past the end must be clipped
    np.testing.assert_array_equal(mask_decode(rle), _reference_mask_decode(rle))


def test_config_ids_roundtrip_and_grid_dedup():
    ec = EditConfig(0.75, 0.3, 0.0, True)
    assert ec.id == "ts0.75_te0.30_d0.00_c1"
    assert EditConfig.from_id(ec.id) == ec
    s = load_settings()
    ids = [c.id for c in s.grid()]
    assert len(ids) == len(set(ids))
    assert s.named["paper_simplified"] in ids and s.named["paper_default"] in ids
    assert len(ids) == 7 * 4 + 1  # simplified config is already on the grid


def test_split_is_stratified_and_deterministic():
    items = [PieItem(f"{c}{k:03d}", f"{c}/x{k}.jpg", "a", "b", "", c, ()) for c in range(10) for k in range(70)]
    a = make_split(items, 0.3, 0)
    b = make_split(items, 0.3, 0)
    assert a == b
    for c in range(10):
        dev = sum(a[f"{c}{k:03d}"] == "dev" for k in range(70))
        assert dev == 21


def test_clean_prompt():
    assert clean_prompt("a [red]  car on [the] road ") == "a red car on the road"


def test_knee_and_consistency_rules():
    ts = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    s = [0.1, 0.3, 0.8, 0.95, 1.0, 1.0]
    assert knee_select(ts, s, 0.5) == 0.7
    assert knee_select(ts, s, 0.9) == 0.8
    assert knee_select(ts[::-1], s[::-1], 0.9) == 0.8  # order-independent
    assert consistency_select(ts, s, [1, 1, 1, 0.2, 0.2, 0.2]) == 0.7


def test_curve_statistics_properties():
    g = torch.Generator().manual_seed(0)
    base = torch.randn(1, 1, 4, 16, 16, generator=g)
    same = base.repeat(3, 2, 1, 1, 1) * torch.tensor([1.0, 2.0, 3.0]).view(3, 1, 1, 1, 1)
    st = curve_statistics(same)
    np.testing.assert_allclose(st["rho"], 1.0, atol=1e-5)          # identical directions
    np.testing.assert_allclose(st["s"] / st["s"][0], [1, 2, 3], rtol=1e-5)
    np.testing.assert_allclose(st["s_bar"], st["s"], rtol=1e-5)
    smooth = torch.ones(1, 1, 4, 16, 16)
    assert curve_statistics(smooth)["lf"][0] == pytest.approx(1.0)


def test_probe_matches_chordedit_update_direction():
    """With t_delta=0, ChordEdit's transport update is mean_k[x0(z_t^k,c_tgt) - x0(z_t^k,c_src)].
    The probe at that t must return exactly this quantity (per noise)."""
    pipe = FakeChordPipeline()
    from PIL import Image

    img = Image.fromarray((np.random.RandomState(0).rand(64, 64, 3) * 255).astype("uint8"))
    deltas, calls = compute_prompt_deltas(pipe, img, "a cat", "a dog", [0.6, 0.9], noise_samples=2, seed=7, chunk=3)
    assert deltas.shape[:2] == (2, 2) and calls == 8
    x = pipe._encode_image_to_latent(pipe._prepare_image_tensor(img))
    noises = pipe._prepare_noise_list(x, 7, 2)
    src, tgt = pipe.encode_prompt(["a cat"]), pipe.encode_prompt(["a dog"])
    for ti, t in enumerate([0.6, 0.9]):
        idx = pipe._time_to_index(1, t, None)
        a, s = pipe._get_alpha_sigma(x, idx)
        for k, eps in enumerate(noises):
            z = a * x + s * eps
            manual = ((z - s * pipe._predict_noise(z, idx, tgt)) / a) - ((z - s * pipe._predict_noise(z, idx, src)) / a)
            torch.testing.assert_close(deltas[ti, k], manual[0], rtol=1e-5, atol=1e-5)


def test_keyword_classifier_examples():
    assert keyword_category("a blue car", "a red car") == 6
    assert keyword_category("a wooden chair", "a glass chair") == 7
    assert keyword_category("a photo of a cat", "a watercolor painting of a cat") == 9
    assert keyword_category("a table", "a table with a vase") == 2
    assert keyword_category("a man with a hat", "a man") == 3


def _toy_study():
    configs = ["ts0.50_te0.20_d0.00_c1", "ts0.90_te0.20_d0.00_c1", "ts0.90_te0.30_d0.15_c1"]
    rows = []
    rng = np.random.RandomState(0)
    samples = [f"s{i}" for i in range(40)]
    category = {s: (0 if i < 20 else 1) for i, s in enumerate(samples)}
    split = {s: ("dev" if i % 2 == 0 else "test") for i, s in enumerate(samples)}
    for s in samples:
        for c in configs:
            good = (category[s] == 0 and c.startswith("ts0.50")) or (category[s] == 1 and c.startswith("ts0.90_te0.20"))
            for seed in (1, 2):
                rows.append({
                    "sample_id": s, "config_id": c, "seed": seed,
                    "clip_similarity_target_image": 25 + (1.0 if good else 0) + 0.1 * rng.randn(),
                    "clip_similarity_target_image_edit_part": 22 + (1.0 if good else 0) + 0.1 * rng.randn(),
                    "psnr_unedit_part": 22 + 0.1 * rng.randn(),
                    "lpips_unedit_part": 0.12 + 0.001 * rng.randn(),
                    "structure_distance": 0.03 + 0.001 * rng.randn(),
                })
    return Study(pd.DataFrame(rows), split, category, "ts0.90_te0.30_d0.15_c1")


def test_category_table_and_oracles_recover_planted_structure():
    st = _toy_study()
    table = st.category_table()
    assert table[0].startswith("ts0.50") and table[1].startswith("ts0.90_te0.20")
    oracle = st.per_image_oracle(st.ids("test"))
    assert all(oracle[s] == table[st.category[s]] for s in st.ids("test"))
    assert st.held_out_seed_oracle(st.ids("test")) is not None


def test_paired_bootstrap_ci_contains_mean():
    m, lo, hi = paired_bootstrap(np.random.RandomState(0).randn(200) + 0.5, n=500)
    assert lo < m < hi and lo > 0.2
