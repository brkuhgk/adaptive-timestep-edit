"""Thin wrapper around the lab's ChordEditPipeline, plus a tiny fake pipeline for CPU smoke tests.

The real pipeline is imported from the lab repository (Harvard-AI-and-Robotics-Lab/ChordEdit-Reproduction),
so every edit in this project uses exactly the lab's implementation; we only choose its timesteps.
"""
from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from PIL import Image

from .config import EditConfig


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_lab_pipeline(lab_repo: Path, model_root: Path, base_config: Dict[str, Any],
                      device: Optional[str] = None):
    lab_repo = Path(lab_repo).resolve()
    if str(lab_repo) not in sys.path:
        sys.path.insert(0, str(lab_repo))
    from pipeline_chord import ChordEditPipeline  # type: ignore
    from run_pie_bench import expand_component_paths, paths_from_model_root  # type: ignore

    component_paths = expand_component_paths(paths_from_model_root(str(model_root), "sd"), "sd")
    default_cfg = dict(base_config)
    default_cfg.update(t_start=0.9, t_end=0.3, t_delta=0.15, cleanup=True)
    return ChordEditPipeline.from_local_weights(
        component_paths=component_paths,
        model_type="sd",
        default_edit_config=default_cfg,
        device=device,
        torch_dtype=torch.float32,
        compute_dtype=torch.float32,
        use_center_crop=True,
        use_attention_mask=False,
        use_safety_checker=False,
    )


def load_pipeline(settings, fake: bool = False, device: Optional[str] = None):
    if fake:
        return FakeChordPipeline(device=device or "cpu")
    return load_lab_pipeline(settings.path("lab_repo"), settings.path("model_root"),
                             settings.base_edit_config(), device=device)


def edit_image(pipe, image: Image.Image, source_prompt: str, target_prompt: str,
               config: EditConfig, base: Dict[str, Any], seed: int, quiet: bool = True) -> Image.Image:
    # The lab pipeline prints its timestep settings on every call; silence it for 60k-edit sweeps.
    with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
        result = pipe(
            image=image,
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            edit_config=config.as_edit_config(base),
            seed=seed,
            output_type="pil",
        )
    return result.images[0]


# --------------------------------------------------------------------------------------
# Probe: x0-prediction disagreement between target and source prompts across timesteps
# --------------------------------------------------------------------------------------
@torch.no_grad()
def compute_prompt_deltas(pipe, image: Image.Image, source_prompt: str, target_prompt: str,
                          t_candidates: Sequence[float], noise_samples: int, seed: int,
                          chunk: int = 14) -> tuple[torch.Tensor, int]:
    """Return d[t, k] = x0_hat(z_t^k, c_tgt) - x0_hat(z_t^k, c_src) in latent space, and #UNet evaluations.

    z_t^k = alpha_t * x_src + sigma_t * eps_k uses the same per-seed noises as the edit itself, so the probe
    "sees" the same noisy latents that ChordEdit's transport step will use. Shape: [T, K, C, H, W].
    """
    pixel = pipe._prepare_image_tensor(image)
    x_src = pipe._encode_image_to_latent(pixel)               # [1, C, H, W]
    src_e = pipe.encode_prompt([source_prompt])
    tgt_e = pipe.encode_prompt([target_prompt])
    noises = pipe._prepare_noise_list(latents=x_src, seed_value=int(seed), num_noises=int(noise_samples))

    samples: List[torch.Tensor] = []
    t_idx: List[torch.Tensor] = []
    conds: List[Any] = []
    alphas: List[torch.Tensor] = []
    sigmas: List[torch.Tensor] = []
    for t in t_candidates:
        idx = pipe._time_to_index(1, float(t), device=x_src.device)
        alpha, sigma = pipe._get_alpha_sigma(x_src, idx)
        for eps in noises:
            z = alpha * x_src + sigma * eps
            for cond in (src_e, tgt_e):
                samples.append(z)
                t_idx.append(idx)
                conds.append(cond)
                alphas.append(alpha)
                sigmas.append(sigma)

    x0_list: List[torch.Tensor] = []
    for start in range(0, len(samples), max(1, int(chunk))):
        sl = slice(start, start + max(1, int(chunk)))
        z_b = torch.cat(samples[sl], dim=0)
        t_b = torch.cat(t_idx[sl], dim=0)
        c_b = pipe._cat_conditions(conds[sl])
        a_b = torch.cat(alphas[sl], dim=0)
        s_b = torch.cat(sigmas[sl], dim=0)
        eps_hat = pipe._predict_noise(sample=z_b, timesteps=t_b, cond=c_b)
        x0_list.append((z_b - s_b * eps_hat) / a_b)
    x0 = torch.cat(x0_list, dim=0)
    T, K = len(t_candidates), len(noises)
    x0 = x0.reshape(T, K, 2, *x_src.shape[1:])
    deltas = (x0[:, :, 1] - x0[:, :, 0]).float().cpu()
    return deltas, int(T * K * 2)


# --------------------------------------------------------------------------------------
# Fake pipeline (CPU, no weights) mirroring the internals we use; only for smoke tests.
# --------------------------------------------------------------------------------------
class _Out:
    def __init__(self, images):
        self.images = images


class FakeChordPipeline:
    """Deterministic toy stand-in for ChordEditPipeline. Lets the whole kit run end-to-end on CPU."""

    image_size = 64

    def __init__(self, device: str = "cpu") -> None:
        self._device = torch.device(device)
        g = torch.Generator().manual_seed(0)
        self._proj = torch.randn(4, 3, generator=g) * 0.5
        self._text = torch.randn(64, 4, generator=g)
        betas = torch.linspace(1e-4, 0.02, 1000)
        self._alphas_cumprod = torch.cumprod(1 - betas, dim=0)
        self._max_unet_timestep = 999

    # --- public call mimicking the real pipeline ---
    def __call__(self, image, *, source_prompt, target_prompt, edit_config, seed, output_type="pil"):
        x = self._encode_image_to_latent(self._prepare_image_tensor(image))
        src, tgt = self.encode_prompt([source_prompt]), self.encode_prompt([target_prompt])
        noise = self._prepare_noise_list(latents=x, seed_value=seed, num_noises=1)[0]
        ts, te = float(edit_config["t_start"]), float(edit_config["t_end"])
        idx = self._time_to_index(1, ts, x.device)
        a, s = self._get_alpha_sigma(x, idx)
        z = a * x + s * noise
        x0_t = (z - s * self._predict_noise(z, idx, tgt)) / a
        x0_s = (z - s * self._predict_noise(z, idx, src)) / a
        x = x + float(edit_config.get("step_scale", 1.0)) * (x0_t - x0_s)
        if edit_config.get("cleanup", True):
            idx = self._time_to_index(1, te, x.device)
            a, s = self._get_alpha_sigma(x, idx)
            z = a * x + s * noise
            x = (z - s * self._predict_noise(z, idx, tgt)) / a
        img = torch.sigmoid(torch.einsum("bchw,cd->bdhw", x, self._proj))[0]
        arr = (img.permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        return _Out([Image.fromarray(arr).resize((512, 512))])

    # --- internals used by compute_prompt_deltas ---
    def _prepare_image_tensor(self, image):
        img = image.convert("RGB").resize((self.image_size, self.image_size))
        import numpy as np

        t = torch.from_numpy(np.asarray(img, dtype=np.float32).copy())
        return (t.permute(2, 0, 1)[None] / 127.5) - 1.0

    def _encode_image_to_latent(self, pixel):
        lat = torch.nn.functional.avg_pool2d(pixel, 8)
        return torch.einsum("bdhw,cd->bchw", lat, self._proj)

    def encode_prompt(self, prompts):
        codes = [sum(ord(ch) for ch in p) % 64 for p in prompts]
        return self._text[codes][:, None, :]  # [B, 1, 4]

    def _prepare_noise_list(self, latents, seed_value, num_noises):
        g = torch.Generator().manual_seed(int(seed_value))
        return [torch.randn(latents.shape, generator=g) for _ in range(num_noises)]

    def _time_to_index(self, batch, t_scalar, device, dtype=torch.long):
        idx = max(0, min(self._max_unet_timestep, round(self._max_unet_timestep * float(t_scalar))))
        return torch.full((batch,), idx, dtype=dtype)

    def _get_alpha_sigma(self, tensor, timesteps):
        ac = self._alphas_cumprod[timesteps]
        return ac.sqrt().view(-1, 1, 1, 1).clamp(min=1e-6), (1 - ac).sqrt().view(-1, 1, 1, 1)

    def _cat_conditions(self, conditions):
        return torch.cat(list(conditions), dim=0)

    def _predict_noise(self, sample, timesteps, cond):
        # Toy "UNet": the prompt shifts the predicted noise, more strongly at higher noise levels.
        tt = (timesteps.float() / 1000.0).view(-1, 1, 1, 1)
        shift = cond[:, 0, :].view(-1, cond.shape[-1], 1, 1)
        smooth = torch.nn.functional.avg_pool2d(sample, 3, stride=1, padding=1)
        return 0.9 * (sample - smooth) + tt * shift
