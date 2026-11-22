---
title: "Choosing the Noise Level from the Prompt: Adaptive Timesteps for One-Step Image Editing"
author: "[Your Name] ([email] · [GitHub])"
date: "December 2026"
---

<!-- Draft. Run `python scripts/07_fill_report.py` after the study to replace every {{placeholder}} from
work/results/summary.json and main_table.md. Rewrite the Results and Discussion sections in your own words
once you have looked at the numbers and images. Convert with: pandoc report_filled.md -o report.pdf -->

## Abstract

One-step editors such as ChordEdit apply a single transport step at a fixed noise level, then a refinement step at a lower one. A recent reproduction study found two things. The chord window mainly acts as a timestep shift. High-noise steps make coarse, low-frequency changes, and low-noise steps add fine detail. The study proposed choosing the timestep from the prompt as a next step [Li et al. 2026].

We test that proposal on PIE-Bench with SD-Turbo. First, we sweep {{n_configs}} timestep configurations over all 700 images to measure how much quality a fixed setting leaves on the table. Second, we compare three cheap ways to choose the timesteps per edit:

- a category table, applied with the true category label or with one predicted from the prompt pair;
- a small-LLM edit-type classifier;
- a training-free **velocity probe**, which finds the smallest noise level at which the target and source prompts' predictions disagree.

Everything is tuned on a dev split and reported on a held-out test split of {{n_test}} images. The best practical predictor, {{best_practical_method}}, closes **{{best_practical_gap_closed_%:.0f}}%** of the gap between the paper's default and a per-image oracle. Its score gain is ΔJ = {{best_practical_dJ:.3f}} (95% CI [{{best_practical_dJ_lo:.3f}}, {{best_practical_dJ_hi:.3f}}]), at a cost of {{best_practical_extra_unet_calls:.0f}} extra batched UNet evaluations.

## 1. Introduction

Choosing the noise level is the oldest trade-off in diffusion editing. In SDEdit [Meng et al. 2022], more noise buys editability at the cost of faithfulness to the source. Few-step and one-step editors make the choice starker, because there is no trajectory to spread the change over.

ChordEdit [Lu et al. 2026] performs a single "chord" transport step at t_start and a proximal clean-up at t_end. Its reproduction study [Li et al. 2026] made three findings:

- The chord window δ acts largely as an effective shift from t to t − δ.
- Setting δ = 0, t_start = 0.75 roughly matches the default.
- Transport at high noise performs low-frequency semantic edits, while alignment at low noise adds high-frequency detail.

The study proposed *prompt-conditioned dynamic timestep selection* but did not test it.

We ask three questions:

1. How much does the best timestep vary across edits?
2. How much quality does a fixed setting lose?
3. How cheaply can that quality be recovered?

Our contributions:

1. A full sweep of (t_start, t_end) on PIE-Bench, with dev/test splits, multiple seeds, paired bootstrap intervals and a held-out-seed oracle that avoids winner's-curse optimism.
2. A training-free probe that turns the model's own prompt disagreement into a timestep, with a short argument for why it should work.
3. All code, outputs and per-image metrics, released so others can rerun every number.

## 2. Related work

**Noise level in editing.** SDEdit [Meng et al. 2022] exposes the noise strength as a user knob. DDS [Hertz et al. 2023] and FlowEdit [Kulikov et al. 2024] integrate the *difference* between target- and source-conditioned predictions. Delta Rectified Flow Sampling [Beaudouin et al. 2026] unifies the two with a time-dependent shift.

**One-step editing.** ChordEdit [Lu et al. 2026] builds on adversarially distilled models such as SD-Turbo [Sauer et al. 2023]. The reproduction study [Li et al. 2026] is our starting point.

**Prompt-aware editing.** SplitFlow [Yoon et al. 2025] uses an LLM to split prompts into sub-edits. Our LLM classifier borrows that idea to choose timesteps instead.

**Evaluation.** We use PIE-Bench [Ju et al. 2024] with the metrics and evaluator used by the ChordEdit reproduction.

## 3. Method

**Setup.** Let x be the source latent, c_s and c_t the source and target prompts, z_t = α_t x + σ_t ε, and x̂₀(z_t, c) = (z_t − σ_t ε̂(z_t, t, c)) / α_t the model's clean-image prediction.

With δ = 0, ChordEdit's transport update is

```latex
u(t) = \mathbb{E}_{\epsilon}\left[\hat{x}_0(z_t, c_t) - \hat{x}_0(z_t, c_s)\right], \qquad x' = x + u(t_{\text{start}})
```

The clean-up step then returns x̂₀(α_{t_end} x′ + σ_{t_end} ε, c_t). The probe computes exactly the bracketed difference, d(t), at a set of candidate timesteps.

**Why the right t depends on the edit.** By Tweedie's formula, x̂₀(z_t, c) approximates E[x₀ | z_t, c], so d(t) is the difference between two posterior means, one for each prompt. For small σ_t, any image consistent with z_t must be close to x, so both posteriors collapse onto x and d(t) → 0. The edit is not yet reachable.

Adding Gaussian noise of variance σ_t² hides every frequency band whose signal power falls below roughly σ_t²/α_t². Natural images have a power spectrum that falls roughly as 1/f², so the hidden band grows from fine to coarse as t increases. An edit becomes expressible once the band it must change is hidden:

- fine-texture and small-detail edits become reachable at low t;
- color, layout and style edits, which live at low frequencies, need high t.

Beyond that point, extra noise only hides more of the background, which the model may re-imagine. This is the drift the preservation metrics penalize. The reproduction study's coarse/fine finding is the same picture seen from the output side.

**Rules for choosing t.** Two rules follow from this picture:

- **Knee rule:** choose the smallest t at which the magnitude of the applied update, s̄(t) = RMS(E_k d_k(t)), reaches a fraction τ of its maximum over the candidates. This is the lowest noise at which the edit is already expressible.
- **Consistency rule:** choose the t that maximizes s(t)·ρ(t), where ρ is the cosine agreement of d across noise draws. This favors strong *and* reproducible edit directions.

τ and the rule itself are tuned on the dev split. t_end is set to the dev-best value for the chosen t_start.

**Cost.** The probe batches |T| candidates × K = 2 noises × 2 prompts = {{probe_unet_calls:.0f}} UNet evaluations into a single forward pass. That measured **{{probe_seconds_per_image:.2f}} s per image**, against about 0.5 s for the edit itself.

**Category baselines.** A table maps each PIE edit category to its dev-best configuration. It is applied with the true label (an upper bound for any prompt classifier), with a keyword rule over the prompt word-diff, and with a 1.5B instruct LLM (Qwen2.5-1.5B-Instruct, few-shot). The classifiers reach {{keyword_classifier_acc_%:.0f}}% and {{llm_classifier_acc_%:.0f}}% accuracy respectively.

## 4. Experimental setup

**Model and data.**

- **Model:** SD-Turbo, run through the lab's ChordEdit reproduction pipeline (fp32, 512², one noise sample, one step, clean-up on).
- **Data:** PIE-Bench, 700 images in 10 edit categories.

**Grid.**

- t_start ∈ {0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0}, t_end ∈ {0.1, 0.2, 0.3, 0.4}, δ = 0.
- Plus the paper default (0.9, 0.3, δ = 0.15), for {{n_configs}} configurations in total.
- Seeds {{seeds}}.

**Metrics.** Structure distance; background PSNR, LPIPS and SSIM; and CLIP similarity of the whole image and the edited region. All are computed with the lab's PIE-Bench evaluator.

**Score.** J is the edit term plus the preservation term, each built from metrics z-scored on dev (README, "How best is defined"). `sensitivity.csv` reports the conclusions with the preservation term weighted 0.5×, 1× and 2×.

**Protocol.** The split is stratified by category: {{n_dev}} images for dev, {{n_test}} for test. All choices are fit on dev. 95% CIs come from a paired bootstrap over test images (2,000 resamples).

**Oracles.**

- In-sample per-image oracle: J = {{oracle_in_sample_J:.3f}}.
- Held-out-seed oracle, which selects on seed 42 and scores on the other seeds: J = {{oracle_held_out_seed_J:.3f}}.
- Paper default, for reference: J = {{reference_J:.3f}}.

## 5. Results

**Table 1.** Test split; methods tuned on dev.

{{MAIN_TABLE}}

**Is there a gap?** The per-image oracle beats the paper default by ΔJ = {{oracle_gap:.3f}}. The best *single* configuration on dev ({{global_best_config}}) already recovers part of that gap. [Describe: how much is left after that, and whether the CI excludes zero.]

**Does the best timestep depend on the edit?** Yes or no; see Figure 1. The best t_start ranges from {{lowest_t_category_t:.2f}} for *{{lowest_t_category_name}}* to {{highest_t_category_t:.2f}} for *{{highest_t_category_name}}*. [Compare with the coarse/fine prediction of Section 3: are color and background edits high-t, and texture or pose edits low-t?]

**Predictors.** Figure 2 shows how much of the oracle gap each predictor closes. [Say which predictor wins, and whether the probe beats the zero-cost category rule. The rank correlation between the probe's mean knee and the best t_start, across categories, is ρ_s = {{spearman_knee_vs_best_t_by_category:.2f}}.]

**CLIP-edited.** For reference:

- The best practical method reaches {{best_practical_clip_edited:.2f}}.
- The original paper reports {{paper_clip_edited:.2f}} for ChordEdit.
- The lab's reproduction reports {{lab_repro_clip_edited:.2f}}.

![Figure 1. Edit score by category and t_start (row-centered; box = best).](../work/results/fig_heatmap.png)

![Figure 2. Gain over the paper default, with the share of the oracle gap closed.](../work/results/fig_gap_closed.png)

![Figure 3. Probe signal: normalized prompt disagreement vs. timestep, by category.](../work/results/fig_probe_curves.png)

## 6. Limitations and next steps

**Limitations.**

- **Narrow setup.** The study covers one model (SD-Turbo), one benchmark (PIE-Bench) and automatic metrics only. CLIP-based edit scores are known to reward prompt keywords rather than faithful edits. A small human or VLM study, in the style of FiVE-Acc, would strengthen the claims.
- **What J rewards.** J weights edit and preservation equally. The sensitivity table shows whether the ranking survives other weightings.
- **Prompt formatting.** We feed PIE prompts as the lab's script does, with bracket markers. A clean-prompt run is a one-line config change.

**Next steps.**

- **Video.** The same probe applies to rectified-flow video editors such as Wan-Edit on FiVE-Bench [Li et al. 2025], where a single shift schedule is currently used for all edits.
- **Optimizing the step itself.** The probe's disagreement curve is the quantity that DDS, FlowEdit and Delta Rectified Flow Sampling integrate over time. Choosing *where* to evaluate it, rather than integrating it uniformly, may generalize beyond one-step editing.

## References

- Beaudouin, G., Li, M., Kim, J., Yoon, S.-H., Wang, M. Delta Rectified Flow Sampling for Text-to-Image Editing. CVPR 2026. arXiv:2509.05342
- Hertz, A., Aberman, K., Cohen-Or, D. Delta Denoising Score. ICCV 2023. arXiv:2304.07090
- Ju, X., Zeng, A., Bian, Y., Liu, S., Xu, Q. PnP Inversion: Boosting Diffusion-based Editing with 3 Lines of Code (PIE-Bench). ICLR 2024. arXiv:2310.01506
- Kulikov, V., Kleiner, M., Huberman-Spiegelglas, I., Michaeli, T. FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models. arXiv:2412.08629
- Li, M., Moebel, J., Wang, M. Rethinking One-Step Image Editing through ChordEdit: Reproduction, Simplification, and New Insights. arXiv:2606.14042, 2026
- Li, M., Xie, C., Wu, Y., Zhang, L., Wang, M. FiVE: A Fine-grained Video Editing Benchmark for Evaluating Emerging Diffusion and Rectified Flow Models. ICCV 2025. arXiv:2503.13684
- Lu, L., Chen, X., Guo, M., Li, S., Wang, J., Shi, Y. ChordEdit: One-Step Low-Energy Transport for Image Editing. CVPR 2026. arXiv:2602.19083
- Meng, C., He, Y., Song, Y., Song, J., Wu, J., Zhu, J.-Y., Ermon, S. SDEdit: Guided Image Synthesis and Editing with SDEs. ICLR 2022. arXiv:2108.01073
- Sauer, A., Lorenz, D., Blattmann, A., Rombach, R. Adversarial Diffusion Distillation. arXiv:2311.17042
- Yoon, S.-H., Li, M., Beaudouin, G., Wen, C., Azhar, M. R., Wang, M. SplitFlow: Flow Decomposition for Inversion-Free Text-to-Image Editing. NeurIPS 2025. arXiv:2510.25970
