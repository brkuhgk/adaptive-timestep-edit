| Method | J | dJ | dJ lo | dJ hi | Gap closed % (vs default) | Gap closed % (vs global best) | CLIP edited | BG PSNR | BG LPIPS | Struct. dist. | Extra UNet calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fixed: paper default | 0.279 | 0.000 | 0.000 | 0.000 | 0.000 | -156.144 | 22.303 | 22.758 | 0.116 | 0.029 | 0.000 |
| Fixed: dev global best | 0.738 | 0.459 | 0.400 | 0.520 | 60.959 | 0.000 | 20.756 | 27.734 | 0.055 | 0.009 | 0.000 |
| Rule: category table (true label) | 0.738 | 0.459 | 0.400 | 0.520 | 60.959 | 0.000 | 20.756 | 27.734 | 0.055 | 0.009 | 0.000 |
| Prompt keywords -> category table | 0.738 | 0.459 | 0.400 | 0.520 | 60.959 | 0.000 | 20.756 | 27.734 | 0.055 | 0.009 | 0.000 |
| LLM classifier -> category table | 0.738 | 0.459 | 0.400 | 0.520 | 60.959 | 0.000 | 20.756 | 27.734 | 0.055 | 0.009 | 0.000 |
| Velocity probe (knee, tau=0.3) | 0.586 | 0.307 | 0.256 | 0.359 | 40.757 | -51.748 | 21.119 | 25.816 | 0.069 | 0.014 | 28.000 |
| Oracle: per image (upper bound) | 1.032 | 0.753 | 0.710 | 0.802 | 100.000 | 100.000 | 23.298 | 25.919 | 0.072 | 0.016 | nan |
