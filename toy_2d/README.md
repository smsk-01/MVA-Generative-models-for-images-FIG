# Toy 2D Diffusion Benchmark

This module adds a small, self-contained benchmark for comparing `DPS` and `FIG-Diffusion` on two synthetic 2D datasets:

- `two_moons`
- `eight_gaussians`

It includes:

- dataset generators
- a small MLP denoiser with sinusoidal time embeddings
- DDPM-style training with linear or cosine schedules
- conditional samplers for DPS and FIG-Diffusion
- evaluation with posterior-mean MSE, ground-truth MSE, measurement MSE, and sliced Wasserstein distance
- CSV export and matplotlib scatter plots

## Train

```bash
python -m toy_2d.train \
  --dataset two_moons \
  --num_steps 20000 \
  --batch_size 512 \
  --output_dir toy_2d_outputs
```

## Benchmark

```bash
python -m toy_2d.benchmark \
  --dataset two_moons \
  --checkpoint toy_2d_outputs/two_moons/ddpm_toy_2d.pt \
  --noise_levels 0.05,0.5,1.0 \
  --fig_k_values 1,3,5 \
  --fig_w_values 0.0,0.5,1.0 \
  --dps_zeta 0.05 \
  --fig_lr 0.5 \
  --output_dir toy_2d_outputs
```

By default the observation matrix is `[1, 0]`, so the inverse problem observes the first coordinate and hides the second.
