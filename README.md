# Noise Geometry Determines Audit Efficiency: Restoring Grokking Under Structured Label Noise with Targeted Annotation Review

**Dylan Ashraf** — *Journal of High School Science*

---

## Overview

This repository contains the code and experimental results needed to reproduce the paper. The study examines how the spatial geometry of label corruption affects grokking dynamics and the efficiency of label correction under a fixed annotation budget, using modular addition (predicting (a + b) mod 97) as a controlled testbed.

**Key findings:**
- Loss-based audit triage achieves ~15× the hit rate of random auditing at a budget of just 4% of the training set, and is the only policy that reliably restores grokking.
- Region auditing is ~4× more efficient than random under input-dependent noise but provides no advantage under uniform noise — noise geometry determines audit strategy value.
- The loss-based advantage holds across noise rates η ∈ {0.01, 0.04, 0.08, 0.12} and generalizes to a softmax-entropy (uncertainty) scorer.
- Grokking recovery is optimization-history dependent: correcting labels mid-training without reinitialization grokks faster than retraining from scratch on the same corrected data (hysteresis).

---

## Scripts

| File | Description |
|------|-------------|
| `run_audit_experiment.py` | Main experiment — audit budget policies (none, random, region, loss, uncertainty, forgetting), noise injection, checkpointing, resume |
| `run_temporal_experiment.py` | Temporal path-dependence experiments (clean / A / B / C / D conditions) |
| `projectA_grokking.py` | Baseline grokking pipeline (clean labels + noise sweep) |
| `make_figures.py` | Generates audit-experiment figures from `summary.csv` + history JSONs |
| `make_temporal_figures.py` | Generates temporal-experiment figures |

## Result directories

| Directory | Contents |
|-----------|----------|
| `runs_audit_pilot/` | Main audit experiment, η = 0.04, both noise types, all policies, budgets 0–800 |
| `runs_eta_sweep/` | Noise-rate sweep, η ∈ {0.01, 0.08, 0.12} (126 runs) |
| `runs_alt_scorers/` | Alternative scorer sweep (loss / uncertainty / forgetting / random), η = 0.04 (72 runs) |
| `runs_temporal_full/` | Temporal path-dependence experiments (30 runs) |
| `runs_temporal_smoke/` | Quick smoke-test runs for the temporal pipeline |
| `figures_final/`, `figures_temporal/` | Generated figures (Figures 1–10) |

Each run directory contains a `summary.csv` (one row per run) plus per-run `*_summary.json` and `*_history.json` files.

---

## Reproduce

**Install dependencies:**
```bash
pip install torch numpy matplotlib pandas seaborn
```

**Main audit experiment (η = 0.04, seeds 0–2, both noise types, all policies):**
```bash
python run_audit_experiment.py \
  --seeds 0 1 2 --etas 0.04 \
  --noise-types input_dep uniform \
  --policies none random_audit region_audit loss_audit \
  --budgets 0 200 400 800 \
  --loss-warmup-epochs 2000 \
  --out-dir runs_audit_pilot
```

**Noise-rate sweep:**
```bash
python run_audit_experiment.py \
  --out-dir runs_eta_sweep \
  --etas 0.01 0.08 0.12 --seeds 0 1 2 \
  --noise-types input_dep uniform \
  --policies none random_audit region_audit loss_audit \
  --budgets 0 400 800
```

**Alternative scorers:**
```bash
python run_audit_experiment.py \
  --out-dir runs_alt_scorers \
  --policies uncertainty_audit forgetting_audit loss_audit random_audit \
  --etas 0.04 --budgets 200 400 800 \
  --seeds 0 1 2 --noise-types input_dep uniform
```

**Temporal path-dependence experiments:**
```bash
python run_temporal_experiment.py \
  --out-dir runs_temporal_full \
  --seeds 0 1 2 --noise-types input_dep uniform \
  --eta 0.04 --transition-epoch 50000 --epochs 200000
```

**Generate figures:**
```bash
python make_figures.py --in-dir runs_audit_pilot --out-dir figures_final
python make_temporal_figures.py --in-dir runs_temporal_full --out-dir figures_temporal
```

All runs are resume-safe: interrupted runs pick up from the last checkpoint automatically.

---

## Results (main experiment, η = 0.04, 3 seeds)

| Policy | Budget | Input-dep. hit rate | Uniform hit rate | Input-dep. grokked |
|--------|--------|--------------------|-----------------|--------------------|
| Random | 200 | 3.8% | 3.0% | 0/3 |
| Region | 200 | 17.0% | 3.8% | 0/3 |
| Loss | 200 | 45.3% | 41.7% | 1/3 |
| Loss | 400 | 29.4% | 22.7% | 3/3 |
| Loss | 800 | 17.7% | 13.3% | 3/3 |

Full results: `runs_audit_pilot/summary.csv` (and the corresponding `summary.csv` in each other run directory).

---

## Hardware

Experiments were run on an Apple M4 Max (MPS, float32). All runs are resume-safe via checkpointing.

---

## Citation

```
Ashraf, D. Noise Geometry Determines Audit Efficiency: Restoring Grokking
Under Structured Label Noise with Targeted Annotation Review.
Journal of High School Science.
```
