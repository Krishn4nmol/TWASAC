# TASCAR: Transformer-Attention Soft Actor-Critic for Adaptive Resource Optimization in Serverless Computing

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-brightgreen)
![Beats CASR](https://img.shields.io/badge/Beats%20CASR-13--18pp-red)
![Seeds](https://img.shields.io/badge/Seeds-42%20%7C%20123%20%7C%20456-purple)
![Metrics](https://img.shields.io/badge/Metrics-18%20Comprehensive-blue)

---

## Overview

**TASCAR** is a novel serverless container scheduling system that extends and significantly outperforms **CASR** (Chen et al., *Future Generation Computer Systems*, 2025).

TASCAR replaces CASR's PPO reinforcement learning agent with three architectural innovations:

- **Transformer encoder** — processes sequences of 10 historical cache states with cross-queue attention, capturing temporal workload patterns invisible to single-snapshot approaches
- **Soft Actor-Critic (SAC)** — off-policy learning with dual critics and entropy-driven exploration, replacing PPO's on-policy updates
- **Dynamic reward weighting** — automatically adapts θ (0.5–0.9) based on observed cold start rate, replacing CASR's fixed θ=0.8

> **Key result:** TASCAR reduces cold start rate by **13.41 to 18.51 percentage points** over CASR across three independent random seeds (42, 123, 456), with near-zero cross-seed variance of 0.00–0.02 pp, while maintaining zero wasted memory time.

---

## Results

### Primary Cold Start Rate Comparison (Seed 42)

| Workload | CASR CSR | TASCAR CSR | Improvement |
|----------|----------|------------|-------------|
| Common | 89.840% | 72.111% | **−17.729 pp** |
| Significant | 92.024% | 74.973% | **−17.051 pp** |
| Random | 86.051% | 72.377% | **−13.674 pp** |

### Multi-Seed Validation (Seeds 42, 123, 456)

| Workload | CASR (mean ± std) | TASCAR (mean ± std) | Diff (pp) |
|----------|-------------------|---------------------|-----------|
| Common | 90.25 ± 0.54% | 71.86 ± 0.02% | 18.39 ± 0.54 |
| Significant | 93.07 ± 0.83% | 74.55 ± 0.00% | 18.51 ± 0.83 |
| Random | 84.26 ± 5.43% | 70.84 ± 0.01% | 13.41 ± 5.43 |

TASCAR's cross-seed standard deviation (0.00–0.02 pp) is an order of magnitude lower than CASR's (0.54–5.43 pp).

### Equal-Budget Comparison (Common Workload)

| Configuration | Episodes | CSR |
|---------------|----------|-----|
| CASR-200 (original) | 200 | 89.105% |
| CASR-500 (retrained) | 500 | 87.869% |
| TASCAR | 500 | 72.101% |

CASR-500 recovers only ~7% of the gap with 2.5× more training. TASCAR's advantage is architectural, not a training-budget effect.

### Ablation Study: V1 vs V4 Across Seeds (CSR %)

| Workload | Seed | V1 (CASR) | V4 (TASCAR) | Diff |
|----------|------|-----------|-------------|------|
| Common | 42 | 88.796 | 72.146 | 16.650 |
| Common | 123 | 90.989 | 72.111 | 18.878 |
| Common | 456 | 89.146 | 72.114 | 17.032 |
| Significant | 42 | 95.626 | 75.521 | 20.105 |
| Significant | 123 | 83.726 | 75.521 | 8.205 |
| Significant | 456 | 95.238 | 75.525 | 19.713 |
| Random | 42 | 89.199 | 68.950 | 20.249 |
| Random | 123 | 89.286 | 68.940 | 20.346 |
| Random | 456 | 90.169 | 68.940 | 21.229 |

V4 (Full TASCAR) outperforms V1 (CASR) on every single (seed, workload) combination. Minimum improvement: 8.205 pp.

### Baseline Comparison (Seed 42)

| Workload | TASCAR vs CASR | TASCAR vs FaaSCache | TASCAR vs Hist |
|----------|----------------|---------------------|----------------|
| Common | +17.729 pp | +27.888 pp | −10.897 pp |
| Significant | +17.051 pp | +25.027 pp | −13.331 pp |
| Random | +13.674 pp | +27.623 pp | −10.996 pp |

TASCAR outperforms FaaSCache and CASR while maintaining **zero wasted memory time** — a balance Hist cannot achieve (12.7–25.7s WMT).

### Comprehensive Metrics Summary (Seed 42)

| Metric | Winner |
|--------|--------|
| Cold Start Rate | TASCAR ✅ |
| P95 / P99 Latency | TASCAR ✅ |
| Average Response Time | TASCAR ✅ |
| Container Utilization (98–214% improvement) | TASCAR ✅ |
| Resource Utilization Efficiency | TASCAR ✅ |
| SLA Violation Rate | TASCAR ✅ |
| Energy per Request | TASCAR ✅ |
| CO2 Estimate (10.4–16.8% reduction) | TASCAR ✅ |
| TPI Composite Score | TASCAR ✅ |
| Wasted Memory Time | Tie (both 0.000s) |
| Throughput / Burst Handling | Tie |
| Avg Cold Start Delay | CASR |
| Scaling Accuracy / Elasticity Score | CASR |

**TASCAR wins 11/18, ties 4/18, CASR wins 3/18.**

### RL Training Metrics (Seed 42)

| Metric | Value |
|--------|-------|
| Training Time | 4512.3 seconds (~75 min) |
| Best Reward | −0.1351 |
| Best Checkpoint | Episode 350 |
| Total Training Samples | 50,000 |
| Sample Efficiency | −0.0270 |
| Cumulative Reward | −310.00 |
| Random Seed | 42 |
| Episodes | 500 |

---

## Architecture

```
Azure Function Traces
        │
        ▼
  S-Cache (W-TinyLFU, K=3 queues)
        │ state vector (21-dim)
        ▼
  State History Buffer (last 10 states)
        │ sequence (10 × 21)
        ▼
  Transformer Encoder
  ├── Linear projection: 21 → 64
  ├── Positional encoding
  ├── 2 Transformer layers, 4 heads
  ├── Cross-queue attention
  └── Output: 64-dim encoded state
        │
        ▼
  SAC Agent
  ├── Actor:    64 → 128 → 128 → 27
  ├── Critic 1: 64 → 128 → 128 → 27
  └── Critic 2: 64 → 128 → 128 → 27
        │ action (0–26)
        ▼
  Dynamic Reward θ (adapts 0.5–0.9)
        │ scale action applied
        ▼
  S-Cache ◄──────────────────────────┘

  MetricsTracker (non-invasive wrapper)
  └── 18 metrics computed at evaluation
```

### TASCAR vs CASR: Architecture Comparison

| Component | CASR | TASCAR |
|-----------|------|--------|
| RL Algorithm | PPO (on-policy) | SAC (off-policy) |
| State Input | Single snapshot (21-dim) | Sequence of 10 states |
| Temporal Model | None | Transformer encoder |
| Cross-queue reasoning | Independent | Cross-queue attention |
| Reward weighting | Fixed θ=0.8 | Dynamic θ (0.5–0.9) |
| Exploration | Clipped gradient | Entropy temperature |
| Decisions/episode | 10 | 100 |
| Sample reuse | No | Yes (replay buffer) |
| Critics | 1 | 2 (reduces overestimation) |

---

## Project Structure

```
TASCAR/
├── config.py                    ← All hyperparameters
├── simulator.py                 ← Azure dataset loader
├── scache.py                    ← W-TinyLFU S-Cache (K=3 queues)
├── transformer_encoder.py       ← Transformer + StateHistoryBuffer
├── sac_agent.py                 ← SAC with dual critics
├── train_tascar.py              ← Main training script
├── evaluate_tascar.py           ← Full CASR vs TASCAR evaluation
├── multiseed_ablation_eval.py   ← 12-model ablation evaluation
├── eval_n5.py                   ← Extended seed evaluation
├── run_tascar_seed42.py         ← Seed 42 wrapper
├── run_tascar_seed123.py        ← Seed 123 wrapper
├── run_tascar_seed456.py        ← Seed 456 wrapper
├── run_sac_only_seed123.py      ← Ablation V2 seed 123
├── run_sac_only_seed456.py      ← Ablation V2 seed 456
├── run_transformer_ppo_seed123.py ← Ablation V3 seed 123
├── run_transformer_ppo_seed456.py ← Ablation V3 seed 456
├── train_casr200_seed123.py     ← Ablation V1 seed 123
├── train_casr200_seed456.py     ← Ablation V1 seed 456
├── requirements.txt
├── results_tascar/
│   ├── casr_vs_tascar.json
│   ├── fig1_cold_start.png
│   ├── fig2_latency_memory.png
│   ├── fig3_resource.png
│   ├── fig4_qos_throughput.png
│   ├── fig5_energy_scaling.png
│   ├── fig6_tpi_agi.png
│   ├── fig7_rl_metrics.png
│   └── fig8_master_all_metrics.png
├── results_tascar_seed123/
├── results_tascar_seed456/
└── results_ablation/
```

---

## Generated Figures

All figures saved to `results_tascar/`:

| Figure | File | Description |
|--------|------|-------------|
| Fig 1 | `fig1_cold_start.png` | CSR, ACSD, P95 across workloads |
| Fig 2 | `fig2_latency_memory.png` | P99, ART, WMT across workloads |
| Fig 3 | `fig3_resource.png` | CUR, RUE, SER across workloads |
| Fig 4 | `fig4_qos_throughput.png` | SVR, TPT, BHE across workloads |
| Fig 5 | `fig5_energy_scaling.png` | EPR, CO2, SA across workloads |
| Fig 6 | `fig6_tpi_agi.png` | TPI and AGI composite scores |
| Fig 7 | `fig7_rl_metrics.png` | Training curves (reward, cold%, θ) |
| Fig 8 | `fig8_master_all_metrics.png` | All 18 metrics overview |

---

## Installation

### Requirements

- Python 3.11
- 32GB RAM recommended
- No GPU required (CPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/Krishn4nmol/TASCAR.git
cd TASCAR

# Create and activate conda environment
conda create -n casr_env python=3.11
conda activate casr_env

# Install dependencies
pip install -r requirements.txt

# Download Azure Functions 2019 dataset
# https://github.com/Azure/AzurePublicDataset
# Place CSV files in data/ folder
```

---

## How to Run

### Step 1: Train TASCAR (seed 42)

```bash
python run_tascar_seed42.py
```

Takes ~75 minutes. Checkpoints saved every 50 episodes. Best checkpoint saved automatically.

### Step 2: Multi-seed training

```bash
python run_tascar_seed123.py
python run_tascar_seed456.py
```

### Step 3: Full evaluation

```bash
python evaluate_tascar.py
```

Generates all 8 figures and comprehensive metrics JSON. Takes ~90 minutes.

### Step 4: Ablation study

```bash
python train_casr200_seed123.py
python train_casr200_seed456.py
python run_sac_only_seed123.py
python run_sac_only_seed456.py
python run_transformer_ppo_seed123.py
python run_transformer_ppo_seed456.py
python multiseed_ablation_eval.py
```

Trains 12 models total (4 variants × 3 seeds) and evaluates all.

---

## Dataset

**Microsoft Azure Functions 2019**

- Source: [Azure Public Dataset](https://github.com/Azure/AzurePublicDataset)
- Training: Days 1–5 (top 2,000 functions)
- Evaluation workloads:
  - **Common** — top 2,000 frequent functions, Day 1
  - **Significant** — top 2,000 high cold-start-overhead functions, Day 2
  - **Random** — 2,000 randomly selected functions, Day 3

---

## Hyperparameters

| Parameter | CASR | TASCAR |
|-----------|------|--------|
| Episodes | 200 | 500 |
| Steps/episode | 10 | 100 |
| Learning rate | 0.001 | 0.0001 |
| Replay buffer | — | 100,000 |
| Batch size | 20 | 64 |
| SAC updates/step | — | 10 |
| Sequence length | 1 | 10 |
| Transformer dim | — | 64 |
| θ | 0.8 (fixed) | 0.5–0.9 (dynamic) |
| Random seeds | 42 | 42, 123, 456 |

---

## Metrics Reference

| Category | Metric | Better |
|----------|--------|--------|
| Cold Start | CSR, ACSD, P95, P99 | Lower |
| Resource | CUR, RUE | Higher |
| Resource | WMT | Lower |
| QoS | ART, SVR | Lower |
| Throughput | TPT, SER, BHE | Higher |
| Energy | EPR, CO2 | Lower |
| Scalability | SA, ES | Higher |
| Composite | TPI, AGI | Higher |

**TPI formula:**
```
TPI = 0.25×(1−CSR) + 0.20×(1−WMT_n) + 0.20×TPT_n + 0.20×(1−SVR) + 0.15×RUE
```

---

## Citation

Paper under review. Citation will be added upon publication.

---

## Acknowledgment

This work was conducted under the guidance of Dr. Rajiv Misra, Department of Computer Science and Engineering, IIT Patna. The Azure Functions 2019 dataset is provided by Microsoft Research. This work builds on the CASR framework by Chen et al. (2025).

---

## License

MIT License — see LICENSE file for details.
