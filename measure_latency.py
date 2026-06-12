"""
measure_latency.py

Measures per-decision inference latency for:
  - CASR:   PPOAgent.choose_action(state_21dim)
  - TASCAR: TransformerEncoder.forward(history_10x21)
            + SACAgent.choose_action(encoded_64dim, evaluate=True)

Run from the TASCAR project root:
    python measure_latency.py
"""

import time
import numpy as np
import torch

from config import *
from ppo_agent import PPOAgent
from sac_agent import SACAgent
from transformer_encoder import TransformerEncoder, StateHistoryBuffer

NUM_TRIALS = 1000
NUM_WARMUP = 50

STATE_DIM  = 21
ACTION_DIM = 27
SEQ_LEN    = 10


def main():
    # ---------------------------------------------------
    # Load CASR (PPO, single 21-dim snapshot)
    # ---------------------------------------------------
    ppo_agent = PPOAgent(STATE_DIM, ACTION_DIM)
    ppo_agent.load("trained_model/best/")

    # ---------------------------------------------------
    # Load TASCAR (Transformer encoder + SAC)
    # ---------------------------------------------------
    transformer = TransformerEncoder(STATE_DIM)
    transformer.eval()

    transformer_dim = transformer.get_output_dim()

    sac_agent = SACAgent(transformer_dim, ACTION_DIM, transformer)
    sac_agent.load("trained_model_tascar/best/")

    # ---------------------------------------------------
    # Dummy inputs matching real shapes
    # ---------------------------------------------------
    dummy_state_casr = np.random.randn(STATE_DIM).astype(np.float32)

    # 10-step history of 21-dim states -> (1, 10, 21)
    dummy_history = torch.randn(1, SEQ_LEN, STATE_DIM)

    # ---------------------------------------------------
    # Warm-up (avoid first-call overhead skew)
    # ---------------------------------------------------
    for _ in range(NUM_WARMUP):
        _ = ppo_agent.choose_action(dummy_state_casr)

        with torch.no_grad():
            encoded = transformer(dummy_history)
        _ = sac_agent.choose_action(
            encoded.numpy(), evaluate=True)

    # ---------------------------------------------------
    # Measure CASR (PPO only)
    # ---------------------------------------------------
    casr_times = []
    for _ in range(NUM_TRIALS):
        start = time.perf_counter()
        _ = ppo_agent.choose_action(dummy_state_casr)
        casr_times.append((time.perf_counter() - start) * 1000)

    # ---------------------------------------------------
    # Measure TASCAR (Transformer + SAC)
    # ---------------------------------------------------
    tascar_times = []
    transformer_times = []
    sac_times = []

    for _ in range(NUM_TRIALS):
        start_total = time.perf_counter()

        start_t = time.perf_counter()
        with torch.no_grad():
            encoded = transformer(dummy_history)
        t_transformer = (time.perf_counter() - start_t) * 1000

        start_s = time.perf_counter()
        _ = sac_agent.choose_action(
            encoded.numpy(), evaluate=True)
        t_sac = (time.perf_counter() - start_s) * 1000

        total = (time.perf_counter() - start_total) * 1000

        transformer_times.append(t_transformer)
        sac_times.append(t_sac)
        tascar_times.append(total)

    # ---------------------------------------------------
    # Report
    # ---------------------------------------------------
    casr_mean, casr_std = np.mean(casr_times), np.std(casr_times)
    tascar_mean, tascar_std = np.mean(tascar_times), np.std(tascar_times)
    trans_mean, trans_std = np.mean(transformer_times), np.std(transformer_times)
    sac_mean, sac_std = np.mean(sac_times), np.std(sac_times)

    overhead = tascar_mean - casr_mean
    amortized_us = overhead / TASCAR_EVAL_DELTA * 1000  # microseconds/request

    print("=" * 60)
    print(f"Trials per configuration: {NUM_TRIALS}")
    print("=" * 60)
    print(f"CASR   (PPO only):            "
          f"{casr_mean:.4f} +/- {casr_std:.4f} ms")
    print(f"TASCAR (Transformer + SAC):   "
          f"{tascar_mean:.4f} +/- {tascar_std:.4f} ms")
    print(f"  -- Transformer component:   "
          f"{trans_mean:.4f} +/- {trans_std:.4f} ms")
    print(f"  -- SAC actor component:     "
          f"{sac_mean:.4f} +/- {sac_std:.4f} ms")
    print("-" * 60)
    print(f"Overhead per decision:         {overhead:.4f} ms")
    print(f"Decisions amortized over:      "
          f"{TASCAR_EVAL_DELTA} requests (TASCAR_EVAL_DELTA)")
    print(f"Amortized overhead/request:    {amortized_us:.6f} microseconds")
    print("=" * 60)


if __name__ == "__main__":
    main()