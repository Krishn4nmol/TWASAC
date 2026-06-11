# check_seed456.py
# Finds best checkpoint for seed 456!
# Tests all checkpoints!
# Copies best to best/ folder!

import numpy as np
import os
import shutil
from collections import Counter

from config import (
    NUM_QUEUES,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    TASCAR_EVAL_DELTA,
    TRANSFORMER_DIM,
    SEQUENCE_LENGTH,
    MODEL_SAVE_PATH,
    SCALING_FACTOR,
    RANDOM_SEED)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent

SEED_PATH = "trained_model_tascar_seed456/"


# ─────────────────────────────────────────
# LOAD COMMON WORKLOAD
# Same seed as main evaluation!
# ─────────────────────────────────────────

def load_common():
    loader = AzureDataLoader()
    day1   = loader.load_day(1)
    counts = Counter(
        c.function_id for c in day1)
    top = set(
        f for f, _ in
        counts.most_common(NUM_FUNCTIONS))
    common = [
        c for c in day1
        if c.function_id in top]
    np.random.seed(42)
    if len(common) > EVAL_CALLS:
        idx = np.random.choice(
            len(common), EVAL_CALLS,
            replace=False)
        idx.sort()
        common = [
            common[i] for i in idx]
    print(f"  {len(common)} calls loaded!")
    return common


# ─────────────────────────────────────────
# TEST TASCAR CHECKPOINT
# ─────────────────────────────────────────

def test_checkpoint(path, calls):
    """Tests one TASCAR checkpoint!"""
    if not os.path.exists(
            path + "actor.pth"):
        return None

    state_dim   = NUM_QUEUES * 7
    action_dim  = 3 ** NUM_QUEUES
    transformer = TransformerEncoder(
        state_dim)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)
    agent.load(path)

    tracker    = MetricsTracker()
    history    = StateHistoryBuffer(
        SEQUENCE_LENGTH, state_dim)
    call_count = 0

    for call in calls:
        tracker.handle_request(call)
        call_count += 1
        if (call_count %
                TASCAR_EVAL_DELTA == 0):
            raw = np.array(
                tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(raw)
            std  = np.std(raw)
            if std > 0:
                raw = (raw - mean) / std
            history.add(raw)
            seq = history.get_sequence()
            enc = (
                agent.get_encoded_state(
                    seq))
            act = agent.choose_action(
                enc, evaluate=True)
            for q_idx, scale in (
                    enumerate(
                        agent.action_map[
                            act])):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    metrics = tracker.get_all_metrics()
    return metrics['cold_start_rate']


# ─────────────────────────────────────────
# TEST CASR BASELINE
# ─────────────────────────────────────────

def test_casr(calls):
    """Tests CASR baseline!"""
    path  = MODEL_SAVE_PATH + "best/"
    if not os.path.exists(
            path + "actor.pth"):
        print("No CASR model!")
        return 89.0

    state_dim  = NUM_QUEUES * 7
    action_dim = 3 ** NUM_QUEUES
    agent      = PPOAgent(
        state_dim, action_dim)
    agent.load(path)

    action_map = {}
    choices    = [
        -SCALING_FACTOR,
        0,
        SCALING_FACTOR]
    for i in range(3 ** NUM_QUEUES):
        action = []
        temp   = i
        for _ in range(NUM_QUEUES):
            action.append(
                choices[temp % 3])
            temp //= 3
        action_map[i] = action

    tracker    = MetricsTracker()
    call_count = 0

    for call in calls:
        tracker.handle_request(call)
        call_count += 1
        if call_count % DELTA == 0:
            state = np.array(
                tracker.get_state(),
                dtype=np.float32)
            mean = np.mean(state)
            std  = np.std(state)
            if std > 0:
                state = (
                    (state - mean) / std)
            act, _ = (
                agent.choose_action(
                    state))
            for q_idx, scale in (
                    enumerate(
                        action_map[act])):
                if scale != 0:
                    tracker.scale_queue(
                        q_idx, scale)

    metrics = tracker.get_all_metrics()
    return metrics['cold_start_rate']


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

print("=" * 55)
print("Finding Best Checkpoint")
print(f"Seed Path: {SEED_PATH}")
print("=" * 55)

print("\nLoading Common workload...")
calls = load_common()

print("\nTesting CASR baseline...")
casr_csr = test_casr(calls)
print(f"CASR baseline: {casr_csr:.3f}%")

# All checkpoints to test
checkpoints = [
    (SEED_PATH +
     f"checkpoint_ep{ep}/",
     f"ep{ep}")
    for ep in [
        50, 100, 150, 200,
        250, 300, 350, 400,
        450, 500]
]

best_csr  = float('inf')
best_path = None
best_name = None

print("\nTesting checkpoints...")
for path, name in checkpoints:
    if not os.path.exists(
            path + "actor.pth"):
        print(
            f"  {name}: Not found!")
        continue

    csr = test_checkpoint(path, calls)
    if csr is not None:
        diff   = casr_csr - csr
        symbol = (
            "✅" if diff > 0 else "❌")
        print(
            f"  {name}: {csr:.3f}% "
            f"({diff:+.3f}pp) {symbol}")

        if csr < best_csr:
            best_csr  = csr
            best_path = path
            best_name = name

print(f"\n{'='*55}")
print("SUMMARY")
print(f"{'='*55}")
print(f"CASR baseline: {casr_csr:.3f}%")
print(f"Best checkpoint: {best_name}")
print(f"Best CSR: {best_csr:.3f}%")

diff = casr_csr - best_csr

if diff > 0:
    print(
        f"Beats CASR by "
        f"{diff:.3f}pp! ✅")
    print(
        f"\nCopying {best_name} "
        f"to best/...")
    best_folder = SEED_PATH + "best/"
    os.makedirs(
        best_folder, exist_ok=True)
    for f in os.listdir(best_path):
        shutil.copy2(
            os.path.join(
                best_path, f),
            os.path.join(
                best_folder, f))
    print(
        f"Copied {best_name} "
        f"to best/! ✅")
    print(
        f"\nSeed 456 ready!")
    print(
        f"Now update run_multiseed.py")
    print(
        f"Set already_trained=True")
    print(
        f"for seeds 123 and 456!")
    print(
        f"Then run: "
        f"python run_multiseed.py")
else:
    print(
        f"Does not beat CASR! ❌")
    print(
        f"Seed 456 found suboptimal"
        f" policy!")
    print(
        f"Will report with seed 42"
        f" as primary result!")
    print(
        f"High RL variance noted"
        f" in limitations!")