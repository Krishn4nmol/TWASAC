# eval_n5.py
# Extends multi-seed evaluation to n=5 seeds.
# Only evaluates V1 (CASR) and V4 (TASCAR) for
# seeds 789 and 1000 — all other results pre-filled
# from the completed n=3 run.
#
# After this runs, updates Table VI (primary
# multiseed V1 vs V4) and Table VIII (ablation
# multiseed V1/V4 rows) to n=5.

import json
import os
import numpy as np
import torch
from collections import Counter

from config import (
    NUM_QUEUES,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    SCALING_FACTOR,
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    ABLATION_RESULTS)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from ppo_agent import PPOAgent
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent

WORKLOADS  = ['Common', 'Significant', 'Random']
STATE_DIM  = NUM_QUEUES * 7   # 21
ACTION_DIM = 3 ** NUM_QUEUES  # 27

NEW_SEEDS = [789, 1000]

# ─────────────────────────────────────────
# CHECKPOINT PATHS — NEW SEEDS ONLY
# ─────────────────────────────────────────

V1_PATHS = {
    789:  "trained_model_casr200_seed789/best/",
    1000: "trained_model_casr200_seed1000/best/",
}

V4_PATHS = {
    789:  "trained_model_tascar_seed789/best/",
    1000: "trained_model_tascar_seed1000/best/",
}

# ─────────────────────────────────────────
# PRE-FILLED RESULTS FROM n=3 RUN
# (V1 and V4, seeds 42/123/456)
# ─────────────────────────────────────────

V1_RESULTS = {
    42:  {'Common': 88.796, 'Significant': 95.626, 'Random': 89.199},
    123: {'Common': 90.989, 'Significant': 83.726, 'Random': 89.286},
    456: {'Common': 89.146, 'Significant': 95.238, 'Random': 90.169},
}

V4_RESULTS = {
    42:  {'Common': 72.146, 'Significant': 75.521, 'Random': 68.950},
    123: {'Common': 72.111, 'Significant': 75.521, 'Random': 68.940},
    456: {'Common': 72.114, 'Significant': 75.525, 'Random': 68.940},
}

# Also the quick-check CSR for V1 new seeds
# (from training output) -- just for cross-reference
V1_QUICK = {789: 91.681, 1000: 90.464}


# ─────────────────────────────────────────
# ACTION MAP
# ─────────────────────────────────────────

def build_action_map():
    action_map = {}
    choices = [-SCALING_FACTOR, 0, SCALING_FACTOR]
    for i in range(ACTION_DIM):
        action = []
        temp = i
        for _ in range(NUM_QUEUES):
            action.append(choices[temp % 3])
            temp //= 3
        action_map[i] = action
    return action_map

ACTION_MAP = build_action_map()


# ─────────────────────────────────────────
# NORMALIZE STATE
# ─────────────────────────────────────────

def normalize_state(raw_state):
    state = np.array(raw_state, dtype=np.float32)
    if np.isnan(state).any():
        return np.zeros_like(state)
    mean = np.mean(state)
    std  = np.std(state)
    if std > 0:
        state = (state - mean) / std
    if np.isnan(state).any():
        return np.zeros(len(raw_state), dtype=np.float32)
    return state


# ─────────────────────────────────────────
# LOAD WORKLOADS (exact copy from evaluate_tascar.py)
# ─────────────────────────────────────────

def load_workloads():
    loader = AzureDataLoader()
    workloads = {}

    print("  Loading Common...")
    day1 = loader.load_day(1)
    counts = Counter(c.function_id for c in day1)
    top = set(f for f, _ in counts.most_common(NUM_FUNCTIONS))
    common = [c for c in day1 if c.function_id in top]
    np.random.seed(42)
    if len(common) > EVAL_CALLS:
        idx = np.random.choice(len(common), EVAL_CALLS, replace=False)
        idx.sort()
        common = [common[i] for i in idx]
    workloads['Common'] = common
    print(f"    {len(common)} calls")

    print("  Loading Significant...")
    day2 = loader.load_day(2)
    heavy = [c for c in day2 if c.cold_start_overhead > 1]
    significant = [c for c in heavy if c.function_id in top]
    np.random.seed(42)
    if len(significant) > EVAL_CALLS:
        idx = np.random.choice(len(significant), EVAL_CALLS, replace=False)
        idx.sort()
        significant = [significant[i] for i in idx]
    workloads['Significant'] = significant
    print(f"    {len(significant)} calls")

    print("  Loading Random...")
    day3 = loader.load_day(3)
    funcs = list(set(c.function_id for c in day3))
    np.random.seed(123)
    np.random.shuffle(funcs)
    selected = set(funcs[:NUM_FUNCTIONS])
    random_wl = [c for c in day3 if c.function_id in selected]
    np.random.seed(123)
    if len(random_wl) > EVAL_CALLS:
        idx = np.random.choice(len(random_wl), EVAL_CALLS, replace=False)
        idx.sort()
        random_wl = [random_wl[i] for i in idx]
    workloads['Random'] = random_wl
    print(f"    {len(random_wl)} calls")

    return workloads


# ─────────────────────────────────────────
# EVALUATE V1 (CASR/PPO)
# ─────────────────────────────────────────

def eval_v1(seed, workload_calls):
    path = V1_PATHS[seed]
    if not os.path.exists(path):
        print(f"    [SKIP] V1 seed={seed}: not found at {path}")
        return None
    agent = PPOAgent(STATE_DIM, ACTION_DIM)
    agent.load(path)
    tracker = MetricsTracker()
    call_count = 0
    for call in workload_calls:
        tracker.handle_request(call)
        call_count += 1
        if call_count % DELTA == 0:
            raw = np.array(tracker.get_state(), dtype=np.float32)
            norm = normalize_state(raw)
            action, _ = agent.choose_action(norm)
            for q_idx, scale in enumerate(ACTION_MAP[action]):
                if scale != 0:
                    tracker.scale_queue(q_idx, scale)
    return tracker.get_all_metrics()


# ─────────────────────────────────────────
# EVALUATE V4 (Full TASCAR)
# ─────────────────────────────────────────

def eval_v4(seed, workload_calls):
    path = V4_PATHS[seed]
    if not os.path.exists(path):
        print(f"    [SKIP] V4 seed={seed}: not found at {path}")
        return None
    transformer = TransformerEncoder(STATE_DIM)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=ACTION_DIM,
        transformer=transformer)
    agent.load(path)
    history = StateHistoryBuffer(SEQUENCE_LENGTH, STATE_DIM)
    tracker = MetricsTracker()
    call_count = 0
    for call in workload_calls:
        tracker.handle_request(call)
        call_count += 1
        if call_count % DELTA == 0:
            raw = np.array(tracker.get_state(), dtype=np.float32)
            norm = normalize_state(raw)
            history.add(norm)
            seq = history.get_sequence()
            seq_tensor = torch.FloatTensor(seq)
            with torch.no_grad():
                encoded = agent.transformer(seq_tensor)
            action = agent.choose_action(encoded.numpy(), evaluate=True)
            for q_idx, scale in enumerate(ACTION_MAP[action]):
                if scale != 0:
                    tracker.scale_queue(q_idx, scale)
    return tracker.get_all_metrics()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 60)
    print("n=5 Evaluation: V1 + V4, seeds 789 and 1000")
    print("=" * 60)

    print("\nLoading workloads...")
    workloads = load_workloads()

    new_v1 = {}
    new_v4 = {}

    for seed in NEW_SEEDS:
        print(f"\n--- Seed {seed} ---")
        new_v1[seed] = {}
        new_v4[seed] = {}

        for wl_name, wl_calls in workloads.items():
            print(f"  V1 {wl_name}...")
            m = eval_v1(seed, wl_calls)
            if m:
                csr = m['cold_start_rate']
                new_v1[seed][wl_name] = csr
                print(f"    CSR: {csr:.3f}%")

        for wl_name, wl_calls in workloads.items():
            print(f"  V4 {wl_name}...")
            m = eval_v4(seed, wl_calls)
            if m:
                csr = m['cold_start_rate']
                new_v4[seed][wl_name] = csr
                print(f"    CSR: {csr:.3f}%")

    # ─────────────────────────────────────────
    # COMBINE WITH n=3 RESULTS → n=5 SUMMARY
    # ─────────────────────────────────────────

    all_seeds = [42, 123, 456, 789, 1000]

    combined_v1 = dict(V1_RESULTS)
    combined_v4 = dict(V4_RESULTS)
    combined_v1.update(new_v1)
    combined_v4.update(new_v4)

    print("\n" + "=" * 60)
    print("n=5 V1 vs V4: PER-SEED RESULTS")
    print("=" * 60)

    diffs = {wl: [] for wl in WORKLOADS}

    for seed in all_seeds:
        print(f"\nSeed {seed}:")
        for wl in WORKLOADS:
            v1 = combined_v1.get(seed, {}).get(wl)
            v4 = combined_v4.get(seed, {}).get(wl)
            if v1 is None or v4 is None:
                print(f"  {wl}: [missing]")
                continue
            diff = v1 - v4
            diffs[wl].append(diff)
            print(f"  {wl:12s} V1={v1:.3f}  V4={v4:.3f}  diff={diff:+.3f}")

    print("\n" + "=" * 60)
    print("n=5 SUMMARY: V1-V4 IMPROVEMENT (pp)")
    print("=" * 60)

    summary = {}
    for wl in WORKLOADS:
        arr = np.array(diffs[wl])
        mean = float(np.mean(arr))
        std  = float(np.std(arr, ddof=1))
        mn   = float(np.min(arr))
        mx   = float(np.max(arr))
        summary[wl] = {
            'mean': mean, 'std': std,
            'min': mn, 'max': mx,
            'values': arr.tolist()
        }
        print(f"\n{wl}:")
        print(f"  Mean ± std: {mean:.3f} ± {std:.3f} pp")
        print(f"  Range:      [{mn:.3f}, {mx:.3f}] pp")
        print(f"  All diffs:  {[round(x,3) for x in arr.tolist()]}")

    # ─────────────────────────────────────────
    # PAIRED T-TEST + SIGN TEST at n=5
    # ─────────────────────────────────────────

    from scipy import stats
    from math import comb

    print("\n" + "=" * 60)
    print("STATS: PAIRED T-TEST + SIGN TEST (n=5)")
    print("=" * 60)

    for wl in WORKLOADS:
        arr = np.array(diffs[wl])
        t, p = stats.ttest_1samp(arr, 0)
        n = len(arr)
        k = int(np.sum(arr > 0))
        p_sign = sum(comb(n,i) * 0.5**n for i in range(k, n+1)) * 2
        p_sign = min(p_sign, 1.0)
        ci_lo, ci_hi = stats.t.interval(
            0.95, df=n-1,
            loc=np.mean(arr),
            scale=stats.sem(arr))
        print(f"\n{wl}:")
        print(f"  Mean diff: {np.mean(arr):.3f} pp")
        print(f"  95% CI:    [{ci_lo:.3f}, {ci_hi:.3f}]")
        print(f"  t-test:    t={t:.3f}, p={p:.4f}")
        print(f"  Sign test: {k}/{n} seeds favor TASCAR, p={p_sign:.4f}")

    # ─────────────────────────────────────────
    # SAVE
    # ─────────────────────────────────────────

    os.makedirs(ABLATION_RESULTS, exist_ok=True)
    save_path = os.path.join(ABLATION_RESULTS, "n5_results.json")
    with open(save_path, 'w') as f:
        json.dump({
            'v1': {str(k): v for k, v in combined_v1.items()},
            'v4': {str(k): v for k, v in combined_v4.items()},
            'summary': summary,
        }, f, indent=2)
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()