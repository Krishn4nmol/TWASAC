# run_multiseed.py
# Runs TASCAR training for multiple seeds!
# Seed 42: Already done! Loads existing!
# Seed 123: Trains fresh!
# Seed 456: Trains fresh!
# Reports mean +- std across all seeds!
# Addresses single-seed reviewer concern!

import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

from config import (
    NUM_QUEUES,
    NUM_FUNCTIONS,
    EVAL_CALLS,
    DELTA,
    TASCAR_EVAL_DELTA,
    SEQUENCE_LENGTH,
    TRANSFORMER_DIM,
    TASCAR_EPISODES,
    TASCAR_DELTA,
    SAC_BATCH_SIZE,
    SAC_UPDATES_PER_STEP,
    CONVERGENCE_WINDOW,
    CONVERGENCE_THRESHOLD,
    TRAIN_DAYS,
    THETA,
    THETA_MIN,
    THETA_MAX,
    THETA_ADAPT_RATE,
    SCALING_FACTOR,
    MODEL_SAVE_PATH,
    SLA_THRESHOLD,
    CARBON_INTENSITY)
from simulator import AzureDataLoader
from metrics_tracker import MetricsTracker
from transformer_encoder import (
    TransformerEncoder,
    StateHistoryBuffer)
from sac_agent import SACAgent
from ppo_agent import PPOAgent
from train_tascar import (
    load_filtered_data,
    normalize_state,
    compute_dynamic_theta,
    RewardNormalizer,
    TASCARLogger,
    warmup_buffer)
from scache import SCache


# ─────────────────────────────────────────
# SEED CONFIGURATIONS
# ─────────────────────────────────────────

SEEDS = [42, 123, 456]

SEED_CONFIGS = {
    42: {
        'model_path':
            'trained_model_tascar/',
        'results_path':
            'results_tascar/',
        'already_trained': True,
    },
    123: {
        'model_path':
            'trained_model_tascar_seed123/',
        'results_path':
            'results_tascar_seed123/',
        'already_trained': False,
    },
    456: {
        'model_path':
            'trained_model_tascar_seed456/',
        'results_path':
            'results_tascar_seed456/',
        'already_trained': False,
    },
}

MULTISEED_RESULTS = (
    "results_multiseed/")


# ─────────────────────────────────────────
# TRAIN ONE SEED
# ─────────────────────────────────────────

def train_one_seed(seed,
                   model_path,
                   results_path):
    """
    Trains TASCAR for one seed!
    Same as train_tascar.py but
    with custom paths and seed!
    """
    os.makedirs(
        model_path, exist_ok=True)
    os.makedirs(
        results_path, exist_ok=True)

    np.random.seed(seed)
    print(
        f"\nTraining seed {seed}...")
    print(f"  Model: {model_path}")
    print(
        f"  Results: {results_path}")

    train_data = load_filtered_data()

    state_dim    = NUM_QUEUES * 7
    action_dim   = 3 ** NUM_QUEUES
    steps_per_ep = (
        EVAL_CALLS // TASCAR_DELTA)

    transformer = TransformerEncoder(
        state_dim)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)

    warmup_buffer(
        agent, train_data,
        state_dim,
        warmup_episodes=20)

    reward_norm   = RewardNormalizer()
    logger        = TASCARLogger()
    current_theta = THETA

    logger.start_training()

    print(
        f"  Training "
        f"{TASCAR_EPISODES} episodes...")

    for episode in range(
            1, TASCAR_EPISODES + 1):

        max_start = max(
            1,
            len(train_data) -
            EVAL_CALLS)
        start_idx = np.random.randint(
            0, max_start)
        episode_calls = train_data[
            start_idx:
            start_idx + EVAL_CALLS]
        if len(episode_calls) < 1000:
            episode_calls = (
                train_data[:EVAL_CALLS])

        scache  = SCache()
        history = StateHistoryBuffer(
            SEQUENCE_LENGTH, state_dim)

        raw_state = normalize_state(
            scache.get_state())
        history.add(raw_state)
        seq = history.get_sequence()
        encoded_state = (
            agent.get_encoded_state(seq))

        ep_reward  = 0.0
        ep_cold    = 0
        ep_warm    = 0
        step_cold  = 0
        step_warm  = 0
        call_count = 0
        wmt_before = 0.0
        steps_done = 0

        for call in episode_calls:
            is_warm = (
                scache.handle_request(
                    call))
            if is_warm:
                step_warm += 1
                ep_warm   += 1
            else:
                step_cold += 1
                ep_cold   += 1
            call_count += 1

            if (call_count %
                    TASCAR_DELTA == 0):

                new_raw = normalize_state(
                    scache.get_state())
                history.add(new_raw)
                new_seq = (
                    history.get_sequence())
                next_encoded = (
                    agent.get_encoded_state(
                        new_seq))

                total = (
                    step_cold + step_warm)
                cold_rate = (
                    step_cold / total
                    if total > 0 else 0)
                current_theta = (
                    compute_dynamic_theta(
                        cold_rate,
                        current_theta))

                current_wmt = (
                    scache
                    .get_total_wasted_memory_time())
                wmt_change = max(
                    0,
                    current_wmt -
                    wmt_before)
                wmt_before = current_wmt

                reward = (
                    reward_norm.calculate(
                        step_cold,
                        wmt_change,
                        current_theta))

                action = (
                    agent.choose_action(
                        encoded_state))

                if (not np.isnan(
                        encoded_state
                    ).any() and
                        not np.isnan(
                            next_encoded
                        ).any()):
                    agent.store_experience(
                        encoded_state,
                        action, reward,
                        next_encoded,
                        False)

                ep_reward  += reward
                steps_done += 1

                for _ in range(
                        SAC_UPDATES_PER_STEP):
                    agent.update()

                scales = (
                    agent.action_map[
                        action])
                for q_idx, scale in (
                        enumerate(
                            scales)):
                    if scale != 0:
                        scache.scale_queue(
                            q_idx, scale)

                encoded_state = (
                    next_encoded)
                step_cold = 0
                step_warm = 0

        total_calls = ep_cold + ep_warm
        cold_pct = (
            ep_cold / total_calls * 100
            if total_calls > 0 else 0)
        avg_ep_reward = (
            ep_reward / steps_done
            if steps_done > 0
            else ep_reward)

        logger.log_episode(
            episode,
            avg_ep_reward,
            cold_pct,
            scache
            .get_total_wasted_memory_time(),
            current_theta,
            steps_this_ep=steps_done)

        if (avg_ep_reward >
                logger.best_reward):
            agent.save(
                model_path + "best/")

        if episode % 50 == 0:
            agent.save(
                model_path +
                f"checkpoint_ep"
                f"{episode}/")

        if episode % 10 == 0:
            avg_r = np.mean(
                logger.rewards[-10:])
            avg_c = np.mean(
                logger.cold_start_rates[
                    -10:])
            print(
                f"  Ep {episode:3d} | "
                f"Reward: {avg_r:7.4f} | "
                f"Cold%: {avg_c:5.1f}% | "
                f"Time: "
                f"{logger.get_training_time():.0f}s")

    logger.end_training()
    agent.save(model_path + "best/")
    logger.save_logs(results_path)

    print(
        f"\n  Seed {seed} training done!")
    print(
        f"  Best reward: "
        f"{logger.best_reward:.4f}")
    print(
        f"  Time: "
        f"{logger.get_training_time():.1f}s")

    return agent


# ─────────────────────────────────────────
# EVALUATE ONE SEED
# ─────────────────────────────────────────

def evaluate_one_seed(seed,
                      model_path,
                      results_path,
                      workloads):
    """
    Evaluates TASCAR for one seed!
    Returns metrics for all workloads!
    """
    print(
        f"\nEvaluating seed {seed}...")

    best_path = model_path + "best/"
    if not os.path.exists(
            best_path + "actor.pth"):
        print(
            f"  No model for "
            f"seed {seed}!")
        return None

    state_dim   = NUM_QUEUES * 7
    action_dim  = 3 ** NUM_QUEUES
    transformer = TransformerEncoder(
        state_dim)
    agent = SACAgent(
        transformer_dim=TRANSFORMER_DIM,
        action_dim=action_dim,
        transformer=transformer)
    agent.load(best_path)

    # Also load CASR for comparison
    casr_path  = MODEL_SAVE_PATH + "best/"
    state_dim2 = NUM_QUEUES * 7
    casr_agent = PPOAgent(
        state_dim2, action_dim)

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

    if os.path.exists(
            casr_path + "actor.pth"):
        casr_agent.load(casr_path)

    seed_results = {}

    for wl_name, calls in (
            workloads.items()):
        print(
            f"  Workload: {wl_name}")

        # Run CASR
        casr_tracker = MetricsTracker()
        call_count   = 0
        for call in calls:
            casr_tracker.handle_request(
                call)
            call_count += 1
            if call_count % DELTA == 0:
                state = np.array(
                    casr_tracker
                    .get_state(),
                    dtype=np.float32)
                mean = np.mean(state)
                std  = np.std(state)
                if std > 0:
                    state = (
                        (state - mean) /
                        std)
                act, _ = (
                    casr_agent
                    .choose_action(state))
                for q_idx, scale in (
                        enumerate(
                            action_map[
                                act])):
                    if scale != 0:
                        casr_tracker\
                            .scale_queue(
                            q_idx, scale)
        casr_m = (
            casr_tracker.get_all_metrics())

        # Run TASCAR
        tascar_tracker = MetricsTracker()
        history        = StateHistoryBuffer(
            SEQUENCE_LENGTH, state_dim)
        call_count     = 0
        for call in calls:
            tascar_tracker.handle_request(
                call)
            call_count += 1
            if (call_count %
                    TASCAR_EVAL_DELTA == 0):
                raw = np.array(
                    tascar_tracker
                    .get_state(),
                    dtype=np.float32)
                mean = np.mean(raw)
                std  = np.std(raw)
                if std > 0:
                    raw = (
                        (raw - mean) / std)
                history.add(raw)
                seq = (
                    history.get_sequence())
                enc = (
                    agent.get_encoded_state(
                        seq))
                act = (
                    agent.choose_action(
                        enc,
                        evaluate=True))
                for q_idx, scale in (
                        enumerate(
                            agent.action_map[
                                act])):
                    if scale != 0:
                        tascar_tracker\
                            .scale_queue(
                            q_idx, scale)
        tascar_m = (
            tascar_tracker
            .get_all_metrics())

        casr_csr   = (
            casr_m['cold_start_rate'])
        tascar_csr = (
            tascar_m['cold_start_rate'])
        tascar_m['agi'] = (
            (casr_csr - tascar_csr) /
            casr_csr * 100
            if casr_csr > 0 else 0)
        casr_m['agi'] = 0.0

        seed_results[wl_name] = {
            'CASR':   casr_m,
            'TASCAR': tascar_m,
        }

        print(
            f"    CASR:   "
            f"{casr_csr:.3f}%")
        print(
            f"    TASCAR: "
            f"{tascar_csr:.3f}% "
            f"(AGI: "
            f"{tascar_m['agi']:.2f}%)")

    # Save seed results
    os.makedirs(
        results_path, exist_ok=True)
    serializable = {}
    for wl, algos in (
            seed_results.items()):
        serializable[wl] = {}
        for algo, metrics in (
                algos.items()):
            serializable[wl][algo] = {
                k: float(v)
                for k, v in
                metrics.items()
                if isinstance(
                    v, (int, float))}
    with open(
            results_path +
            'casr_vs_tascar.json',
            'w') as f:
        json.dump(
            serializable, f, indent=2)

    return seed_results


# ─────────────────────────────────────────
# LOAD WORKLOADS
# ─────────────────────────────────────────

def load_workloads():
    loader    = AzureDataLoader()
    workloads = {}

    print("\nPreparing workloads...")

    # Use seed 42 for workload
    # selection - same as main eval!
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
    workloads['Common'] = common
    print(
        f"  Common: {len(common)} calls")

    day2  = loader.load_day(2)
    heavy = [
        c for c in day2
        if c.cold_start_overhead > 1]
    counts = Counter(
        c.function_id for c in heavy)
    top = set(
        f for f, _ in
        counts.most_common(NUM_FUNCTIONS))
    significant = [
        c for c in heavy
        if c.function_id in top]
    np.random.seed(42)
    if len(significant) > EVAL_CALLS:
        idx = np.random.choice(
            len(significant),
            EVAL_CALLS,
            replace=False)
        idx.sort()
        significant = [
            significant[i]
            for i in idx]
    workloads['Significant'] = (
        significant)
    print(
        f"  Significant: "
        f"{len(significant)} calls")

    day3  = loader.load_day(3)
    funcs = list(set(
        c.function_id for c in day3))
    np.random.seed(43)
    np.random.shuffle(funcs)
    selected  = set(funcs[:NUM_FUNCTIONS])
    random_wl = [
        c for c in day3
        if c.function_id in selected]
    np.random.seed(43)
    if len(random_wl) > EVAL_CALLS:
        idx = np.random.choice(
            len(random_wl),
            EVAL_CALLS,
            replace=False)
        idx.sort()
        random_wl = [
            random_wl[i] for i in idx]
    workloads['Random'] = random_wl
    print(
        f"  Random: "
        f"{len(random_wl)} calls")

    return workloads


# ─────────────────────────────────────────
# COMPUTE STATISTICS
# Mean +- std across seeds!
# ─────────────────────────────────────────

def compute_statistics(all_results):
    """
    Computes mean +- std across seeds!
    Key for IEEE paper submission!
    """
    workloads  = [
        'Common', 'Significant', 'Random']
    algorithms = ['CASR', 'TASCAR']
    metrics    = [
        'cold_start_rate',
        'avg_cold_start_overhead',
        'p95_latency',
        'p99_latency',
        'avg_response_time',
        'avg_wasted_memory_time',
        'container_utilization_rate',
        'resource_utilization_efficiency',
        'sla_violation_rate',
        'throughput',
        'energy_per_request',
        'co2_estimate',
        'tpi',
        'agi',
    ]

    stats = {}

    for wl in workloads:
        stats[wl] = {}
        for algo in algorithms:
            stats[wl][algo] = {}
            for metric in metrics:
                values = []
                for seed_r in (
                        all_results
                        .values()):
                    if (wl in seed_r and
                            algo in
                            seed_r[wl]):
                        val = (
                            seed_r[wl]
                            [algo]
                            .get(metric, 0))
                        values.append(val)
                if values:
                    stats[wl][algo][
                        metric] = {
                        'mean': float(
                            np.mean(
                                values)),
                        'std':  float(
                            np.std(values)),
                        'min':  float(
                            np.min(values)),
                        'max':  float(
                            np.max(values)),
                        'values': values,
                    }

    return stats


# ─────────────────────────────────────────
# PRINT STATISTICS TABLE
# ─────────────────────────────────────────

def print_statistics(stats):
    """
    Prints mean +- std table!
    Ready for IEEE paper!
    """
    workloads = [
        'Common', 'Significant', 'Random']

    print("\n" + "=" * 75)
    print("MULTI-SEED RESULTS")
    print(f"Seeds: {SEEDS}")
    print("=" * 75)

    for wl in workloads:
        print(f"\nWorkload: {wl}")
        print("-" * 70)
        print(
            f"{'Metric':<30}"
            f"{'CASR (mean±std)':>20}"
            f"{'TASCAR (mean±std)':>20}")
        print("-" * 70)

        key_metrics = [
            ('cold_start_rate',
             'Cold Start Rate (%)'),
            ('avg_response_time',
             'Avg Response Time (s)'),
            ('p95_latency',
             'P95 Latency (s)'),
            ('sla_violation_rate',
             'SLA Violation Rate (%)'),
            ('container_utilization_rate',
             'Container Util (%)'),
            ('co2_estimate',
             'CO2 Estimate (kg)'),
            ('tpi',
             'TPI Score'),
            ('agi',
             'AGI (%)'),
        ]

        for metric, label in key_metrics:
            casr_s = (
                stats[wl]['CASR']
                .get(metric, {}))
            tascar_s = (
                stats[wl]['TASCAR']
                .get(metric, {}))

            if not casr_s or not tascar_s:
                continue

            casr_str = (
                f"{casr_s['mean']:.2f}"
                f"±{casr_s['std']:.2f}")
            tascar_str = (
                f"{tascar_s['mean']:.2f}"
                f"±{tascar_s['std']:.2f}")

            print(
                f"{label:<30}"
                f"{casr_str:>20}"
                f"{tascar_str:>20}")

    # Improvement summary
    print("\n" + "=" * 75)
    print("IMPROVEMENT SUMMARY")
    print("(mean ± std across seeds)")
    print("=" * 75)

    for wl in workloads:
        casr_csr = (
            stats[wl]['CASR']
            .get('cold_start_rate', {}))
        tascar_csr = (
            stats[wl]['TASCAR']
            .get('cold_start_rate', {}))

        if casr_csr and tascar_csr:
            diff_mean = (
                casr_csr['mean'] -
                tascar_csr['mean'])
            diff_std  = np.sqrt(
                casr_csr['std']**2 +
                tascar_csr['std']**2)
            print(
                f"  {wl:<15}: "
                f"CASR={casr_csr['mean']:.2f}±"
                f"{casr_csr['std']:.2f}% "
                f"TASCAR={tascar_csr['mean']:.2f}±"
                f"{tascar_csr['std']:.2f}% "
                f"Diff={diff_mean:.2f}±"
                f"{diff_std:.2f}pp")


# ─────────────────────────────────────────
# PLOT MULTI-SEED RESULTS
# ─────────────────────────────────────────

def plot_multiseed(stats, all_results):
    """
    Plots mean ± std bar charts!
    Shows variance across seeds!
    """
    os.makedirs(
        MULTISEED_RESULTS, exist_ok=True)

    workloads  = [
        'Common', 'Significant', 'Random']
    algorithms = ['CASR', 'TASCAR']
    colors     = {
        'CASR':   '#2196F3',
        'TASCAR': '#FF5722'}

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 7))
    fig.suptitle(
        f'TASCAR vs CASR: '
        f'Multi-Seed Validation\n'
        f'Seeds: {SEEDS} | '
        f'Cold Start Rate (%) | '
        f'Error bars = std',
        fontsize=13,
        fontweight='bold')

    for ax_idx, wl in enumerate(
            workloads):
        ax     = axes[ax_idx]
        x      = np.arange(
            len(algorithms))
        means  = []
        stds   = []
        colors_list = []

        for algo in algorithms:
            s = (stats[wl][algo]
                 .get('cold_start_rate',
                      {}))
            means.append(
                s.get('mean', 0))
            stds.append(
                s.get('std', 0))
            colors_list.append(
                colors[algo])

        bars = ax.bar(
            x, means,
            color=colors_list,
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5)
        ax.errorbar(
            x, means, yerr=stds,
            fmt='none',
            color='black',
            capsize=8,
            linewidth=2,
            capthick=2)

        for bar, mean, std in zip(
                bars, means, stds):
            ax.text(
                bar.get_x() +
                bar.get_width()/2,
                bar.get_height() +
                std + 0.5,
                f'{mean:.1f}%\n'
                f'±{std:.1f}',
                ha='center',
                fontsize=9,
                fontweight='bold')

        ax.set_title(
            f'{wl} Workload',
            fontweight='bold',
            fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel(
            'Cold Start Rate (%)')

    plt.tight_layout()
    path = (
        MULTISEED_RESULTS +
        'multiseed_comparison.png')
    plt.savefig(
        path, dpi=150,
        bbox_inches='tight')
    plt.close()
    print(f"\nGraph saved: {path}")

    # Per-seed plot
    fig2, axes2 = plt.subplots(
        1, 3, figsize=(18, 7))
    fig2.suptitle(
        f'Per-Seed Cold Start Rate\n'
        f'TASCAR across Seeds '
        f'{SEEDS}',
        fontsize=13,
        fontweight='bold')

    seed_colors = [
        '#FF5722', '#9C27B0', '#009688']

    for ax_idx, wl in enumerate(
            workloads):
        ax = axes2[ax_idx]
        seed_vals = []
        seed_labels = []

        for s_idx, (seed, seed_r) in (
                enumerate(
                    all_results.items())):
            if (wl in seed_r and
                    'TASCAR' in
                    seed_r[wl]):
                val = (
                    seed_r[wl]['TASCAR']
                    ['cold_start_rate'])
                seed_vals.append(val)
                seed_labels.append(
                    f'Seed\n{seed}')

        if seed_vals:
            bars = ax.bar(
                range(len(seed_vals)),
                seed_vals,
                color=seed_colors[
                    :len(seed_vals)],
                alpha=0.85,
                edgecolor='black')
            for bar, val in zip(
                    bars, seed_vals):
                ax.text(
                    bar.get_x() +
                    bar.get_width()/2,
                    bar.get_height() +
                    0.2,
                    f'{val:.2f}%',
                    ha='center',
                    fontsize=9,
                    fontweight='bold')

            mean_val = np.mean(seed_vals)
            ax.axhline(
                y=mean_val,
                color='red',
                linestyle='--',
                linewidth=2,
                label=f'Mean: '
                      f'{mean_val:.2f}%')
            ax.legend(fontsize=9)

        ax.set_title(
            f'{wl} Workload',
            fontweight='bold',
            fontsize=11)
        ax.set_xticks(
            range(len(seed_labels)))
        ax.set_xticklabels(seed_labels)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylabel(
            'Cold Start Rate (%)')

    plt.tight_layout()
    path2 = (
        MULTISEED_RESULTS +
        'per_seed_comparison.png')
    plt.savefig(
        path2, dpi=150,
        bbox_inches='tight')
    plt.close()
    print(f"Graph saved: {path2}")


# ─────────────────────────────────────────
# SAVE COMBINED RESULTS
# ─────────────────────────────────────────

def save_combined_results(
        stats, all_results):
    """
    Saves combined multi-seed results!
    Used for paper tables!
    """
    os.makedirs(
        MULTISEED_RESULTS, exist_ok=True)

    combined = {
        'seeds': SEEDS,
        'statistics': {},
        'per_seed': {},
    }

    # Statistics
    for wl, algos in stats.items():
        combined['statistics'][wl] = {}
        for algo, metrics in (
                algos.items()):
            combined['statistics'][
                wl][algo] = {
                k: {
                    'mean': v['mean'],
                    'std':  v['std'],
                    'min':  v['min'],
                    'max':  v['max'],
                }
                for k, v in
                metrics.items()
            }

    # Per seed
    for seed, seed_r in (
            all_results.items()):
        combined['per_seed'][
            str(seed)] = {}
        for wl, algos in seed_r.items():
            combined['per_seed'][
                str(seed)][wl] = {
                algo: {
                    k: float(v)
                    for k, v in
                    m.items()
                    if isinstance(
                        v, (int, float))}
                for algo, m in
                algos.items()
            }

    path = (
        MULTISEED_RESULTS +
        'multiseed_results.json')
    with open(path, 'w') as f:
        json.dump(
            combined, f, indent=2)
    print(f"\nCombined results: {path}")

    return combined


# ─────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────

def run_multiseed():
    """
    Main function!
    Trains and evaluates all seeds!
    Reports mean +- std!
    """
    os.makedirs(
        MULTISEED_RESULTS,
        exist_ok=True)

    print("=" * 60)
    print("TASCAR Multi-Seed Validation")
    print(f"Seeds: {SEEDS}")
    print(
        "Addresses single-seed "
        "reviewer concern!")
    print("=" * 60)

    # Load workloads once
    # Use fixed seed 42 for
    # workload selection!
    workloads = load_workloads()

    all_results = {}

    for seed in SEEDS:
        config    = SEED_CONFIGS[seed]
        model_path   = config['model_path']
        results_path = config['results_path']
        already_done = config[
            'already_trained']

        print(
            f"\n{'='*50}")
        print(
            f"Seed: {seed}")
        print(
            f"{'='*50}")

        # Train if needed
        if already_done:
            print(
                f"Seed {seed} already "
                f"trained! Loading...")
            existing = (
                results_path +
                'casr_vs_tascar.json')
            if os.path.exists(existing):
                print(
                    f"  Loading existing "
                    f"results!")
                with open(existing) as f:
                    saved = json.load(f)
                seed_results = {}
                for wl in saved:
                    if wl == 'rl_metrics':
                        continue
                    seed_results[wl] = {
                        'CASR': saved[
                            wl].get(
                            'CASR', {}),
                        'TASCAR': saved[
                            wl].get(
                            'TASCAR', {}),
                    }
                all_results[seed] = (
                    seed_results)
                print(
                    f"  Loaded! "
                    f"Workloads: "
                    f"{list(seed_results.keys())}")
                continue

        # Train new seed
        train_one_seed(
            seed, model_path,
            results_path)

        # Evaluate
        seed_results = evaluate_one_seed(
            seed,
            model_path,
            results_path,
            workloads)

        if seed_results:
            all_results[seed] = (
                seed_results)

        time.sleep(30)

    # Compute statistics
    print(
        "\n" + "=" * 60)
    print("Computing statistics...")
    stats = compute_statistics(
        all_results)

    # Print results
    print_statistics(stats)

    # Save combined results
    save_combined_results(
        stats, all_results)

    # Plot graphs
    plot_multiseed(stats, all_results)

    print(
        "\n" + "=" * 60)
    print("Multi-Seed Complete!")
    print(
        f"Results: {MULTISEED_RESULTS}")
    print("=" * 60)

    return stats, all_results


if __name__ == "__main__":
    print("=" * 60)
    print("TASCAR Multi-Seed Validation")
    print("=" * 60)
    print(
        "Seed 42: Load existing results!")
    print(
        "Seed 123: Train fresh! (~75 min)")
    print(
        "Seed 456: Train fresh! (~75 min)")
    print(
        "Total time: ~2.5 hours!")
    print("=" * 60)
    run_multiseed()