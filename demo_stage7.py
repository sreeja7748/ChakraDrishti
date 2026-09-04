"""
demo_stage7.py
---------------
Evaluate PeriodicLockScheduler (unknown-period discovery) against every
scheduler so far, on the same held-out-seed methodology as Stage 6. Also
prints what periods/phases it actually discovered on one example run,
so we can check its "guess" against radar-1's REAL configuration
(scan_bands=[0,3,5,7], dwell=2 -> true period 8) without ever telling
the algorithm that answer.
"""

import random
from schedulers import (RoundRobinScheduler, RandomScheduler, EpsilonGreedyScheduler,
                         UCBScheduler, PeriodicLockScheduler)
from simulate import run_simulation, make_default_emitters
from metrics import compute_metrics
from train_qlearning import train_qlearning

NUM_BANDS = 8
NUM_STEPS = 60
P_DETECT = 0.9
P_FALSE_ALARM = 0.05
NUM_TEST_SEEDS = 30
TEST_SEED_OFFSET = 10_000


def evaluate_across_seeds(scheduler_factory, label, num_seeds=NUM_TEST_SEEDS, seed_offset=TEST_SEED_OFFSET):
    interception_ratios, avg_rewards = [], []
    for i in range(num_seeds):
        seed = seed_offset + i
        random.seed(seed)
        emitters = make_default_emitters(NUM_BANDS)
        scheduler = scheduler_factory(seed)
        env, receiver = run_simulation(
            scheduler, NUM_BANDS, NUM_STEPS, emitters=emitters,
            p_detect=P_DETECT, p_false_alarm=P_FALSE_ALARM, receiver_seed=seed,
        )
        m = compute_metrics(env, receiver, NUM_STEPS)
        interception_ratios.append(m["interception_ratio"])
        avg_rewards.append(m["avg_reward"])

    mean_ir = sum(interception_ratios) / len(interception_ratios)
    mean_reward = sum(avg_rewards) / len(avg_rewards)
    worst, best = min(interception_ratios), max(interception_ratios)
    print(f"{label:>16}: mean interception ratio = {mean_ir:.1%}  "
          f"(range {worst:.1%} - {best:.1%})   mean reward = {mean_reward:.3f}")


print("Training Q-learning agent (same as Stage 6, for comparison)...")
trained_agent = train_qlearning(
    NUM_BANDS, NUM_STEPS, num_episodes=500,
    p_detect=P_DETECT, p_false_alarm=P_FALSE_ALARM,
    period=NUM_BANDS, training_epsilon=0.15, alpha=0.1, gamma=0.5, seed_offset=0,
)
print("Training done.\n")

print(f"Evaluating on {NUM_TEST_SEEDS} held-out seeds:\n")
evaluate_across_seeds(lambda seed: RoundRobinScheduler(NUM_BANDS), "RoundRobin")
evaluate_across_seeds(lambda seed: RandomScheduler(NUM_BANDS, seed=seed + 1000), "Random")
evaluate_across_seeds(lambda seed: EpsilonGreedyScheduler(NUM_BANDS, epsilon=0.1, seed=seed + 1000), "EpsilonGreedy")
evaluate_across_seeds(lambda seed: UCBScheduler(NUM_BANDS, c=2.0, seed=seed + 1000), "UCB")
evaluate_across_seeds(lambda seed: trained_agent, "QLearning")
evaluate_across_seeds(lambda seed: PeriodicLockScheduler(NUM_BANDS, seed=seed + 1000), "PeriodicLock")

# --- Inspect what PeriodicLock actually discovered, on one concrete run ---
print("\nWhat did PeriodicLock actually learn, on one example run (seed 10000)?")
random.seed(10_000)
emitters = make_default_emitters(NUM_BANDS)
scheduler = PeriodicLockScheduler(NUM_BANDS, seed=1)
env, receiver = run_simulation(
    scheduler, NUM_BANDS, NUM_STEPS, emitters=emitters,
    p_detect=P_DETECT, p_false_alarm=P_FALSE_ALARM, receiver_seed=10_000,
)
print(f"  (probed for first {scheduler.probe_ticks} ticks, then estimated once)")
print(f"  Discovered (band -> (period, phase)): {scheduler.locked_bands}")
print("  Real radar-1 config: scan_bands=[0,3,5,7], dwell=2 -> true period=8, "
      "phases roughly {0:0/1, 3:2/3, 5:4/5, 7:6/7} depending on which sub-tick got sampled")