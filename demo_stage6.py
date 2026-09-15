"""
demo_stage6.py
---------------
Train QLearningScheduler on seeds 0-499 ("training set"), then evaluate
EVERY scheduler so far on seeds 10000-10029 ("held-out test set" - the
agent has never seen these exact random draws during training). This
confirms any advantage is genuine, learned skill, not memorization.
"""

import random
from schedulers import RoundRobinScheduler, RandomScheduler, EpsilonGreedyScheduler, UCBScheduler
from simulate import run_simulation, make_default_emitters
from metrics import compute_metrics
from train_qlearning import train_qlearning

NUM_BANDS = 8
NUM_STEPS = 60
P_DETECT = 0.9
P_FALSE_ALARM = 0.05
NUM_TEST_SEEDS = 30
TEST_SEED_OFFSET = 10_000  # far away from training seeds (0..N), guarantees no overlap


def evaluate_across_seeds(scheduler_factory, label, num_seeds=NUM_TEST_SEEDS, seed_offset=TEST_SEED_OFFSET):
    interception_ratios = []
    avg_rewards = []
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


print("Training Q-learning agent on 500 practice episodes (seeds 0-499)...")
trained_agent = train_qlearning(
    NUM_BANDS, NUM_STEPS, num_episodes=500,
    p_detect=P_DETECT, p_false_alarm=P_FALSE_ALARM,
    period=NUM_BANDS, training_epsilon=0.15, alpha=0.1, gamma=0.5,
    seed_offset=0,
)
print("Training done.\n")

print(f"Evaluating on {NUM_TEST_SEEDS} HELD-OUT seeds (never used in training):\n")
evaluate_across_seeds(lambda seed: RoundRobinScheduler(NUM_BANDS), "RoundRobin")
evaluate_across_seeds(lambda seed: RandomScheduler(NUM_BANDS, seed=seed + 1000), "Random")
evaluate_across_seeds(lambda seed: EpsilonGreedyScheduler(NUM_BANDS, epsilon=0.1, seed=seed + 1000), "EpsilonGreedy")
evaluate_across_seeds(lambda seed: UCBScheduler(NUM_BANDS, c=2.0, seed=seed + 1000), "UCB")
evaluate_across_seeds(lambda seed: trained_agent, "QLearning")

print("\nLearned observation-aware Q-table:")
print("state = (phase, last_hit_band, hit_age)")
print()

for state, q_values in sorted(trained_agent.q_table.items()):
    best_band = max(
        range(NUM_BANDS),
        key=lambda band: q_values[band]
    )

    print(
        f"{state} -> "
        f"best_band={best_band} "
        f"Q={q_values[best_band]:.3f}"
    )