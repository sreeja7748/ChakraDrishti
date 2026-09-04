"""
demo_stage5.py
---------------
Compares all four schedulers so far (RoundRobin, Random, EpsilonGreedy,
UCB) - but this time averaged across MANY different random seeds, not
just one. A single seed can flatter or punish a scheduler by luck (as
we saw in Stage 3); averaging over many seeds tells us which scheduler
is actually better on average, with the luck averaged out.
"""

import random
from schedulers import RoundRobinScheduler, RandomScheduler, EpsilonGreedyScheduler, UCBScheduler
from simulate import run_simulation, make_default_emitters
from metrics import compute_metrics

NUM_BANDS = 8
NUM_STEPS = 60
P_DETECT = 0.9
P_FALSE_ALARM = 0.05
NUM_SEEDS = 30  # how many independent random worlds to average over


def evaluate_across_seeds(scheduler_factory, label, num_seeds=NUM_SEEDS):
    interception_ratios = []
    avg_rewards = []
    for seed in range(num_seeds):
        random.seed(seed)                              # a fresh, different world each time
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
    # simple spread measure so we can see how CONSISTENT each scheduler is
    worst = min(interception_ratios)
    best = max(interception_ratios)

    print(f"{label:>16}: mean interception ratio = {mean_ir:.1%}  "
          f"(range {worst:.1%} - {best:.1%})   mean reward = {mean_reward:.3f}")
    return mean_ir


print(f"Averaging over {NUM_SEEDS} independent random worlds, {NUM_STEPS} ticks each:\n")

evaluate_across_seeds(lambda seed: RoundRobinScheduler(NUM_BANDS), "RoundRobin")
evaluate_across_seeds(lambda seed: RandomScheduler(NUM_BANDS, seed=seed + 1000), "Random")
evaluate_across_seeds(lambda seed: EpsilonGreedyScheduler(NUM_BANDS, epsilon=0.1, seed=seed + 1000), "EpsilonGreedy")
evaluate_across_seeds(lambda seed: UCBScheduler(NUM_BANDS, c=2.0, seed=seed + 1000), "UCB")