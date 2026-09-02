"""
demo_stage3.py
---------------
Compare RoundRobinScheduler vs RandomScheduler on the SAME emitter
behavior, so the comparison is fair (neither gets an easier/harder
environment by luck).
"""

import random
from schedulers import RoundRobinScheduler, RandomScheduler
from simulate import run_simulation, make_default_emitters

NUM_BANDS = 8
NUM_STEPS = 30


def evaluate(scheduler_factory, label):
    """
    scheduler_factory: a function that returns a fresh scheduler instance.
    We re-seed random right before building emitters, so every scheduler
    we test faces the EXACT same sequence of emitter behavior - the only
    thing that differs between runs is the scheduler's own choices.
    """
    random.seed(42)                              # reset global random state
    emitters = make_default_emitters(NUM_BANDS)   # same emitters, same seed
    scheduler = scheduler_factory()
    env, receiver = run_simulation(scheduler, NUM_BANDS, NUM_STEPS, emitters=emitters)

    total_events = sum(sum(row) for row in env.truth_log)
    hits = len(receiver.hits())
    ratio = hits / total_events if total_events else 0.0

    print(f"{label:>16}: {hits}/{total_events} caught -> interception ratio = {ratio:.1%}")
    return ratio


print(f"Comparing schedulers over {NUM_STEPS} ticks, {NUM_BANDS} bands:\n")
evaluate(lambda: RoundRobinScheduler(NUM_BANDS), "RoundRobin")
evaluate(lambda: RandomScheduler(NUM_BANDS, seed=7), "Random")

print("\nRunning Random with a few different seeds, to see how much luck matters:")
for seed in [1, 2, 3, 4, 5]:
    evaluate(lambda: RandomScheduler(NUM_BANDS, seed=seed), f"Random(seed={seed})")