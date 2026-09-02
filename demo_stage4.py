"""
demo_stage4.py
---------------
Run RoundRobin and Random with a REALISTIC imperfect sensor
(p_detect=0.9, p_false_alarm=0.05 - i.e. 90% chance to correctly catch
a signal it's tuned to, 5% chance to falsely "see" one that isn't
there) and print the full metrics report for each.
"""

import random
from schedulers import RoundRobinScheduler, RandomScheduler
from simulate import run_simulation, make_default_emitters
from metrics import compute_metrics, print_report

NUM_BANDS = 8
NUM_STEPS = 60          # a bit longer than before, so rarer events (FP) show up
P_DETECT = 0.9
P_FALSE_ALARM = 0.05


def evaluate(scheduler_factory, label):
    random.seed(42)
    emitters = make_default_emitters(NUM_BANDS)
    scheduler = scheduler_factory()
    env, receiver = run_simulation(
        scheduler, NUM_BANDS, NUM_STEPS, emitters=emitters,
        p_detect=P_DETECT, p_false_alarm=P_FALSE_ALARM, receiver_seed=99,
    )
    m = compute_metrics(env, receiver, NUM_STEPS)
    print_report(label, m)
    return m


evaluate(lambda: RoundRobinScheduler(NUM_BANDS), "RoundRobin (imperfect sensor)")
evaluate(lambda: RandomScheduler(NUM_BANDS, seed=7), "Random (imperfect sensor)")