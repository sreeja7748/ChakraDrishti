"""
metrics.py
-----------
Computes the "figures of merit" the problem statement asks for, by
comparing the receiver's scan_log (what it REPORTED) against the
environment's truth_log (what ACTUALLY happened).

IMPORTANT: this comparison is done by US, the outside evaluator/grader,
AFTER the simulation has finished. No scheduler or receiver code is
allowed to see this comparison while making decisions - that would be
cheating (using the answer key while taking the exam). This file is
purely for scoring runs after the fact.
"""

import math
from environment import RFEnvironment
from receiver import Receiver


def confusion_counts(env: RFEnvironment, receiver: Receiver):
    """
    Walks every scan the receiver made and buckets it into one of the
    four classic outcomes of a binary detection problem:

      TP (true positive)  = band WAS transmitting, sensor reported HIT
      FN (false negative)  = band WAS transmitting, sensor reported MISS
                              -> a "missed detection"
      FP (false positive)  = band was SILENT, sensor reported HIT
                              -> a "false alarm"
      TN (true negative)   = band was SILENT, sensor reported MISS
                              -> correctly stayed quiet

    This 2x2 table (called a "confusion matrix") is the foundation
    every detection-theory metric (Pd, Pfa, accuracy...) is built from.
    """
    TP = FN = FP = TN = 0
    for r in receiver.scan_log:
        true_state = bool(env.truth_log[r.t][r.band])
        if true_state and r.hit:
            TP += 1
        elif true_state and not r.hit:
            FN += 1
        elif not true_state and r.hit:
            FP += 1
        else:
            TN += 1
    return TP, FN, FP, TN


def compute_metrics(env: RFEnvironment, receiver: Receiver, num_steps: int) -> dict:
    """
    Returns a dict of every figure of merit the problem statement asks
    for, computed from one finished simulation run.
    """
    TP, FN, FP, TN = confusion_counts(env, receiver)

    # --- Probability of Detection (Pd) ---
    # Of all the times we were ACTUALLY looking at a transmitting band,
    # what fraction did the sensor correctly catch?
    # This is the SAME quantity conventionally called "sensitivity" in
    # detection/statistics theory - two names, one number.
    Pd = TP / (TP + FN) if (TP + FN) > 0 else float("nan")
    sensitivity = Pd

    # --- Probability of False Alarm (Pfa) ---
    # Of all the times we were looking at a SILENT band, what fraction
    # did the sensor wrongly claim was a hit?
    Pfa = FP / (FP + TN) if (FP + TN) > 0 else float("nan")

    # --- Interception ratio ---
    # Of ALL transmission events that happened ANYWHERE in the whole
    # environment (most of which we never even looked at, because we
    # can only scan one band per tick), what fraction did we catch?
    # This is fundamentally limited by SCHEDULING, not just sensor
    # quality - even a perfect sensor (Pd=1) can't catch a transmission
    # on a band it never pointed at.
    total_events = sum(sum(row) for row in env.truth_log)
    interception_ratio = TP / total_events if total_events > 0 else 0.0

    # --- Average intercept rate ---
    # How many confirmed intercepts per tick, on average.
    avg_intercept_rate = TP / num_steps if num_steps > 0 else 0.0

    # --- Percentage of correct predictions ---
    # Of every band CHOICE the scheduler made, what fraction turned out
    # to be a genuine hit? This scores the SCHEDULER's judgement, mixing
    # together "did it pick a good band" AND "did the sensor confirm it".
    total_scans = len(receiver.scan_log)
    pct_correct_predictions = TP / total_scans if total_scans > 0 else 0.0

    # --- Average intercept time (and its spread) ---
    # The mean number of ticks between consecutive successful intercepts.
    # Smaller = catching transmissions more frequently/regularly.
    # We also report the standard deviation as a simple proxy for
    # "intercept time error" (how much intercept timing varies, tick to
    # tick) - a tighter spread means more PREDICTABLE intercept timing.
    hit_ticks = [r.t for r in receiver.scan_log if r.hit]
    if len(hit_ticks) >= 2:
        gaps = [hit_ticks[i + 1] - hit_ticks[i] for i in range(len(hit_ticks) - 1)]
        avg_intercept_time = sum(gaps) / len(gaps)
        mean = avg_intercept_time
        variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
        intercept_time_std = math.sqrt(variance)
    else:
        avg_intercept_time = float("nan")
        intercept_time_std = float("nan")

    # --- Average reward (for later RL use, Stage 6) ---
    # Simple reward: +1 for a hit, 0 for a miss. Defining this now, as
    # its own metric, means Stage 6's RL agent will be optimizing
    # exactly the number we're already reporting here - no mismatch
    # between "what we measure" and "what the agent is trained to do".
    avg_reward = TP / total_scans if total_scans > 0 else 0.0

    return {
        "TP": TP, "FN": FN, "FP": FP, "TN": TN,
        "probability_of_detection": Pd,
        "sensitivity": sensitivity,
        "probability_of_false_alarm": Pfa,
        "interception_ratio": interception_ratio,
        "avg_intercept_rate": avg_intercept_rate,
        "pct_correct_predictions": pct_correct_predictions,
        "avg_intercept_time": avg_intercept_time,
        "intercept_time_std": intercept_time_std,
        "avg_reward": avg_reward,
        "total_events": total_events,
        "total_scans": total_scans,
    }


def print_report(label: str, metrics: dict):
    """Pretty-prints a metrics dict in a readable, labeled block."""
    print(f"\n=== {label} ===")
    print(f"  Confusion matrix:  TP={metrics['TP']}  FN={metrics['FN']}  "
          f"FP={metrics['FP']}  TN={metrics['TN']}")
    print(f"  Probability of Detection (Pd) / Sensitivity: {metrics['probability_of_detection']:.1%}")
    print(f"  Probability of False Alarm (Pfa):             {metrics['probability_of_false_alarm']:.1%}")
    print(f"  Interception ratio (of ALL events, any band): {metrics['interception_ratio']:.1%}")
    print(f"  Avg intercept rate (hits/tick):                {metrics['avg_intercept_rate']:.3f}")
    print(f"  Pct correct predictions (hits/scans):          {metrics['pct_correct_predictions']:.1%}")
    print(f"  Avg intercept time (ticks between hits):       {metrics['avg_intercept_time']:.2f}"
          f"  (std={metrics['intercept_time_std']:.2f})")
    print(f"  Avg reward:                                    {metrics['avg_reward']:.3f}")