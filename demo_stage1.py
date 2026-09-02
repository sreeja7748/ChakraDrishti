"""
demo_stage1.py
---------------
Stage 1 demo: build a small RF environment with 3 different emitter types,
run it for 30 timesteps, and print the ground-truth grid so you can SEE
what the receiver is up against.
"""

import random
from environment import RFEnvironment
from emitters import FixedFrequencyEmitter, FrequencyAgileEmitter, PeriodicScanEmitter

random.seed(42)

NUM_BANDS = 8
NUM_STEPS = 30

emitters = [
    FixedFrequencyEmitter("radio-1", NUM_BANDS, band=2, p_burst_start=0.15, burst_len=3),
    FrequencyAgileEmitter("jammer-1", NUM_BANDS, p_transmit=0.25, seed=1),
    PeriodicScanEmitter("radar-1", NUM_BANDS, scan_bands=[0, 3, 5, 7], dwell=2),
]

env = RFEnvironment(NUM_BANDS, emitters)
truth = env.run(NUM_STEPS)

# Print as a band x time grid, easier to read than time x band
print(f"{'band':>4} | " + "".join(f"{t%10}" for t in range(NUM_STEPS)))
print("-" * (7 + NUM_STEPS))
for b in range(NUM_BANDS):
    row = "".join("#" if truth[t][b] else "." for t in range(NUM_STEPS))
    print(f"{b:>4} | {row}")

total_transmissions = sum(sum(row) for row in truth)
busy_slots = sum(1 for row in truth if sum(row) > 0)
print(f"\nTotal (band,time) transmission events: {total_transmissions}")
print(f"Timesteps with at least one active transmission: {busy_slots}/{NUM_STEPS}")