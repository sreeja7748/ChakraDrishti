"""
demo_stage2.py
---------------
Stage 2 demo: same RF environment as before, but now a Receiver actually
scans it tick-by-tick, ONE band at a time, using the simplest possible
strategy (round robin: 0,1,2,...,7,0,1,2,...). We compare what it managed
to catch against the full truth (which WE can see, but the receiver can't).
"""

import random
from environment import RFEnvironment
from emitters import FixedFrequencyEmitter, FrequencyAgileEmitter, PeriodicScanEmitter
from receiver import Receiver

random.seed(42)

NUM_BANDS = 8
NUM_STEPS = 30

emitters = [
    FixedFrequencyEmitter("radio-1", NUM_BANDS, band=2, p_burst_start=0.15, burst_len=3),
    FrequencyAgileEmitter("jammer-1", NUM_BANDS, p_transmit=0.25, seed=1),
    PeriodicScanEmitter("radar-1", NUM_BANDS, scan_bands=[0, 3, 5, 7], dwell=2),
]

env = RFEnvironment(NUM_BANDS, emitters)
receiver = Receiver(NUM_BANDS)

# --- the actual tick-by-tick loop ---
# Notice: we call env.step(t) ONE tick at a time now (not env.run(30) all
# at once like Stage 1), because the receiver needs to react tick-by-tick,
# not see the whole future in advance.
for t in range(NUM_STEPS):
    truth_row = env.step(t)          # the world advances by one tick
    band_to_check = t % NUM_BANDS    # naive round-robin: 0,1,2,...,7,0,1,...
    receiver.scan(t, band_to_check, truth_row)

# --- show what the receiver "experienced" (its own log, not the full grid) ---
print("Receiver's own scan log (this is ALL it knows):")
for r in receiver.scan_log:
    marker = " <-- HIT" if r.hit else ""
    print(f"  {r}{marker}")

print(f"\nTotal scans: {len(receiver.scan_log)}")
print(f"Hits: {len(receiver.hits())}   Misses: {len(receiver.misses())}")

# --- compare against the full truth (WE can see this, receiver can't) ---
total_transmissions = sum(sum(row) for row in env.truth_log)
print(f"\n(For comparison) total transmission events that actually happened: {total_transmissions}")
print(f"Receiver caught {len(receiver.hits())} of those {total_transmissions} -> "
      f"interception ratio = {len(receiver.hits()) / total_transmissions:.1%}")

print("\nHits per band (receiver's own memory):")
for band, count in receiver.hit_count_per_band().items():
    print(f"  band {band}: {count} hits")