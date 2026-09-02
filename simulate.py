"""
simulate.py
------------
Reusable helper: build a fresh environment + receiver, run a given
scheduler against it for N steps, and hand back everything needed to
score it afterwards. Every stage from here on reuses this exact function
instead of copy-pasting the tick loop each time.
"""

from environment import RFEnvironment
from receiver import Receiver
from schedulers import Scheduler


def make_default_emitters(num_bands: int):
    """
    Builds the same 3-emitter scenario we've used since Stage 1, as a
    function so every demo/stage can get an IDENTICAL environment setup
    without copy-pasting the emitter list everywhere.
    """
    from emitters import FixedFrequencyEmitter, FrequencyAgileEmitter, PeriodicScanEmitter
    return [
        FixedFrequencyEmitter("radio-1", num_bands, band=2, p_burst_start=0.15, burst_len=3),
        FrequencyAgileEmitter("jammer-1", num_bands, p_transmit=0.25, seed=1),
        PeriodicScanEmitter("radar-1", num_bands, scan_bands=[0, 3, 5, 7], dwell=2),
    ]


def run_simulation(scheduler: Scheduler, num_bands: int, num_steps: int, emitters=None):
    """
    Runs one full simulation:
      1. build a fresh environment (new emitters, empty history)
      2. build a fresh receiver (empty scan_log)
      3. tick forward num_steps times, letting `scheduler` choose the
         band each tick, and the receiver scan it
      4. return (environment, receiver) so the caller can compute
         whatever metrics/printouts it wants

    `emitters` can be passed in explicitly so different schedulers can be
    compared on the EXACT same emitter behavior (same random rolls) -
    otherwise we build a fresh default set.
    """
    if emitters is None:
        emitters = make_default_emitters(num_bands)

    env = RFEnvironment(num_bands, emitters)
    receiver = Receiver(num_bands)

    for t in range(num_steps):
        truth_row = env.step(t)
        band = scheduler.choose_band(t, receiver)
        receiver.scan(t, band, truth_row)

    return env, receiver