"""
environment.py
---------------
Combines multiple Emitter objects into a simulated RF environment and
produces the ground-truth grid: truth[t][band] = 1 if ANY emitter is
transmitting on that band at that time, else 0.

This grid is used for:
  1. Driving what the receiver actually observes when it scans a band.
  2. Scoring the scheduler afterwards (things it never got to see live).
"""

from emitters import Emitter


class RFEnvironment:
    def __init__(self, num_bands: int, emitters: list[Emitter]):
        self.num_bands = num_bands
        self.emitters = emitters
        self.truth_log: list[list[int]] = []  # filled in as we simulate

    def step(self, t: int) -> list[int]:
        """
        Advance the world by one timestep. Returns the ground-truth row:
        a list of length num_bands, 1 where a transmission is happening.
        """
        row = [0] * self.num_bands
        for e in self.emitters:
            band = e.step(t)
            if band is not None:
                row[band] = 1
        self.truth_log.append(row)
        return row

    def run(self, num_steps: int) -> list[list[int]]:
        """Simulate num_steps timesteps and return the full truth grid."""
        for t in range(num_steps):
            self.step(t)
        return self.truth_log