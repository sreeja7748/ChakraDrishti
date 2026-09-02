"""
emitters.py
-----------
Defines emitter behavior models. Each emitter decides, at every timestep,
which band (if any) it is transmitting on. Together, many emitters generate
the "ground truth" grid: environment[band][time] = 0 or 1.

This is the TRUTH the receiver is NOT allowed to see directly — it can only
observe whichever single band it chooses to scan at each timestep.
"""

import random
from abc import ABC, abstractmethod


class Emitter(ABC):
    """Base class every emitter type inherits from."""

    def __init__(self, name: str, num_bands: int):
        self.name = name
        self.num_bands = num_bands

    @abstractmethod
    def step(self, t: int) -> int | None:
        """
        Called once per timestep.
        Returns the band index this emitter transmits on at time t,
        or None if it is silent at time t.
        """
        raise NotImplementedError


class FixedFrequencyEmitter(Emitter):
    """
    Always lives on ONE band, but transmits in intermittent bursts
    (on/off), like a radio operator keying a mic on and off.
    Modeled as a simple duty-cycle: transmits with probability p_on
    each timestep it's "available", with bursts of length burst_len.
    """

    def __init__(self, name, num_bands, band: int, p_burst_start=0.1, burst_len=3):
        super().__init__(name, num_bands)
        self.band = band
        self.p_burst_start = p_burst_start
        self.burst_len = burst_len
        self._remaining_burst = 0

    def step(self, t: int) -> int | None:
        if self._remaining_burst > 0:
            self._remaining_burst -= 1
            return self.band
        if random.random() < self.p_burst_start:
            self._remaining_burst = self.burst_len - 1
            return self.band
        return None


class FrequencyAgileEmitter(Emitter):
    """
    Hops to a (pseudo-)random band each time it transmits, and is
    silent in between hops with probability (1 - p_transmit).
    This models an emitter deliberately evading a fixed-tuned receiver.
    """

    def __init__(self, name, num_bands, p_transmit=0.3, seed=None):
        super().__init__(name, num_bands)
        self.p_transmit = p_transmit
        self._rng = random.Random(seed)

    def step(self, t: int) -> int | None:
        if self._rng.random() < self.p_transmit:
            return self._rng.randrange(self.num_bands)
        return None


class PeriodicScanEmitter(Emitter):
    """
    Sweeps bands in a repeating cycle: band 0, 1, 2, ..., k-1, 0, 1, ...
    with a fixed dwell time per band. This is predictable in principle
    (that's the whole point of the PS's "periodic scan receiver" case)
    but a naive scheduler still won't exploit the pattern.
    """

    def __init__(self, name, num_bands, scan_bands: list[int], dwell=2, start_offset=0):
        super().__init__(name, num_bands)
        self.scan_bands = scan_bands
        self.dwell = dwell
        self.start_offset = start_offset

    def step(self, t: int) -> int | None:
        cycle_len = len(self.scan_bands) * self.dwell
        pos = (t + self.start_offset) % cycle_len
        idx = pos // self.dwell
        return self.scan_bands[idx]