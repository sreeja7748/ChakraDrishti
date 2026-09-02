"""
schedulers.py
--------------
A Scheduler's ONLY job: given the tick number and the receiver's own
scan history so far (never the truth grid!), decide which single band
to scan next.

This file has the abstract contract + two "dumb" baselines. Every
future scheduler (bandit in Stage 5, RL agent in Stage 6) will be
another class that follows this exact same contract, so we can swap
them in and out without touching receiver.py or environment.py at all.
"""

import random
from abc import ABC, abstractmethod
from receiver import Receiver


class Scheduler(ABC):
    """Base contract every scheduler must follow."""

    def __init__(self, num_bands: int):
        self.num_bands = num_bands

    @abstractmethod
    def choose_band(self, t: int, receiver: Receiver) -> int:
        """
        Decide which band to scan at tick t.
        IMPORTANT: this method is only ever given `receiver` (its own
        past scan_log), never the truth grid. That's the whole rule of
        the game - the scheduler is not allowed to cheat by looking at
        the future or at bands it hasn't actually scanned.
        """
        raise NotImplementedError

    def name(self) -> str:
        return self.__class__.__name__


class RoundRobinScheduler(Scheduler):
    """
    The naive baseline from Stage 2: 0, 1, 2, ..., num_bands-1, 0, 1, ...
    Doesn't use receiver history at all - it's an "open loop" strategy,
    exactly what the problem statement says is the current, unsatisfying
    default approach.
    """

    def choose_band(self, t: int, receiver: Receiver) -> int:
        return t % self.num_bands


class RandomScheduler(Scheduler):
    """
    Picks a uniformly random band every tick. Also open-loop (ignores
    history) but without round-robin's accidental resonance risk from
    Stage 2. This is actually a meaningful baseline: it's what you'd
    get if you were scared of resonance/aliasing and wanted to avoid
    any fixed pattern at all - but it throws away information too.
    """

    def __init__(self, num_bands: int, seed=None):
        super().__init__(num_bands)
        self._rng = random.Random(seed)

    def choose_band(self, t: int, receiver: Receiver) -> int:
        return self._rng.randrange(self.num_bands)