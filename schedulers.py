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

import math
import random
from abc import ABC, abstractmethod
from collections import Counter
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


class EpsilonGreedyScheduler(Scheduler):
    """
    The simplest LEARNING scheduler: track each band's hit rate so far
    (hits / times scanned), and:
      - with probability epsilon: scan a totally random band (EXPLORE)
      - otherwise: scan whichever band has the best hit rate so far
        (EXPLOIT)

    A band that has never been scanned has an undefined hit rate - we
    treat those as "infinitely interesting" so every band gets tried
    at least once before we start exploiting, otherwise we could get
    unlucky and permanently ignore a great band we just haven't tried.
    """

    def __init__(self, num_bands: int, epsilon: float = 0.1, seed=None):
        super().__init__(num_bands)
        self.epsilon = epsilon
        self._rng = random.Random(seed)

    def choose_band(self, t: int, receiver: Receiver) -> int:
        scan_counts = receiver.scan_count_per_band()
        # Force trying every band at least once before anything clever.
        never_scanned = [b for b in range(self.num_bands) if scan_counts[b] == 0]
        if never_scanned:
            return self._rng.choice(never_scanned)

        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.num_bands)  # EXPLORE

        hit_counts = receiver.hit_count_per_band()
        hit_rates = {b: hit_counts[b] / scan_counts[b] for b in range(self.num_bands)}
        best_rate = max(hit_rates.values())
        # Ties happen a lot early on (e.g. two bands both at 0 hits/1 scan).
        # Break ties randomly instead of always picking the lowest band
        # index, so we don't introduce a silent, biased preference.
        best_bands = [b for b, rate in hit_rates.items() if rate == best_rate]
        return self._rng.choice(best_bands)  # EXPLOIT


class UCBScheduler(Scheduler):
    """
    Upper Confidence Bound (UCB1) - a more principled bandit algorithm
    than epsilon-greedy. Instead of exploring randomly, it computes an
    "optimistic score" for every band:

        score = hit_rate + c * sqrt( ln(total_scans) / times_scanned )

    The second term is a bonus that's LARGE when a band has been
    scanned rarely (we're uncertain about it, so give it the benefit
    of the doubt) and SHRINKS toward zero the more it gets scanned (as
    we become confident its hit_rate estimate is accurate). We always
    scan whichever band has the highest score.

    `c` controls how much we value exploration vs trusting known
    hit rates. c=2.0 is a common, reasonable default.
    """

    def __init__(self, num_bands: int, c: float = 2.0, seed=None):
        super().__init__(num_bands)
        self.c = c
        self._rng = random.Random(seed)  # only used for tie-breaking

    def choose_band(self, t: int, receiver: Receiver) -> int:
        scan_counts = receiver.scan_count_per_band()
        never_scanned = [b for b in range(self.num_bands) if scan_counts[b] == 0]
        if never_scanned:
            return self._rng.choice(never_scanned)

        hit_counts = receiver.hit_count_per_band()
        total_scans = sum(scan_counts.values())

        scores = {}
        for b in range(self.num_bands):
            hit_rate = hit_counts[b] / scan_counts[b]
            exploration_bonus = self.c * math.sqrt(math.log(total_scans) / scan_counts[b])
            scores[b] = hit_rate + exploration_bonus

        best_score = max(scores.values())
        best_bands = [b for b, s in scores.items() if s == best_score]
        return self._rng.choice(best_bands)


class QLearningScheduler(Scheduler):
    """
    A tabular Q-learning scheduler. Unlike the bandits above, its state
    includes TIME (t % period), not just per-band averages - so it CAN
    in principle learn "band 0 is hot specifically at phase 0 of the
    cycle", which is exactly what beat the bandits in Stage 5.

    Two distinct usage modes, controlled by `epsilon`:
      - TRAINING: epsilon > 0 (e.g. 0.1) so it still explores while its
        Q-table is updated after every scan via .update().
      - EVALUATION/DEPLOYMENT: epsilon = 0 (or very small) so it always
        exploits the Q-table it already learned, no more updates.

    `period` defaults to num_bands - a reasonable starting assumption
    when we don't know a periodic emitter's true cycle length (in
    Stage 7 we'll handle discovering the ACTUAL period properly; this
    is a simpler, more general stand-in).
    """

    def __init__(self, num_bands: int, period: int = None, epsilon: float = 0.0, seed=None):
        super().__init__(num_bands)
        self.period = period or num_bands
        self.epsilon = epsilon
        self._rng = random.Random(seed)
        self.q_table: dict[int, list[float]] = {}

    def _q_values(self, state: int) -> list[float]:
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_bands
        return self.q_table[state]

    def choose_band(self, t: int, receiver: Receiver) -> int:
        state = t % self.period
        if self._rng.random() < self.epsilon:
            return self._rng.randrange(self.num_bands)  # explore (training only)
        q = self._q_values(state)
        best_q = max(q)
        best_actions = [a for a, v in enumerate(q) if v == best_q]
        return self._rng.choice(best_actions)  # exploit learned Q-table

    def update(self, state: int, action: int, reward: float, next_state: int,
               alpha: float = 0.1, gamma: float = 0.5):
        """
        The Q-learning update rule. Only called during TRAINING (see
        train_qlearning.py) - never during a normal evaluation run,
        since real deployment shouldn't keep rewriting the policy
        mid-mission based on a single scan.
        """
        q = self._q_values(state)
        next_q = self._q_values(next_state)
        td_target = reward + gamma * max(next_q)
        q[action] += alpha * (td_target - q[action])


class PeriodicLockScheduler(Scheduler):
    """
    Discovers unknown periodic emitters purely from observed hit timing
    (like a simplified TOA-difference / PRI histogram, as used in real
    ESM systems - see Stage 7 explanation) and then locks onto them.

    Three phases, all driven entirely by receiver.scan_log - never by
    truth:
      1. PROBE (t < probe_ticks): fall back to a simple scheduler
         (default: round robin) to gather scattered observations
         across all bands.
      2. ESTIMATE (runs once, right after probing ends): for every
         band, look at the gaps between its hit-ticks. If one gap
         value dominates (appears often enough to be confident, not
         just coincidence), record that as the band's estimated
         (period, phase).
      3. LOCK: on each future tick, if any locked band is PREDICTED to
         be active right now (based on its estimated period/phase),
         scan it. Otherwise, fall back to the probe strategy - keeps
         covering bands that turned out not to be cleanly periodic
         (bursty or random emitters) instead of ignoring them forever.
    """

    def __init__(self, num_bands: int, probe_ticks: int = None,
                 min_hits_to_estimate: int = 3, confidence_ratio: float = 0.5,
                 fallback: Scheduler = None, seed=None):
        super().__init__(num_bands)
        self.probe_ticks = probe_ticks if probe_ticks is not None else num_bands * 4
        self.min_hits_to_estimate = min_hits_to_estimate
        self.confidence_ratio = confidence_ratio  # fraction of gaps that must agree to "lock"
        self.fallback = fallback or RoundRobinScheduler(num_bands)
        self._rng = random.Random(seed)
        self.locked_bands: dict[int, tuple[int, int]] = {}  # band -> (period, base_phase)
        self.windows: dict[int, set[int]] = {}               # band -> set of confirmed-active offsets
        self._estimated = False

    def _estimate_periods(self, receiver: Receiver):
        """
        The "PRI histogram" step: for each band, gather its hit-ticks,
        compute consecutive gaps, and check whether one gap value
        dominates strongly enough to trust as the true period.
        """
        hit_ticks_by_band: dict[int, list[int]] = {}
        for r in receiver.scan_log:
            if r.hit:
                hit_ticks_by_band.setdefault(r.band, []).append(r.t)

        for band, ticks in hit_ticks_by_band.items():
            if len(ticks) < self.min_hits_to_estimate:
                continue  # not enough evidence yet - stay unlocked, keep probing
            gaps = [ticks[i + 1] - ticks[i] for i in range(len(ticks) - 1)]
            if not gaps:
                continue
            gap_counts = Counter(gaps)
            period_guess, votes = gap_counts.most_common(1)[0]
            if votes / len(gaps) >= self.confidence_ratio:
                phase = ticks[-1] % period_guess
                self.locked_bands[band] = (period_guess, phase)
                self.windows[band] = {phase}  # start the window with just the sampled phase

    def _expand_windows(self, receiver: Receiver):
        """
        For each locked band, check whether the ticks IMMEDIATELY
        ADJACENT to its known-active phase (phase-1, phase+1, wrapped
        by period) have also been observed to be active - if so, the
        band's true dwell time is longer than 1 tick, and we grow its
        window to include that offset too. This is what lets us
        eventually catch BOTH ticks of a 2-tick dwell, not just one.
        """
        for band, (period, phase) in self.locked_bands.items():
            window = self.windows.setdefault(band, {phase})
            for candidate in [(phase + 1) % period, (phase - 1) % period]:
                if candidate in window:
                    continue
                observed = [r.hit for r in receiver.scan_log
                            if r.band == band and r.t % period == candidate]
                if len(observed) >= 2 and sum(observed) / len(observed) >= 0.5:
                    window.add(candidate)

    def _needs_testing(self, band: int, period: int, offset: int, receiver: Receiver) -> bool:
        """Have we already tested this (band, offset-in-cycle) combo enough times to judge it?"""
        tested = sum(1 for r in receiver.scan_log if r.band == band and r.t % period == offset)
        return tested < 2

    def choose_band(self, t: int, receiver: Receiver) -> int:
        if t < self.probe_ticks:
            return self.fallback.choose_band(t, receiver)

        if not self._estimated:
            self._estimate_periods(receiver)
            self._estimated = True

        self._expand_windows(receiver)

        # 1) Is a locked band predicted active (in its confirmed window) right now?
        predicted_active = [
            band for band, (period, phase) in self.locked_bands.items()
            if t % period in self.windows.get(band, {phase})
        ]
        if predicted_active:
            return self._rng.choice(predicted_active)

        # 2) Otherwise, spend this "free" tick testing an untested
        #    adjacent offset for a locked band, to try to grow its window.
        for band, (period, phase) in self.locked_bands.items():
            for candidate in [(phase + 1) % period, (phase - 1) % period]:
                if t % period == candidate and self._needs_testing(band, period, candidate, receiver):
                    return band

        # 3) Nothing productive to test right now - fall back.
        return self.fallback.choose_band(t, receiver)