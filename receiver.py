"""
receiver.py
------------
Models the physical receiver. It can only look at ONE band per timestep.
It has NO access to the full truth grid — only to whatever single cell
it chooses to look at, each tick. Everything it "knows" comes from its
own scan_log, built up one entry at a time as it goes.
"""


class ScanRecord:
    """
    One entry in the receiver's memory: what happened on ONE scan.
    Using a small class instead of a raw tuple/dict just so the fields
    have names (record.band, not record[0]) - easier to read later.
    """

    def __init__(self, t: int, band: int, hit: bool):
        self.t = t          # which tick this scan happened on
        self.band = band    # which band it looked at
        self.hit = hit      # True if it caught a transmission, False if not

    def __repr__(self):
        # __repr__ controls how this object looks when you print() it.
        # Purely for readability when debugging.
        status = "HIT" if self.hit else "miss"
        return f"t={self.t} band={self.band} {status}"


class Receiver:
    def __init__(self, num_bands: int):
        self.num_bands = num_bands
        self.scan_log: list[ScanRecord] = []  # its entire memory, built over time

    def scan(self, t: int, band: int, truth_row: list[int]) -> bool:
        """
        Look at ONE band, at tick t.
        truth_row is the ground-truth row for JUST this tick (a list of
        0/1 across all bands) - we only ever read ONE element out of it,
        at index `band`. We never look at the rest of the row.

        Returns True (hit) or False (miss), and also logs the result.
        """
        hit = bool(truth_row[band])
        self.scan_log.append(ScanRecord(t, band, hit))
        return hit

    # --- convenience helpers for later stages (schedulers/metrics) ---

    def hits(self) -> list[ScanRecord]:
        """All scan records that were hits."""
        return [r for r in self.scan_log if r.hit]

    def misses(self) -> list[ScanRecord]:
        """All scan records that were misses."""
        return [r for r in self.scan_log if not r.hit]

    def hit_count_per_band(self) -> dict[int, int]:
        """
        How many times each band has produced a hit so far.
        This is exactly the kind of memory a smart scheduler will lean on:
        'band 2 has been hot lately, maybe look there more.'
        """
        counts = {b: 0 for b in range(self.num_bands)}
        for r in self.hits():
            counts[r.band] += 1
        return counts

    def scan_count_per_band(self) -> dict[int, int]:
        """How many times each band has been looked at so far (hit or miss)."""
        counts = {b: 0 for b in range(self.num_bands)}
        for r in self.scan_log:
            counts[r.band] += 1
        return counts