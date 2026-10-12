"""Position-only object memory; all lifecycle timing comes from the caller's dt."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Track:
    id: int
    position: np.ndarray  # Fixed board coordinates, meters.
    seen_seconds: float = 0
    missing_seconds: float = 0
    confirmed: bool = False
    detection_index: int | None = None  # Current update only, never a persistent identity.

    @property
    def visible(self):
        return self.detection_index is not None


class ObjectMemory:
    """One-to-one 3D matching, continuous probation, and bounded lost-track memory.

    Missing a detection discards probationary tracks immediately. Confirmed
    tracks survive until forget_after seconds without a match. IDs are never
    reused, even after clear(). Input positions must share a fixed reference.
    """

    def __init__(self, *, match_distance=0.12, probation_period=1.0, forget_after=5.0):
        if (not np.isfinite([match_distance, probation_period, forget_after]).all()
                or match_distance <= 0 or probation_period < 0 or forget_after <= 0):
            raise ValueError("Require positive match distance/forget time and nonnegative probation")
        self.match_distance = match_distance
        self.probation_period = probation_period
        self.forget_after = forget_after
        self.tracks = []
        self.elapsed_seconds = 0
        self.next_id = 0

    def clear(self):
        self.tracks.clear()

    def update(self, positions, dt):
        """Observe one processed frame; dt is nonnegative video time in seconds."""
        positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        if not np.isfinite(positions).all() or not np.isfinite(dt) or dt < 0:
            raise ValueError("Require finite positions and finite nonnegative dt")
        self.elapsed_seconds += dt
        for track in self.tracks:
            track.missing_seconds += dt
            track.detection_index = None
        # Expire before matching: a jump beyond the retention time starts new IDs.
        self.tracks = [t for t in self.tracks if t.missing_seconds < self.forget_after]
        matched = set()
        if self.tracks and len(positions):
            old = np.array([t.position for t in self.tracks])
            distance = np.linalg.norm(old[:, None] - positions[None, :], axis=2)
            penalty = (len(self.tracks) + 1) * self.match_distance
            # Dummy columns let tracks remain unmatched. Penalize invalid pairs
            # before assignment, so they cannot steal a valid detection.
            cost = np.column_stack((np.where(distance <= self.match_distance, distance, 2*penalty),
                                    np.full((len(old), len(old)), penalty)))
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if column >= len(positions):
                    continue
                track = self.tracks[row]
                track.position = positions[column].copy()
                track.detection_index = int(column)
                track.missing_seconds = 0
                track.seen_seconds += dt
                track.confirmed |= track.seen_seconds >= self.probation_period
                matched.add(int(column))
        self.tracks = [t for t in self.tracks if t.visible or t.confirmed]
        for index, position in enumerate(positions):
            if index not in matched:
                self.tracks.append(Track(self.next_id, position.copy(),
                                         confirmed=self.probation_period == 0, detection_index=index))
                self.next_id += 1
        return self.tracks
