"""Seeded, indefinitely extensible command variation for future teacher data.

This module generates requested velocities, never body forces or actor outputs.
It is not used by velocity_recovery_dataset_01 or its current training block.
The explicit full-skill exercises remain the coverage and comparison references.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=512)
def _target(seed, index, brake_every, translation_cm_s, yaw_rad_s):
    if index < 0 or index % brake_every == 0:
        return (0.0, 0.0, 0.0, 0.0)
    # Reproducible random access with a bounded cache and no controller instances.
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(index)]))
    value = rng.uniform(-1, 1, 4)
    value[rng.random(4) < 0.25] = 0
    value *= [translation_cm_s] * 3 + [yaw_rad_s]
    return tuple(value.tolist())


@dataclass(frozen=True)
class SmoothRandomCommands:
    seed: int = 141101
    translation_cm_s: float = 1.5
    yaw_rad_s: float = 4.5
    segment_seconds: float = 2.0
    ramp_seconds: float = 1.0
    brake_every: int = 5

    def __post_init__(self):
        numbers = [
            self.translation_cm_s,
            self.yaw_rad_s,
            self.segment_seconds,
            self.ramp_seconds,
        ]
        if not np.isfinite(numbers).all() or min(numbers) <= 0:
            raise ValueError("Positive finite command limits and timing required")
        if self.ramp_seconds > self.segment_seconds or self.brake_every < 2 or self.seed < 0:
            raise ValueError(
                "Ramp must fit the segment; periodic braking and nonnegative seed required"
            )

    def target(self, index):
        return _target(
            self.seed, index, self.brake_every, self.translation_cm_s, self.yaw_rad_s
        )

    def command(self, seconds):
        if not np.isfinite(seconds) or seconds < 0:
            raise ValueError("Finite nonnegative simulation time required")
        index = int(seconds // self.segment_seconds)
        elapsed = seconds - index * self.segment_seconds
        u = min(elapsed / self.ramp_seconds, 1.0)
        blend = u**3 * (10 - 15 * u + 6 * u * u)
        previous = np.asarray(self.target(index - 1))
        return previous + blend * (np.asarray(self.target(index)) - previous)

    def report(self):
        return {
            "seed": self.seed,
            "translation_bound_cm_s": self.translation_cm_s,
            "yaw_bound_rad_s": self.yaw_rad_s,
            "segment_seconds": self.segment_seconds,
            "ramp_seconds": self.ramp_seconds,
            "brake_every_segments": self.brake_every,
            "continuity": "C2 quintic ramps: command, first and second derivatives continuous",
            "scope": "Command generator only; not incorporated in the current recovery training data",
            "vertical_excursions": "Unconstrained integrated displacement; physical data collection must explicitly bound flight altitude or end at its declared envelope",
        }
