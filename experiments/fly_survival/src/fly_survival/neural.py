"""Frozen measured connectivity; engineered sensory drive and filtered readouts."""

import sys
import time

import numpy as np

from .paths import NEURAL_DATA, VENDOR


class NeuralPopulation:
    def __init__(self, n_flies, device="cpu", seed=42):
        sys.path.insert(0, str(VENDOR))
        from fly_brain import FlyBrain

        self.brain = FlyBrain(data=NEURAL_DATA, batch=n_flies, device=device, seed=seed)
        self.input_ids = {
            (kind, side): self.brain.cells(types, side=side)
            for kind, types in (("loom", ["LC4", "LPLC2"]), ("target", ["LC10a"]))
            for side in ("L", "R")
        }
        self.output_ids = {
            (kind, side): self.brain.cells([kind], side=side)
            for kind in ("DNa02", "DNp01")
            for side in ("L", "R")
        }
        if any(
            len(x) == 0 for x in [*self.input_ids.values(), *self.output_ids.values()]
        ):
            raise RuntimeError("Required annotated neural groups are missing")
        self.rate = np.zeros((n_flies, 2, 2), np.float32)
        self.last_spikes = [np.array([], dtype=int) for _ in range(n_flies)]
        self.total_seconds = 0.0
        self.atlas = np.load(NEURAL_DATA / "atlas.npz")
        self.locations = np.full((self.brain.n, 2), np.nan, np.float32)
        self.locations[self.atlas["indices"]] = self.atlas["xy"]

    def step(self, percepts, alive=None):
        inject = []
        n = len(percepts)
        for si, side in enumerate(("L", "R")):
            # Current local range change supplements the engineered looming detector.
            looming = np.array([max(p.threat, p.vision[si, 2]) for p in percepts])
            targets = np.array(
                [max(p.vision[si, 0], p.vision[si, 1]) for p in percepts]
            )
            if alive is not None:
                looming *= alive
                targets *= alive
            inject.extend(
                [
                    (self.input_ids["loom", side], looming * 0.45),
                    (self.input_ids["target", side], targets * 0.4),
                ]
            )
        start = time.perf_counter()
        fired = self.brain.step(inject=inject)
        self.total_seconds += time.perf_counter() - start
        self.last_spikes = [fired] if n == 1 else fired
        decay = np.exp(-0.02 / 0.12)
        self.rate *= decay
        for i, spikes in enumerate(self.last_spikes):
            for ki, kind in enumerate(("DNp01", "DNa02")):
                for si, side in enumerate(("L", "R")):
                    ids = self.output_ids[kind, side]
                    self.rate[i, ki, si] += (
                        (1 - decay) * np.isin(ids, spikes).mean() / 0.02
                    )
        # Normalize 0–50 Hz into a bounded utility feature, never treating raw Hz as probability.
        return np.clip(self.rate / 20, 0, 1)

    def activity(self, i):
        xy = self.locations[self.last_spikes[i]]
        xy = xy[np.isfinite(xy).all(axis=1)]
        if not len(xy):
            return []
        pixels = np.clip((xy * 48).astype(int), 0, 47)
        unique, counts = np.unique(pixels, axis=0, return_counts=True)
        return [
            [float(x / 48), float(y / 48), float(min(1, 0.25 + np.log1p(c) / 7))]
            for (x, y), c in zip(unique, counts)
        ]
