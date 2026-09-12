"""Cached anatomical binning of actual recurrent activity for observer displays.

Measured soma/tosoma coordinates locate cells; colors show the model's signed
latent state, not spikes or calibrated membrane voltages. No signal feeds control.
"""

from pathlib import Path

import numpy as np
import torch


class NeuralProjection:
    def __init__(self, positions, device="cpu", size=128):
        positions = np.asarray(positions)
        valid = np.isfinite(positions).all(1)
        self.neurons = len(positions)
        self.located = int(valid.sum())
        self.size = size
        xy = positions[valid][:, [0, 2]]
        lo, hi = xy.min(0), xy.max(0)
        span = max(float((hi - lo).max()), 1.0)
        # Preserve aspect ratio and all measured locations; no percentile crop.
        normalized = (xy - lo + (span - (hi - lo)) / 2) / span
        pixel = np.clip(np.rint(normalized * (size - 1)).astype(int), 0, size - 1)
        bins = pixel[:, 1] * size + pixel[:, 0]
        counts = np.bincount(bins, minlength=size * size)
        self.ids = torch.as_tensor(np.flatnonzero(valid), dtype=torch.long, device=device)
        self.bins = torch.as_tensor(bins, dtype=torch.long, device=device)
        self.counts = torch.as_tensor(counts, dtype=torch.float32, device=device)
        self.occupancy = (counts > 0).reshape(size, size)

    @classmethod
    def from_graph(cls, graph: Path, device="cpu", size=128):
        with np.load(graph / "brain.npz", allow_pickle=False) as data:
            return cls(data["positions"], device=device, size=size)

    @torch.no_grad()
    def project(self, state):
        """Return [world, height, width, (mean signed, mean magnitude)] on CPU."""
        if state.shape[0] != self.neurons:
            raise ValueError("Neural state does not match the anatomical atlas")
        values = state[self.ids]
        summed = state.new_zeros(self.size * self.size, state.shape[1], 2)
        summed.index_add_(0, self.bins, torch.stack((values, values.abs()), dim=-1))
        mean = summed / self.counts.clamp_min(1)[:, None, None]
        return mean.permute(1, 0, 2).reshape(-1, self.size, self.size, 2).cpu().numpy()

    def report(self):
        return {
            "neurons": self.neurons,
            "located_neurons": self.located,
            "neurons_without_locations": self.neurons - self.located,
            "projection_axes": "MaleCNS measured X/Z coordinates",
            "map_size": [self.size, self.size],
            "channels": ["mean signed latent state", "mean absolute latent state"],
            "controller_input": False,
            "physiological_spikes": False,
        }


def colorize(values, occupancy):
    """Fixed signed color scale; dark empty bins, amber positive / teal negative."""
    signed, magnitude = values[..., 0], values[..., 1]
    strength = np.clip(magnitude / 0.5, 0, 1)[..., None]
    sign = np.clip(signed / np.maximum(magnitude, 1e-6), -1, 1)[..., None]
    amber, teal = np.array([255, 195, 31]), np.array([74, 175, 166])
    hue = (sign + 1) / 2 * amber + (1 - sign) / 2 * teal
    background = np.array([17, 21, 25])
    level = 0.15 + 0.85 * strength
    result = background + (hue - background) * level
    return np.where(occupancy[..., None], result, background).astype(np.uint8)
