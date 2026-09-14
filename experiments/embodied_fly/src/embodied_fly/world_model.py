"""Small causal JEPA-style physical world model; never part of the acting fly."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embodied_fly.world_data import DT, HISTORY, METRIC_SCALE


class CausalLayer(nn.Module):
    def __init__(self, incoming, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(incoming, width, 3, dilation=dilation)
        self.skip = nn.Conv1d(incoming, width, 1) if incoming != width else nn.Identity()

    def forward(self, x):
        return F.gelu(self.conv(F.pad(x, (self.padding, 0))) + self.skip(x))


class FlyWorldModel(nn.Module):
    def __init__(self, feature_size, action_size=78, width=64):
        super().__init__()
        self.config = {
            "feature_size": feature_size,
            "action_size": action_size,
            "width": width,
        }
        self.register_buffer("mean", torch.zeros(feature_size))
        self.register_buffer("scale", torch.ones(feature_size))
        self.encoder = nn.Sequential(
            CausalLayer(feature_size, width, 1),
            CausalLayer(width, width, 2),
            CausalLayer(width, width, 4),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_size, width), nn.GELU(), nn.Linear(width, width)
        )
        self.predictor = nn.GRU(width, width, batch_first=True)
        # Future physical deltas, conditioned on the initial physical summary.
        # This is a training/evaluation readout, not a motor-command decoder.
        self.prober = nn.Sequential(
            nn.Linear(width + 31, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 30),
        )
        nn.init.zeros_(self.prober[-1].bias)
        nn.init.normal_(self.prober[-1].weight, std=0.001)
        self.register_buffer("metric_scale", torch.tensor(METRIC_SCALE.copy()))

    def encode(self, sequence):
        x = (sequence - self.mean) / self.scale
        return self.encoder(x.transpose(1, 2)).transpose(1, 2)

    def latent_rollout(self, history, actions):
        start = self.encode(history)[:, -1]
        predicted, _ = self.predictor(
            self.action_encoder(actions), start.unsqueeze(0).contiguous()
        )
        return predicted

    def latent_loss(self, sequence, actions, directions=None):
        encoded = self.encode(sequence)
        predicted, _ = self.predictor(
            self.action_encoder(actions), encoded[:, HISTORY - 1].unsqueeze(0).contiguous()
        )
        target = encoded[:, HISTORY:]
        mse = F.mse_loss(predicted, target)
        regularizer = sigreg(
            torch.cat((encoded[:, HISTORY - 1 : HISTORY], predicted), dim=1), directions
        )
        return mse + 0.02 * regularizer, {
            "prediction": mse.detach(),
            "sigreg": regularizer.detach(),
            "latent_std": encoded[:, HISTORY:].std(dim=0).mean().detach(),
        }

    def metric_rollout(self, history, actions, initial):
        return self.probe(self.latent_rollout(history, actions), initial)

    def probe(self, predicted, initial):
        b, steps, _ = predicted.shape
        context = initial.clone()
        context[:, :3] = 0  # Absolute x/y never enter prediction; height is in history.
        t = torch.arange(1, steps + 1, device=initial.device, dtype=initial.dtype) * DT
        x = torch.cat(
            (
                predicted,
                (context / self.metric_scale)[:, None].expand(-1, steps, -1),
                t[None, :, None].expand(b, -1, -1),
            ),
            dim=-1,
        )
        result = initial[:, None] + self.prober(x) * self.metric_scale
        position = result[..., :3] + initial[:, None, 3:6] * t[None, :, None]
        return torch.cat((position, result[..., 3:]), dim=-1)


def sigreg(latents, directions=None):
    """Gaussian characteristic-function regularization, independently per time.

    Random unit projections, 17 frequencies on [-3,3] via even symmetry. This
    prevents constant latent embeddings from minimizing prediction loss alone.
    """
    b, _, dim = latents.shape
    if directions is None:
        directions = torch.randn(dim, 32, device=latents.device, dtype=latents.dtype)
    directions = F.normalize(directions, dim=0)
    projected = latents @ directions
    t = torch.linspace(0, 3, 17, device=latents.device, dtype=latents.dtype)
    expected = torch.exp(-0.5 * t.square())
    phases = projected.unsqueeze(-1) * t
    error = (phases.cos().mean(0) - expected).square() + phases.sin().mean(0).square()
    return (2 * b * torch.trapezoid(error * expected, t, dim=-1)).mean()


def physical_errors(predicted, truth):
    """Vector RMSE (not pooled scalar RMSE), evaluated on matching windows."""
    p, t = np.asarray(predicted), np.asarray(truth)
    result = {}
    for name, indices, factor in (
        ("position_mm", slice(0, 3), 10),
        ("velocity_mm_s", slice(3, 6), 10),
        ("angular_rad_s", slice(15, 18), 1),
        ("wing_angle_rad", slice(18, 24), 1),
        ("wing_speed_rad_s", slice(24, 30), 1),
    ):
        error = (p[..., indices] - t[..., indices]) * factor
        result[name] = float(np.sqrt(np.mean(np.sum(error**2, axis=-1))))
    return result


def constant_velocity(initial, steps):
    result = np.repeat(initial[:, None], steps, axis=1)
    result[..., :3] += initial[:, None, 3:6] * (np.arange(1, steps + 1) * DT)[None, :, None]
    return result
