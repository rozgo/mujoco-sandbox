"""JEPA latent predictor with acceleration-only, physics-integrated readout."""

import torch
from torch import nn

from embodied_fly.world_model import FlyWorldModel


class ResidualWorldModel(FlyWorldModel):
    def __init__(self, feature_size=288, action_size=78, width=64):
        super().__init__(feature_size, action_size, width)
        # Only predicted latent activity enters the learned prober. Physical
        # state is consumed by the known integrator, not a direct-state MLP.
        self.prober = nn.Sequential(
            nn.Linear(width, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 6)
        )
        nn.init.zeros_(self.prober[-1].weight)
        nn.init.zeros_(self.prober[-1].bias)
        # Bounded residual authority: cm/s^2 in world axes, rad/s^2 in body axes.
        # Zero initialization exactly recovers the analytical comparator.
        self.register_buffer("residual_scale", torch.tensor([100.0] * 3 + [20.0] * 3))

    def accelerations(self, latent):
        return self.prober(latent).tanh() * self.residual_scale

    def metric_rollout(self, *args, **kwargs):
        raise RuntimeError(
            "Residual prediction requires the physical integrator and initial inertia"
        )
