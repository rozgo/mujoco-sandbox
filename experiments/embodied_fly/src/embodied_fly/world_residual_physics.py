"""Differentiable 1 kHz integration of the canonical fly's reduced dynamics.

The live MuJoCo plant is unchanged. Known independent wing actuation is advanced
at every physics tick. A learned probe supplies world linear and body angular
acceleration residuals, held for the two ticks of each 500 Hz command. Initial
articulation supplies locked inertia; future articulation/contact remain omitted.
"""

import numpy as np
import torch
from torch import nn

from embodied_fly.wing_motion import AGILE_CONFIG, config_for_model
from embodied_fly.world_analytic import AnalyticalFly


def skew(x: torch.Tensor) -> torch.Tensor:
    a, b, c = x.unbind(-1)
    z = torch.zeros_like(a)
    return torch.stack((z, -c, b, c, z, -a, -b, a, z), -1).reshape(-1, 3, 3)


def rotate_step(x: torch.Tensor) -> torch.Tensor:
    # Sinc has a finite value and derivative at zero; avoid norm's zero gradient
    # singularity without measurably changing the physical rotation increment.
    angle = (x.square().sum(-1) + 1e-24).sqrt()
    k = skew(x)
    a = torch.sinc(angle / torch.pi)[:, None, None]
    b = 0.5 * torch.sinc(angle / (2 * torch.pi)).square()[:, None, None]
    return torch.eye(3, dtype=x.dtype, device=x.device) + a * k + b * (k @ k)


def integrate_body(
    initial: torch.Tensor,
    wings: torch.Tensor,
    inertia: torch.Tensor,
    inverse: torch.Tensor,
    residual: torch.Tensor,
    gravity: torch.Tensor,
    mass: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """All steps remain in autograd; no detach or observed future state input."""
    dt = 0.001
    p, v = initial[:, :3], initial[:, 3:6]
    r = initial[:, 6:15].reshape(-1, 3, 3)
    omega = initial[:, 15:18]
    paths, lifts = [], []
    for tick in range(wings.shape[1] - 1):
        q = wings[:, tick, :6].reshape(-1, 2, 3)
        w = wings[:, tick, 6:].reshape(-1, 2, 3)
        effort = (w[:, :, 0].abs() / 50).clamp(0, 1.4) * (0.8 + 0.2 * (q[:, :, 2] + 1).cos())
        collective = effort.mean(-1)
        differential = effort[:, 1] - effort[:, 0]
        lift = 1.6 * gravity[2].abs() * collective  # force / mass, cm/s^2
        engaged = (collective * 2).clamp(0, 1)
        forward = ((q[:, :, 1].mean(-1) - 0.7) * 2).tanh()
        lateral = (2 * (q[:, 1, 1] - q[:, 0, 1])).tanh()
        local = torch.stack((0.25 * lift * forward, 0.25 * lift * lateral, 0.2 * lift), -1)
        up_lift = torch.stack((torch.zeros_like(lift), torch.zeros_like(lift), 0.8 * lift), -1)
        # Match the force law's body/world velocity transforms even for the
        # slight non-orthogonality in float32 recorded rotation matrices.
        world_v = (r @ (r.transpose(-1, -2) @ v.unsqueeze(-1))).squeeze(-1)
        drag = torch.stack(
            (world_v[:, 0] / 0.12, world_v[:, 1] / 0.12, world_v[:, 2] / 0.08), -1
        )
        linear = (
            (r @ local.unsqueeze(-1)).squeeze(-1) + up_lift - engaged[:, None] * drag + gravity
        )
        world_omega = (r @ omega.unsqueeze(-1)).squeeze(-1)
        up = r[:, :, 2]
        restoring = torch.stack((up[:, 1], -up[:, 0], torch.zeros_like(lift)), -1)
        steer = torch.stack(
            (
                -60 * differential,
                torch.zeros_like(lift),
                60 * differential + 120 * (q[:, 1, 2] - q[:, 0, 2]).sin(),
            ),
            -1,
        )
        angular = -world_omega / 0.025 + up * omega[:, 2, None] * (1 / 0.025 - 1 / 0.25)
        angular = angular + 100 * restoring + (r @ steer.unsqueeze(-1)).squeeze(-1)
        angular = (
            angular
            * (250 / angular.square().sum(-1).clamp_min(1e-24).sqrt()).clamp_max(1)[:, None]
        )
        torque = (
            mass
            * 0.06**2
            * engaged[:, None]
            * (r.transpose(-1, -2) @ angular.unsqueeze(-1)).squeeze(-1)
        )
        momentum = (inertia @ omega.unsqueeze(-1)).squeeze(-1)
        accel = (
            inverse @ (torque - torch.cross(omega, momentum, dim=-1)).unsqueeze(-1)
        ).squeeze(-1)
        correction = residual[:, tick // 2]
        omega = omega + dt * (accel + correction[:, 3:])
        r = r @ rotate_step(omega * dt)
        v = v + dt * (linear + correction[:, :3])
        p = p + dt * v
        lifts.append(lift / gravity[2].abs())
        if tick % 2 == 1:
            paths.append(torch.cat((p, v, r.reshape(-1, 9), omega, wings[:, tick + 1]), -1))
    return torch.stack(paths, 1), torch.stack(lifts, 1)


class DifferentiableFly(nn.Module):
    def __init__(self, model):
        super().__init__()
        if config_for_model(model) != AGILE_CONFIG or abs(model.opt.timestep - 0.001) > 1e-12:
            raise ValueError("Residual predictor requires the unchanged agile-v5 1 kHz plant")
        self.analytic = AnalyticalFly(model)
        self.mass = float(model.body_mass.sum())
        self.register_buffer(
            "gravity", torch.tensor(model.opt.gravity.copy(), dtype=torch.float64)
        )
        for key, value in self.analytic.parameters.items():
            self.register_buffer(key, torch.tensor(value.copy(), dtype=torch.float64))
        self.register_buffer("wing_ids", torch.tensor(self.analytic.indices["wing_action"]))

    def wing_rollout(self, initial, actions):
        q, v = initial[:, 18:24], initial[:, 24:30]
        paths = [torch.cat((q, v), -1)]
        commands = actions[:, :, self.wing_ids]
        for t in range(actions.shape[1]):
            target = self.lower + (commands[:, t].clamp(-1, 1) + 1) * 0.5 * (
                self.upper - self.lower
            )
            for _ in range(2):
                actuator = (self.kp * (target - q) - self.kv * v).clamp(
                    -self.limit, self.limit
                )
                force = actuator - self.damping * v - self.stiffness * (q - self.spring)
                v = v + 0.001 * force / (self.armature + 0.001 * self.damping)
                raw = q + 0.001 * v
                q = raw.clamp(self.lower, self.upper)
                v = torch.where(raw == q, v, 0)
                paths.append(torch.cat((q, v), -1))
        return torch.stack(paths, 1)

    def inertia(self, qpos, like):
        values = self.analytic.inertia(np.asarray(qpos))
        inertia = torch.as_tensor(values, dtype=like.dtype, device=like.device)
        return inertia, torch.linalg.inv(inertia)

    def forward(self, initial, wings, inertia, inverse, residual):
        return integrate_body(
            initial, wings, inertia, inverse, residual, self.gravity, self.mass
        )
