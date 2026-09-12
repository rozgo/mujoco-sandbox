"""One controller with interchangeable private or sensor-history body context."""

import numpy as np
import torch
from torch import nn

from .env import CONTEXT_DIM, OBS_DIM


def mlp(input_dim, output_dim, hidden=128):
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ELU(),
        nn.Linear(hidden, hidden),
        nn.ELU(),
        nn.Linear(hidden, output_dim),
    )


class Policy(nn.Module):
    def __init__(self, mode="oracle", hidden=128):
        super().__init__()
        self.mode, self.hidden = mode, hidden
        self.actor = mlp(OBS_DIM + CONTEXT_DIM, 12, hidden)
        self.critic = mlp(OBS_DIM + CONTEXT_DIM + 4, 1, hidden)
        # Five chronological 100-ms windows keep the complete causal 0.5-s history
        # inexpensive. No phase clock, hand-coded gait or mirrored-action averaging.
        self.estimator = mlp(OBS_DIM * 5, CONTEXT_DIM, 64)
        self.log_std = nn.Parameter(torch.full((12,), float(np.log(0.5))))
        self.register_buffer("mean", torch.zeros(OBS_DIM))
        self.register_buffer("var", torch.ones(OBS_DIM))
        self.register_buffer("count", torch.tensor(1e-4))
        nn.init.uniform_(self.actor[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def norm(self, obs):
        return ((obs - self.mean) / self.var.clamp_min(0.01).sqrt()).clamp(-10, 10)

    def estimate(self, history):
        x = self.norm(history).reshape(-1, 5, 5, OBS_DIM).mean(2).flatten(1)
        return self.estimator(x).sigmoid()

    def forward(self, obs, context=None, history=None, critic_extra=None, support=None):
        if self.mode == "oracle":
            if context is None:
                raise ValueError("Oracle requires private body context")
            z = context
        elif self.mode == "history":
            if history is None:
                raise ValueError("History policy requires causal history")
            z = self.estimate(history)
        elif self.mode == "support":
            if support is None:
                support = torch.zeros((*obs.shape[:-1], CONTEXT_DIM), device=obs.device)
            z = 1 + support
        elif self.mode == "blind":
            z = torch.ones((*obs.shape[:-1], CONTEXT_DIM), device=obs.device)
        else:
            raise ValueError(self.mode)
        x = self.norm(obs)
        action = self.actor(torch.cat((x, z), -1))
        value = None
        if critic_extra is not None:
            critic_context = context + support if self.mode == "support" else context
            value = self.critic(
                torch.cat((x, critic_context, critic_extra), -1)
            ).squeeze(-1)
        return action, value

    @torch.no_grad()
    def enable_support(self):
        """Convert constant blind context to learnable sensor inputs, preserving output."""
        self.actor[0].bias.add_(self.actor[0].weight[:, OBS_DIM:].sum(1))
        self.actor[0].weight[:, OBS_DIM:] = 0
        self.mode = "support"

    @torch.no_grad()
    def absorb(self, obs):
        n = obs.shape[0]
        delta = obs.mean(0) - self.mean
        total = self.count + n
        variance = self.var * self.count + obs.var(0, correction=0) * n
        self.var.copy_((variance + delta.square() * self.count * n / total) / total)
        self.mean.add_(delta * n / total)
        self.count.add_(n)


def log_density(noise, log_std):
    return -0.5 * (noise * noise).sum(-1) - log_std.sum() - 0.5 * 12 * np.log(2 * np.pi)
