"""Training-only memory saving and physical-time PPO diagnostics."""

import math

import torch
from torch.utils.checkpoint import checkpoint

from embodied_fly.brain import BrainOutput


def recurrent_forward(brain, observation, state, activity, time_scale, recompute=False):
    """Same actor and gradients; recompute activations only during backpropagation."""

    def forward(obs, memory, choice):
        result = brain(obs, memory, activity_override=choice, time_scale=time_scale)
        return (
            result.action,
            result.state,
            result.utility_logits,
            result.utility_scores,
            result.activity,
        )

    if recompute and torch.is_grad_enabled():
        return BrainOutput(
            *checkpoint(forward, observation, state, activity, use_reentrant=False)
        )
    return BrainOutput(*forward(observation, state, activity))


def physical_timescales(dt, gamma, gae_lambda, horizon, sequence):
    if not 0 < gamma < 1 or not 0 < gae_lambda <= 1:
        raise ValueError("Discount must be in (0,1), GAE lambda in (0,1]")
    return {
        "action_seconds": dt,
        "rollout_seconds": horizon * dt,
        "recurrent_gradient_seconds": sequence * dt,
        "discount_efold_seconds": -dt / math.log(gamma),
        "gae_trace_efold_seconds": -dt / math.log(gamma * gae_lambda),
        "interpretation": "Trace decay is not a hard horizon; the critic bootstraps later returns",
    }
