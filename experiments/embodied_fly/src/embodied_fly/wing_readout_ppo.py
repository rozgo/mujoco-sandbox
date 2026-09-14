"""Exact replay optimization of the existing readout when the core is frozen.

The deployed actor always runs the full connectome. Cached features are valid
only for training on recorded observations while all upstream weights are fixed.
"""

from types import SimpleNamespace

import torch

from embodied_fly.ppo import joint_log_probability, motor_distribution


def freeze_upstream(actor):
    if actor.wing_residual is None:
        raise ValueError("Requires the existing wing readout")
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name.startswith("wing_residual."))


def motor_features(actor, state):
    return actor.motor_decoder[0](state[actor.motor_ids].T)


def readout_action(actor, features):
    hidden = actor.motor_decoder[2](actor.motor_decoder[1](features))
    logits = actor.motor_decoder[3](hidden).clone()
    logits[..., 14:20] += actor.wing_residual(features)
    return actor.motor_decoder[4](logits)


def readout_replay(actor, data, a, b, active, log_std):
    shape = data["mean_action"][a:b].shape
    action = readout_action(actor, data["motor_features"][a:b].flatten(0, 1))
    output = SimpleNamespace(action=action)
    distribution = motor_distribution(output, active, log_std, 0.0005)
    logp = joint_log_probability(
        output, distribution, data["latent"][a:b].flatten(0, 1), None, motor_only=True
    )
    return logp.reshape(shape[:2]), action.reshape(shape), None


@torch.no_grad()
def teacher_features(actor, observations):
    memory = actor.initial_state(observations.shape[0])
    features = []
    for obs in observations.unbind(1):
        result = actor(obs, memory)
        features.append(motor_features(actor, result.state))
        memory = result.state
    return torch.stack(features, 1)
