"""One shared, unmasked motor decoder for every actuator of the fly.

Two generic views of the same motor-cell activity preserve prior normalization
when consolidating old checkpoints. Both views feed one dense hidden layer and
one dense output layer. No body-part selectors exist in the deployed decoder.
"""

import torch
from torch import nn


class MotorFeatures(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.normalization = nn.LayerNorm(size)
        self.register_buffer("feature_mean", torch.zeros(size))
        self.register_buffer("feature_scale", torch.ones(size))

    def forward(self, motor):
        normalized = self.normalization(motor)
        standardized = (
            (normalized - self.feature_mean) / self.feature_scale.clamp_min(0.05)
        ).clamp(-10, 10)
        return torch.cat((normalized, standardized), -1)


class FullBodyDecoder(nn.Sequential):
    def __init__(self, inputs, hidden, outputs):
        super().__init__(
            MotorFeatures(inputs),
            nn.Linear(2 * inputs, hidden),
            nn.Tanh(),
            nn.Linear(hidden, outputs),
            nn.Tanh(),
        )


@torch.no_grad()
def consolidate(old_decoder, old_correction):
    """Algebraic checkpoint conversion only; selectors never enter live forward.

    Keep clipping exactly: folding clipped input normalization into a Linear
    layer would change the old function outside the unclipped input range.
    The new dense weights have no masks or permanent neuron/output partitions.
    """
    if old_correction is None or not hasattr(old_correction, "network"):
        raise ValueError("Consolidation requires the recorded nonlinear legacy correction")
    width, extra = old_decoder[1].out_features, old_correction.network[0].out_features
    inputs, outputs = old_decoder[1].in_features, old_decoder[3].out_features
    if outputs != 78:
        raise ValueError("Legacy conversion expects its recorded 78-output action schema")
    result = FullBodyDecoder(inputs, width + extra, outputs).to(old_decoder[1].weight)
    result[0].normalization.load_state_dict(old_decoder[0].state_dict())
    result[0].feature_mean.copy_(old_correction.feature_mean)
    result[0].feature_scale.copy_(old_correction.feature_scale)
    result[1].weight.zero_()
    result[1].weight[:width, :inputs].copy_(old_decoder[1].weight)
    result[1].weight[width:, inputs:].copy_(old_correction.network[0].weight)
    result[1].bias.copy_(torch.cat((old_decoder[1].bias, old_correction.network[0].bias)))
    result[3].weight.zero_()
    result[3].weight[:, :width].copy_(old_decoder[3].weight)
    result[3].bias.copy_(old_decoder[3].bias)
    result[3].weight[14:20, width:].copy_(old_correction.network[2].weight)
    result[3].bias[14:20].add_(old_correction.network[2].bias)
    return result


def consolidate_actor(actor):
    actor.motor_decoder = consolidate(actor.motor_decoder, actor.wing_residual)
    actor.shared_decoder_hidden = actor.motor_decoder[1].out_features
    actor.wing_residual = None
    actor.wing_residual_hidden = 0
    freeze_for_decoder_training(actor)


def freeze_for_decoder_training(actor):
    if not isinstance(actor.motor_decoder, FullBodyDecoder) or actor.wing_residual is not None:
        raise ValueError(
            "Full-body learning requires one shared decoder and no correction branch"
        )
    for name, parameter in actor.named_parameters():
        parameter.requires_grad_(name.startswith("motor_decoder."))


def motor_features(actor, state):
    # Cache RAW motor cells. Decoder LayerNorm remains trainable and must be
    # recomputed after every update; caching normalized inputs would be stale.
    return state[actor.motor_ids].T


def decode_features(actor, features):
    if actor.wing_residual is not None:
        raise ValueError("No isolated output branches in full-body decoder training")
    return actor.motor_decoder(features)
