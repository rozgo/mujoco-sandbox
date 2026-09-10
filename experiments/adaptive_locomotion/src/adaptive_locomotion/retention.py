"""Training-only reference rewards; the deployed actor remains independent."""

import numpy as np


def healthy_mask(context):
    return (context[:, :4] == 1).all(1) & (context[:, 4:16] >= 0.99).all(1)


def reference_bonus(actions, reference_actions, healthy):
    """Bounded healthy-only similarity bonus, with no target phase or trajectory."""
    error = np.mean((np.clip(actions, -3, 3) - reference_actions) ** 2, axis=1)
    return healthy * np.exp(-error / 0.25)


def reference_loss(mean, reference_actions, healthy):
    """Soft actor anchoring only on intact, full-strength training states."""
    error = (mean - reference_actions).square().mean(-1)
    return (error * healthy).sum() / healthy.sum().clamp_min(1)
