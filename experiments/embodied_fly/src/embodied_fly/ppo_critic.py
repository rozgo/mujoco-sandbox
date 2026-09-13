"""Train the value network on saved causal inputs, independently of actor KL."""

import torch
from torch import nn
from torch.nn import functional as F


def fit_critic(critic, optimizer, features, returns, sequence, epochs, rng):
    """Each epoch visits every rollout sample once; no actor/physics invocation.

    Features contain the observation and preceding descending-neuron state from
    collection. Targets are the fixed, timeout-aware GAE returns for that rollout.
    Detaching both prevents value fitting from changing the deployed brain.
    """
    features, returns = features.detach(), returns.detach()
    if sequence < 1 or epochs < 1 or len(features) % sequence:
        raise ValueError("Positive critic epochs and complete sequence batches required")
    if features.shape[:2] != returns.shape:
        raise ValueError("Critic features/returns must share time and world dimensions")
    chunks = len(features) // sequence

    @torch.no_grad()
    def predict():
        return torch.cat(
            [critic.network(x).squeeze(-1) for x in features.split(sequence)], dim=0
        )

    before = predict()
    losses = []
    for _ in range(epochs):
        for chunk in rng.permutation(chunks):
            a, b = chunk * sequence, (chunk + 1) * sequence
            prediction = critic.network(features[a:b]).squeeze(-1)
            loss = F.mse_loss(prediction, returns[a:b])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite independent critic loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite independent critic gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    after = predict()
    return {
        "losses": losses,
        "updates": len(losses),
        "sample_presentations": returns.numel() * epochs,
        "fit_mse_before": float(F.mse_loss(before, returns)),
        "fit_mse_after": float(F.mse_loss(after, returns)),
        "predictions": after,
    }


def value_quality(predictions, returns, task_ids, task_names):
    """Describe fixed-return fit per task; this is not held-out value accuracy."""
    result = {}
    for i, name in enumerate(task_names):
        mask = torch.as_tensor(task_ids == i, device=returns.device)
        if mask.any():
            target = returns[:, mask]
            residual = target - predictions[:, mask]
            variance = target.var(unbiased=False)
            result[name] = {
                "return_mean": float(target.mean()),
                "return_std": float(variance.sqrt()),
                "prediction_rmse": float(residual.square().mean().sqrt()),
                "explained_variance": float(
                    1 - residual.var(unbiased=False) / variance.clamp_min(1e-8)
                ),
            }
    return result
