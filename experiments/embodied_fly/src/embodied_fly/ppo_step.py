"""Training-only post-update KL acceptance with exact Adam/weight rollback."""

import copy
import math

import torch


@torch.no_grad()
def motor_kl(old_action, new_action, old_log_std, new_log_std):
    """Analytic old->new Normal KL; the common tanh bijection cancels.

    Use the same atanh/clamps as sampling. Float64 accumulation keeps this
    diagnostic reliable when distributions differ by only float32 roundoff.
    """
    old_location = old_action.clamp(-0.9999, 0.9999).atanh().double()
    new_location = new_action.clamp(-0.9999, 0.9999).atanh().double()
    old_scale = old_log_std.clamp(math.log(0.0005), math.log(0.15)).double()
    new_scale = new_log_std.clamp(math.log(0.0005), math.log(0.15)).double()
    per_channel = (
        new_scale
        - old_scale
        + (old_scale.mul(2).exp() + (old_location - new_location).square())
        / (2 * new_scale.mul(2).exp())
        - 0.5
    )
    return float(per_channel.sum(-1).mean().clamp_min(0))


def bounded_step(optimizer, parameters, measure_kl, limit=0.02, max_attempts=8):
    """Retry the same clipped gradient with smaller Adam steps, then commit or undo.

    Rejected trials do not advance moments, step counters or weights. The
    accepted smaller learning rate persists. Physics is never replayed here.
    """
    parameters = list(parameters)
    weights = [p.detach().clone() for p in parameters]
    history = copy.deepcopy(optimizer.state_dict())
    rates = [g["lr"] for g in optimizer.param_groups]
    trials = []
    for attempt in range(max_attempts):
        if attempt:
            with torch.no_grad():
                for p, saved in zip(parameters, weights, strict=True):
                    p.copy_(saved)
            optimizer.load_state_dict(copy.deepcopy(history))
        for group, rate in zip(optimizer.param_groups, rates, strict=True):
            group["lr"] = rate * 0.5**attempt
        optimizer.step()
        with torch.no_grad():
            kl = float(measure_kl())
        trials.append(kl)
        if math.isfinite(kl) and kl <= limit:
            return {
                "accepted": True,
                "attempts": attempt + 1,
                "trial_kl": trials,
                "accepted_kl": kl,
                "lr": optimizer.param_groups[0]["lr"],
            }
    with torch.no_grad():
        for p, saved in zip(parameters, weights, strict=True):
            p.copy_(saved)
    optimizer.load_state_dict(copy.deepcopy(history))
    # A rejected proposal still teaches the next rollout to use a smaller step.
    for group, rate in zip(optimizer.param_groups, rates, strict=True):
        group["lr"] = rate * 0.5**max_attempts
    return {
        "accepted": False,
        "attempts": max_attempts,
        "trial_kl": trials,
        "accepted_kl": None,
        "lr": optimizer.param_groups[0]["lr"],
    }
