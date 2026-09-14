"""Frozen-actor exploration sweep and fixed-return critic capacity diagnostic.

No actor updates or teacher. Critic fits predict four seconds of measured reward,
not the infinite-horizon value used by PPO. They are diagnostic, never deployed.
"""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.hover_only import HoverBalancedReward, HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo import Critic
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def window_returns(reward, gamma, steps):
    """All complete finite windows; failure absorption is applied by collection."""
    reward = np.asarray(reward, np.float64)
    if reward.ndim != 2 or not 0 < gamma <= 1 or not 1 <= steps <= len(reward):
        raise ValueError("Need finite complete reward windows and valid discount")
    if not np.isfinite(reward).all():
        raise ValueError("Nonfinite reward")
    out = np.empty_like(reward)
    carry = np.zeros(reward.shape[1])
    for t in reversed(range(len(reward))):
        carry = reward[t] + gamma * carry
        if t + steps < len(reward):
            carry -= gamma**steps * reward[t + steps]
        out[t] = carry
    return out[: len(reward) - steps + 1].astype(np.float32)


def quality(prediction, target):
    prediction, target = np.asarray(prediction), np.asarray(target)
    residual = target - prediction
    variance = float(np.var(target))
    correlation = None
    if np.std(prediction) > 1e-8 and np.std(target) > 1e-8:
        correlation = float(np.corrcoef(prediction, target)[0, 1])
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "target_std": float(np.std(target)),
        "explained_variance": 1 - float(np.var(residual)) / max(variance, 1e-8),
        "pearson": correlation,
    }


def fit_capacity(network, x, y, train, test, *, steps, seed, normalized):
    """Identical fresh networks/batches; change only scalar target normalization."""
    start = time.perf_counter()
    model = copy.deepcopy(network)
    device = x.device
    train_ids, test_ids = np.flatnonzero(train), np.flatnonzero(test)
    offset = y[train_ids].mean() if normalized else y.new_tensor(0)
    scale = y[train_ids].std().clamp_min(0.01) if normalized else y.new_tensor(1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, eps=1e-5)
    rng = np.random.default_rng(seed)
    losses = []
    synchronize(device)
    optimization_start = time.perf_counter()
    for _ in range(steps):
        ids = rng.choice(train_ids, size=min(512, len(train_ids)), replace=False)
        prediction = model(x[ids]).squeeze(-1)
        loss = ((prediction - (y[ids] - offset) / scale) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite diagnostic critic loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        losses.append(float(loss.detach()))
    synchronize(device)
    optimization_seconds = time.perf_counter() - optimization_start
    with torch.no_grad():
        prediction = model(x).squeeze(-1) * scale + offset
    report = {
        "fit_wall_seconds": time.perf_counter() - start,
        "optimization_seconds": optimization_seconds,
        "updates": steps,
        "training_samples": len(train_ids),
        "test_samples": len(test_ids),
        "normalization_mean": float(offset),
        "normalization_scale": float(scale),
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "train": quality(prediction[train_ids].cpu(), y[train_ids].cpu()),
        "test": quality(prediction[test_ids].cpu(), y[test_ids].cpu()),
    }
    # Fold the scalar transform into the output layer: same critic architecture.
    with torch.no_grad():
        model[-1].weight.mul_(scale)
        model[-1].bias.mul_(scale).add_(offset)
        torch.testing.assert_close(model(x).squeeze(-1), prediction, rtol=1e-4, atol=1e-5)
    return model, report


def run(args):
    if (
        args.worlds != 64
        or not math.isfinite(args.seconds)
        or args.seconds < 5
        or args.fit_steps < 1
    ):
        raise ValueError("This declared sweep uses64worlds, >=5seconds and positive fit steps")
    wall_start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance, start = evidence(), time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    actor.requires_grad_(False)
    initial_actor = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    env = FlyBatch(
        64,
        16,
        actor.sensor_extension_size,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
    )
    assert physical_contract(env.model) == parent["physical_contract"]
    tasks = HoverOnlyTasks(env, args.seed)
    reward_fn = HoverBalancedReward(env, parent["config"]["hover_vertical_speed_scale"])
    critic = Critic(actor).to(device)
    old = torch.load(args.critic, map_location=device, weights_only=False)
    assert old["physical_contract"] == parent["physical_contract"]
    assert old["graph_sha256"] == parent["graph_sha256"]
    critic.load_state_dict(old["critic_state_dict"])
    noise = np.repeat([0.0, 0.001, 0.003, 0.006], 16)
    noise_tensor = torch.as_tensor(noise, device=device, dtype=torch.float32)[:, None]
    state = actor.initial_state(64)
    failed_at = np.full(64, np.nan)
    rows, anchors = [], []
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    synchronize(device)
    setup_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with torch.no_grad():
        for t in range(round(args.seconds / env.control_dt)):
            obs = torch.as_tensor(env.observation(), device=device)
            if t % 25 == 0:
                features = critic.features(actor, obs, state)
                anchors.append(
                    {
                        "index": t,
                        "features": features.cpu().numpy(),
                        "critic_value": critic.network(features).squeeze(-1).cpu().numpy(),
                        "qpos": env.fields["qpos"].copy(),
                        "qvel": env.fields["qvel"].copy(),
                    }
                )
            output = actor(obs, state)
            state = output.state
            # Same16 independent random streams repeated across noise levels.
            eps = torch.randn(16, actor.action_size, device=device).repeat(4, 1)
            latent = torch.atanh(output.action.clamp(-0.9999, 0.9999)) + noise_tensor * eps
            command = latent.tanh()
            command[:16] = output.action[:16]
            previous = env.previous_action.copy()
            env.step(command.cpu().numpy())
            reward, failed, _ = reward_fn(previous)
            alive_before = np.isnan(failed_at)
            failed_at[alive_before & failed] = (t + 1) * env.control_dt
            reward = np.where(alive_before, reward, 0)
            rows.append(
                {
                    "reward": reward.copy(),
                    "action": command.cpu().numpy(),
                    "position_cm": env.fields["qpos"][:, :3].copy(),
                    "velocity_cm_s": env.fields["qvel"][:, :3].copy(),
                    "wing_angles": env.fields["qpos"][
                        :, env.template.wing_angle_indices
                    ].copy(),
                    "wing_speeds": env.fields["qvel"][
                        :, env.template.wing_velocity_indices
                    ].copy(),
                    "forbidden_bodyweights": env.forbidden_peak.copy() / env.body_weight,
                }
            )
    synchronize(device)
    capture_seconds = time.perf_counter() - start
    export_start = time.perf_counter()
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    anchor_arrays = {k: np.stack([r[k] for r in anchors]) for k in anchors[0]}
    np.savez_compressed(
        args.output / "capture.npz", **arrays, noise=noise, first_failure=failed_at
    )
    np.savez_compressed(args.output / "critic_anchors.npz", **anchor_arrays)
    full_returns = window_returns(arrays["reward"], math.exp(-0.002 / 5), 2000)
    valid = anchor_arrays["index"] < len(full_returns)
    x = torch.as_tensor(
        anchor_arrays["features"][valid].reshape(-1, critic.network[0].in_features),
        device=device,
    )
    y = torch.as_tensor(full_returns[anchor_arrays["index"][valid]].reshape(-1), device=device)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    # Hold out complete random streams across every noise group, not nearby frames.
    train = np.tile(np.arange(64) % 16 < 12, valid.sum())
    test = (~train) & np.tile(np.arange(64) >= 16, valid.sum())
    export_and_preparation_seconds = time.perf_counter() - export_start
    torch.manual_seed(args.seed + 1)
    fresh = Critic(actor).network.to(device)
    fits = {}
    for normalized in (False, True):
        name = "normalized_targets" if normalized else "physical_targets"
        model, fits[name] = fit_capacity(
            fresh,
            x,
            y,
            train,
            test,
            steps=args.fit_steps,
            seed=args.seed + 2,
            normalized=normalized,
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "scope": "four-second realized-return diagnostic only",
            },
            args.output / (name + ".pt"),
        )
    old_predictions = anchor_arrays["critic_value"][valid].reshape(-1)
    # Existing critic estimates a different horizon/policy: correlation only.
    old_rank = quality(old_predictions[test], y[test].cpu())["pearson"]
    with torch.no_grad():
        hidden = x[test]
        saturation = []
        for layer in critic.network:
            hidden = layer(hidden)
            if isinstance(layer, nn.Tanh):
                saturation.append(float((hidden.abs() > 0.95).float().mean()))
    groups = []
    for group, scale in enumerate([0.0, 0.001, 0.003, 0.006]):
        ids = np.arange(16 * group, 16 * (group + 1))
        errors = (arrays["position_cm"][:, ids] - tasks.start[ids]) * 10
        groups.append(
            {
                "noise": scale,
                "worlds": 16,
                "surviving_worlds": int(np.isnan(failed_at[ids]).sum()),
                "first_failures_seconds": [
                    None if np.isnan(v) else float(v) for v in failed_at[ids]
                ],
                "mean_return": float(arrays["reward"][:, ids].sum(0).mean()),
                "std_return": float(arrays["reward"][:, ids].sum(0).std()),
                "position_rms_mm_by_world": np.sqrt(
                    np.mean(np.sum(errors**2, axis=-1), axis=0)
                ).tolist(),
                "altitude_rms_mm_by_world": np.sqrt(
                    np.mean(errors[:, :, 2] ** 2, axis=0)
                ).tolist(),
            }
        )
    assert all(torch.equal(v.cpu(), initial_actor[k]) for k, v in actor.state_dict().items())
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "total_wall_seconds": time.perf_counter() - wall_start,
        "trace_export_and_fit_preparation_seconds": export_and_preparation_seconds,
        "checkpoint_sha256": sha256(args.checkpoint),
        "critic_checkpoint_sha256": sha256(args.critic),
        "physical_contract": physical_contract(env.model),
        "reward_recipe": reward_fn.recipe,
        "model_sha256": sha256(args.output / "model.mjb"),
        "capture_sha256": sha256(args.output / "capture.npz"),
        "critic_anchors_sha256": sha256(args.output / "critic_anchors.npz"),
        "setup_seconds": setup_seconds,
        "capture_seconds": capture_seconds,
        "worlds": 64,
        "physics_threads": 16,
        "physics_hz": 1000,
        "action_hz": 500,
        "physical_transitions": len(rows) * 64,
        "actor_updates": 0,
        "actor_unchanged": True,
        "live_resets": 0,
        "groups": groups,
        "critic_fits": fits,
        "existing_critic_rank_correlation": old_rank,
        "existing_critic_hidden_saturation": saturation,
        "critic_scope": "4s measured discounted rewards, gamma=exp(-.002/5), zero after first failure; no bootstrap. Test holds out4of16 complete random streams across nonzero noise groups, excluding deterministic duplicates. Finite-horizon capacity diagnostic, not PPO value accuracy. Existing critic is from the recent PPO child and estimates a different policy/horizon; its correlation is descriptive only.",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "capture_seconds",
                    "groups",
                    "critic_fits",
                    "existing_critic_rank_correlation",
                    "existing_critic_hidden_saturation",
                )
            },
            indent=2,
        )
    )
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "critic", "graph", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--fit-steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=120501)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
