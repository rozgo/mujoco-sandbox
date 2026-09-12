"""Recurrent physical-outcome PPO with explicit, measured imitation rehearsal.

The deployed graph actor is unchanged. A separate critic and motor exploration
distribution exist only during training. Physics is native MuJoCo on CPU threads.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embodied_fly.batch import FlyBatch
from embodied_fly.body import CONTROL_DT
from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import load_episodes, sample, synchronize


class Critic(nn.Module):
    """Training-only value of causal observation plus preceding descending state."""

    def __init__(self, brain):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(brain.observation_size + len(brain.descending_ids), 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, brain, observation, state):
        obs = ((observation - brain.observation_mean) / brain.observation_std).clamp(-10, 10)
        features = torch.cat([obs, state[brain.descending_ids].T.detach()], dim=-1)
        return self.network(features).squeeze(-1)


def motor_distribution(output, active, log_std):
    location = torch.atanh(output.action[:, active].clamp(-0.9999, 0.9999))
    return torch.distributions.Normal(
        location, log_std.clamp(math.log(0.01), math.log(0.15)).exp()
    )


def joint_log_probability(output, distribution, latent_action, activity):
    # Stable tanh change-of-variables; sum over actual 59 controlled channels.
    jacobian = 2 * (math.log(2) - latent_action - F.softplus(-2 * latent_action))
    motor = (distribution.log_prob(latent_action) - jacobian).sum(-1)
    utility = torch.distributions.Categorical(logits=output.utility_logits).log_prob(activity)
    return motor + utility


def advantages(rewards, values, final_value, done, gamma, gae_lambda):
    """Timeout bootstrap is included in rewards; neither reset leaks next episode."""
    result = torch.zeros_like(rewards)
    carry = torch.zeros_like(final_value)
    following = final_value
    for t in reversed(range(len(rewards))):
        live = (~done[t]).float()
        delta = rewards[t] + gamma * following * live - values[t]
        carry = delta + gamma * gae_lambda * live * carry
        result[t] = carry
        following = values[t]
    return result, result + values


REWARD_RECIPE = {
    "velocity_tracking_rate": 2.0,
    "yaw_tracking_rate": 0.5,
    "upright_alive_rate": 0.5,
    "tilt_cost_rate": 0.5,
    "vertical_speed_cost_rate": 0.05,
    "forbidden_support_cost_rate": 2.0,
    "action_change_cost_rate": 0.01,
    "physical_failure_penalty": 1.0,
    "velocity_filter_seconds": 0.05,
    "moving_velocity_width_cm_s": 0.5,
    "holding_velocity_width_cm_s": 0.2,
    "yaw_width_rad_s": 0.75,
    "failure_upright_below": 0.5,
    "failure_height_below_cm": 0.06,
    "units": "rates multiplied by 0.002 s each action; failure penalty once",
}


class OutcomeReward:
    def __init__(self, env):
        self.env = env
        self.filtered_velocity = np.zeros((env.n, 6))

    def reset(self, ids):
        self.filtered_velocity[ids] = 0

    def __call__(self, previous_action):
        env = self.env
        alpha = -np.expm1(-CONTROL_DT / REWARD_RECIPE["velocity_filter_seconds"])
        velocity = env.velocity()
        self.filtered_velocity += alpha * (velocity - self.filtered_velocity)
        upright = env.fields["xmat"][:, env.template.thorax_id, 8]
        failed = (upright < 0.5) | (env.fields["qpos"][:, 2] < 0.06)
        width = np.where(np.linalg.norm(env.command[:, :2], axis=1) < 0.01, 0.2, 0.5)
        velocity_error = self.filtered_velocity[:, 3:5] - env.command[:, :2]
        terms = {
            "velocity": 2.0 * np.exp(-np.sum(velocity_error**2, axis=1) / width**2),
            "yaw": 0.5
            * np.exp(-(((self.filtered_velocity[:, 2] - env.command[:, 2]) / 0.75) ** 2)),
            "alive": 0.5 * (~failed),
            "tilt": -0.5 * (1 - upright) ** 2,
            "vertical": -0.05 * np.minimum((velocity[:, 5] / 0.5) ** 2, 10),
            "support": -2.0 * np.minimum(env.forbidden_peak / env.body_weight, 5),
            "action_change": -0.01
            * np.mean((env.previous_action - previous_action) ** 2, axis=1),
        }
        reward = sum(terms.values()) * CONTROL_DT - failed.astype(float)
        return reward.astype(np.float32), failed, terms


def train(args):
    if args.horizon % args.sequence or min(args.worlds, args.horizon, args.sequence) < 1:
        raise ValueError(
            "Positive rollout dimensions and horizon divisible by sequence required"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    started_setup = time.perf_counter()
    run_evidence = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    brain, parent = load_actor(args.resume, args.graph, device)
    brain.train()
    critic = Critic(brain).to(device)
    env = FlyBatch(args.worlds, args.threads)
    reward_fn = OutcomeReward(env)
    active = torch.as_tensor(~env.template.walking_inactive, device=device)
    log_std = nn.Parameter(
        torch.full((int(active.sum()),), math.log(args.noise), device=device)
    )
    parameters = [*brain.parameters(), log_std]
    optimizer = torch.optim.Adam(parameters, lr=args.lr, eps=1e-5)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
    core_initial = {n: p.detach().clone() for n, p in brain.core.named_parameters()}
    # Preserve the same whole-episode held-out split as the imitation experiments.
    episodes = load_episodes(args.rehearsal)
    validation_ids = set(
        np.random.default_rng(1193).permutation(len(episodes))[: max(1, len(episodes) // 4)]
    )
    rehearsal = [episode for i, episode in enumerate(episodes) if i not in validation_ids]
    cases = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0.5, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [1.5, 0, 0.75],
            [1.5, 0, -0.75],
        ],
        np.float32,
    )
    task_ids = np.arange(args.worlds) % len(cases)
    env.command[:] = cases[task_ids]
    env.reset(np.arange(args.worlds), yaw=rng.uniform(-0.2, 0.2, args.worlds))
    episode_return = np.zeros(args.worlds)
    episode_start = env.fields["qpos"][:, :2].copy()
    episode_records = []
    memory = brain.initial_state(args.worlds)
    observation = torch.as_tensor(env.observation(), device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - started_setup
    counters = {
        "rollouts": 0,
        "ppo_updates": 0,
        "transitions": 0,
        "rehearsal_frames": 0,
        "collection_seconds": 0.0,
        "optimization_seconds": 0.0,
        "ppo_optimization_seconds": 0.0,
        "rehearsal_seconds": 0.0,
    }
    training_started_utc = utc_now()
    start = time.perf_counter()
    gradient_audit = None
    failure = None
    progress = []
    try:
        while time.perf_counter() - start < args.seconds:
            collect_start = time.perf_counter()
            obs_buf, latent_buf, activity_buf, logp_buf, values, rewards, dones = (
                [],
                [],
                [],
                [],
                [],
                [],
                [],
            )
            state_starts = []
            physical_rewards = []
            term_sums = {}
            with torch.no_grad():
                for t in range(args.horizon):
                    if t % args.sequence == 0:
                        state_starts.append(memory.clone())
                    value = critic(brain, observation, memory)
                    output = brain(observation, memory, sample_activity=True)
                    distribution = motor_distribution(output, active, log_std)
                    latent = distribution.sample()
                    action = output.action.clone()
                    action[:, active] = latent.tanh()
                    logp = joint_log_probability(output, distribution, latent, output.activity)
                    previous = env.previous_action.copy()
                    next_observation = env.step(action.cpu().numpy())
                    reward, failed, terms = reward_fn(previous)
                    physical_rewards.append(float(reward.mean()))
                    for key, term in terms.items():
                        term_sums[key] = (
                            term_sums.get(key, 0.0) + float(term.mean()) / args.horizon
                        )
                    episode_return += reward
                    timeout = env.ages >= round(args.episode_seconds / CONTROL_DT)
                    done = failed | timeout
                    next_tensor = torch.as_tensor(next_observation, device=device)
                    reward_tensor = torch.as_tensor(reward, device=device)
                    if np.any(timeout & ~failed):
                        final = critic(brain, next_tensor, output.state)
                        reward_tensor += (
                            args.gamma
                            * final
                            * torch.as_tensor(timeout & ~failed, device=device)
                        )
                    obs_buf.append(observation)
                    latent_buf.append(latent)
                    activity_buf.append(output.activity)
                    logp_buf.append(logp)
                    values.append(value)
                    rewards.append(reward_tensor)
                    dones.append(torch.as_tensor(done, device=device))
                    ids = np.flatnonzero(done)
                    for i in ids:
                        episode_records.append(
                            {
                                "case_id": int(task_ids[i]),
                                "failed": bool(failed[i]),
                                "simulated_seconds": float(env.ages[i] * CONTROL_DT),
                                "return": float(episode_return[i]),
                                "distance_cm": float(
                                    np.linalg.norm(
                                        env.fields["qpos"][i, :2] - episode_start[i]
                                    )
                                ),
                            }
                        )
                    env.reset(ids, yaw=rng.uniform(-0.2, 0.2, len(ids)))
                    reward_fn.reset(ids)
                    episode_return[ids] = 0
                    episode_start[ids] = env.fields["qpos"][ids, :2]
                    task_ids[ids] = rng.integers(len(cases), size=len(ids))
                    env.command[ids] = cases[task_ids[ids]]
                    memory = brain.reset_worlds(output.state, dones[-1])
                    observation = torch.as_tensor(env.observation(), device=device)
                final_value = critic(brain, observation, memory)
            data = {
                key: torch.stack(buffer)
                for key, buffer in (
                    ("obs", obs_buf),
                    ("latent", latent_buf),
                    ("activity", activity_buf),
                    ("logp", logp_buf),
                    ("value", values),
                    ("reward", rewards),
                    ("done", dones),
                )
            }
            adv, returns = advantages(
                data["reward"],
                data["value"],
                final_value,
                data["done"],
                args.gamma,
                args.gae_lambda,
            )
            adv = (adv - adv.mean()) / adv.std().clamp_min(1e-6)
            synchronize(device)
            counters["collection_seconds"] += time.perf_counter() - collect_start
            counters["transitions"] += args.horizon * args.worlds
            update_start = time.perf_counter()
            kl_values, policy_losses, value_losses = [], [], []
            stop_epoch = False
            for _ in range(args.epochs):
                for chunk in rng.permutation(len(state_starts)):
                    a, b = chunk * args.sequence, (chunk + 1) * args.sequence
                    state = state_starts[chunk].detach()
                    new_logps, predictions = [], []
                    entropy = []
                    for t in range(a, b):
                        predictions.append(critic(brain, data["obs"][t], state))
                        result = brain(
                            data["obs"][t], state, activity_override=data["activity"][t]
                        )
                        dist = motor_distribution(result, active, log_std)
                        new_logps.append(
                            joint_log_probability(
                                result, dist, data["latent"][t], data["activity"][t]
                            )
                        )
                        entropy.append(
                            torch.distributions.Categorical(logits=result.utility_logits)
                            .entropy()
                            .mean()
                        )
                        state = brain.reset_worlds(result.state, data["done"][t])
                    new_logp = torch.stack(new_logps)
                    log_ratio = new_logp - data["logp"][a:b]
                    ratio = log_ratio.exp()
                    approx_kl = ((ratio - 1) - log_ratio).mean()
                    kl_values.append(float(approx_kl.detach()))
                    if approx_kl.detach() > args.target_kl:
                        stop_epoch = True
                        break
                    policy_loss = torch.maximum(
                        -adv[a:b] * ratio, -adv[a:b] * ratio.clamp(0.8, 1.2)
                    ).mean()
                    value_loss = F.mse_loss(torch.stack(predictions), returns[a:b])
                    loss = policy_loss - args.entropy * torch.stack(entropy).mean()
                    if not torch.isfinite(loss + value_loss):
                        raise RuntimeError("Nonfinite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if gradient_audit is None:
                        gradient_audit = {
                            n: {
                                "l2": float(p.grad.norm()),
                                "finite": bool(torch.isfinite(p.grad).all()),
                                "nonzero_cells": int((p.grad != 0).sum()),
                            }
                            for n, p in brain.core.named_parameters()
                        }
                        if not all(
                            item["finite"] and item["l2"] > 0
                            for item in gradient_audit.values()
                        ):
                            raise RuntimeError("Physical reward did not reach neural core")
                    nn.utils.clip_grad_norm_(parameters, 1.0)
                    optimizer.step()
                    value_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                    value_optimizer.step()
                    counters["ppo_updates"] += 1
                    policy_losses.append(float(policy_loss.detach()))
                    value_losses.append(float(value_loss.detach()))
                if stop_epoch:
                    break
            synchronize(device)
            ppo_seconds = time.perf_counter() - update_start
            counters["ppo_optimization_seconds"] += ppo_seconds
            rehearsal_start = time.perf_counter()
            # One short explicit rehearsal batch per rollout, including reset contexts.
            reset_start = rng.random() < 0.25
            burnin = 0 if reset_start else 8
            demo = sample(
                rehearsal,
                rng,
                burnin + args.sequence,
                args.worlds,
                device,
                reset_start=reset_start,
            )
            state = brain.initial_state(args.worlds)
            with torch.no_grad():
                for t in range(burnin):
                    state = brain(demo["observation"][t], state).state
            rehearsal_losses = []
            for t in range(burnin, burnin + args.sequence):
                result = brain(demo["observation"][t], state)
                state = result.state
                rehearsal_losses.append(
                    F.mse_loss(result.action, demo["action"][t])
                    + 0.02 * F.cross_entropy(result.utility_logits, demo["activity"][t])
                )
            rehearsal_loss = torch.stack(rehearsal_losses).mean()
            optimizer.zero_grad(set_to_none=True)
            (args.rehearsal_weight * rehearsal_loss).backward()
            nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            synchronize(device)
            rehearsal_seconds = time.perf_counter() - rehearsal_start
            counters["rehearsal_seconds"] += rehearsal_seconds
            counters["optimization_seconds"] += ppo_seconds + rehearsal_seconds
            counters["rehearsal_frames"] += args.sequence * args.worlds
            counters["rollouts"] += 1
            row = {
                **counters,
                "elapsed_seconds": time.perf_counter() - start,
                "mean_physical_reward": float(np.mean(physical_rewards)),
                "reward_rates": term_sums,
                "max_kl": max(kl_values),
                "policy_loss": float(np.mean(policy_losses)) if policy_losses else None,
                "value_loss": float(np.mean(value_losses)) if value_losses else None,
                "rehearsal_loss": float(rehearsal_loss.detach()),
                "completed_episodes": len(episode_records),
                "failed_episodes": sum(e["failed"] for e in episode_records),
            }
            progress.append(row)
            print(json.dumps(row), flush=True)
            with (args.output / "progress.jsonl").open("a") as stream:
                stream.write(json.dumps(row) + "\n")
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        raise
    finally:
        synchronize(device)
        elapsed = time.perf_counter() - start
        config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
        config["internal_steps"] = brain.internal_steps
        checkpoint = {
            "state_dict": {k: v.detach().cpu() for k, v in brain.state_dict().items()},
            "observation_size": brain.observation_size,
            "action_size": brain.action_size,
            "config": config,
            "graph_sha256": parent["graph_sha256"],
            "graph_metadata_sha256": sha256(args.graph / "brain.npz"),
            "source_commit": run_evidence["source_commit"],
            "parent_checkpoint_sha256": sha256(args.resume),
            "method": "recurrent physical-outcome PPO plus explicit imitation rehearsal",
            "critic_state_dict": critic.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "value_optimizer_state_dict": value_optimizer.state_dict(),
            "log_std": log_std.detach().cpu(),
        }
        torch.save(checkpoint, args.output / "actor.pt")
        report = {
            "provenance": run_evidence,
            "config": config,
            "reward_recipe": REWARD_RECIPE,
            "training_started_utc": training_started_utc,
            "completed_utc": utc_now(),
            "setup_seconds": setup_seconds,
            "training_wall_seconds": elapsed,
            **counters,
            "physics_backend": "native MuJoCo CPU / mjbatch",
            "physics_threads": env.batch.num_threads,
            "neural_device": str(device),
            "parallel_physics_worlds": args.worlds,
            "aggregate_simulated_seconds": counters["transitions"] * CONTROL_DT,
            "transitions_per_training_wall_second": counters["transitions"] / elapsed,
            "actor_parameters": sum(p.numel() for p in brain.parameters()),
            "critic_parameters": sum(p.numel() for p in critic.parameters()),
            "core_gradient_audit_from_physical_reward": gradient_audit,
            "core_changes": {
                n: float((p.detach() - core_initial[n]).norm())
                for n, p in brain.core.named_parameters()
            },
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "parent_checkpoint_sha256": sha256(args.resume),
            "checkpoint_sha256": sha256(args.output / "actor.pt"),
            "rehearsal_manifest_sha256": sha256(args.rehearsal / "manifest.json"),
            "rehearsal_episode_sha256": {
                p.name: sha256(p) for p in sorted(args.rehearsal.glob("episode_*.npz"))
            },
            "rehearsal_validation_indices": sorted(int(i) for i in validation_ids),
            "completed_episodes": episode_records,
            "failure": failure,
            "physical_success": "Not established by training reward; independent evaluation required",
            "optimizer_initialization": "new PPO and critic Adam; parent actor and normalization retained",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=300)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--sequence", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--episode-seconds", type=float, default=2)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--noise", type=float, default=0.04)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--entropy", type=float, default=0.001)
    parser.add_argument("--rehearsal-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=28001)
    train(parser.parse_args())
