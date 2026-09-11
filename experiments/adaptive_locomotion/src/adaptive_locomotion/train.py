# PPO rollout/update structure adapted from Kevin Zakka's mjbatch Go1 example,
# Apache-2.0, revision 77966f85bcd8f7ef4351cb4a1a6f42e133d19725.
# No upstream gait reward, symmetry mapping, policy weights or reduced contacts.
"""Bounded PPO learning experiments with cumulative checkpoint training budgets."""

import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from .bodies import PRESETS, ROOT
from .env import CONTEXT_DIM, HISTORY, OBS_DIM, DogEnv
from .healthy_style import style_bonus, style_loss, style_mask, teacher_observation
from .policy import Policy, log_density
from .retention import healthy_mask, reference_bonus, reference_loss, single_damage_mask
from .symmetry import mirror_loss


def load_checkpoint(path, mode=None):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    net = Policy(mode or saved["mode"], saved["hidden"])
    net.load_state_dict(saved["state"])
    return net, saved


def train(
    output,
    seconds=60,
    seed=0,
    num_envs=512,
    mode="oracle",
    resume=None,
    bodies="all",
    terrain="flat",
    device="auto",
    threads=16,
    epochs=4,
    horizon=24,
    allowance=300,
    extension_reason="",
    reward_profile="adaptive",
    support_weight=2.0,
    stride_weight=0.0,
    balance_weight=0.0,
    body_motion_weight=0.0,
    damage_action_rate_weight=0.0,
    damage_angular_rate_weight=0.0,
    damage_joint_accel_weight=0.0,
    damage_flight_weight=0.0,
    retention_curriculum=False,
    reference=None,
    reference_reward_weight=0.0,
    reference_loss_weight=0.0,
    damage_healthy_reward_weight=0.0,
    damage_healthy_loss_weight=0.0,
    learning_rate=0.001,
    pair_level=None,
    single_reference=None,
    single_reference_weight=0.0,
    limb_stage=None,
    neutralize_validity=None,
    initial_std=None,
    front_reference=None,
    front_reference_scale=1.0,
    symmetry_weight=0.0,
):
    if not 0 < seconds <= allowance:
        raise ValueError("Invalid training duration for the chosen allowance")
    if allowance > 300 and not extension_reason:
        raise ValueError("An extension requires a recorded learning-based reason")
    if min(reference_reward_weight, reference_loss_weight) < 0 or learning_rate <= 0:
        raise ValueError(
            "Reference weights must be nonnegative and learning rate positive"
        )
    if (reference_reward_weight or reference_loss_weight) and reference is None:
        raise ValueError("Reference weights require a frozen reference checkpoint")
    if min(damage_healthy_reward_weight, damage_healthy_loss_weight) < 0:
        raise ValueError("Healthy-style weights must be nonnegative")
    healthy_style = bool(damage_healthy_reward_weight or damage_healthy_loss_weight)
    if healthy_style and (reference is None or mode != "blind" or limb_stage is None):
        raise ValueError(
            "Healthy style requires a healthy reference and reactive limb-loss training"
        )
    if retention_curriculum and bodies != "all":
        raise ValueError("Retention curriculum requires --bodies all")
    if single_reference_weight < 0 or (
        single_reference_weight and not single_reference
    ):
        raise ValueError(
            "Single-damage retention needs a reference and nonnegative weight"
        )
    if single_reference and not reference:
        raise ValueError("Single-damage retention also requires a healthy reference")
    if not 0 <= front_reference_scale <= 1:
        raise ValueError("Front reference scale must be in [0, 1]")
    if symmetry_weight < 0 or (symmetry_weight and mode != "blind"):
        raise ValueError("Mirror loss needs a reactive policy and nonnegative weight")
    if front_reference and (not single_reference or limb_stage != "consolidate"):
        raise ValueError(
            "Front reference requires the balanced limb consolidation stage"
        )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    if device == "auto":
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    actor_device = "cuda" if device == "cuda" else "cpu"
    setup_start = time.perf_counter()
    variants = None if bodies == "all" else [PRESETS[k] for k in bodies.split(",")]
    env = DogEnv(
        num_envs,
        seed,
        bodies=variants,
        terrain=terrain,
        threads=threads,
        randomize=True,
        faults=bodies != "healthy",
        randomize_strength=bodies != "healthy",
        reward_profile=reward_profile,
        support_weight=support_weight,
        stride_weight=stride_weight,
        balance_weight=balance_weight,
        body_motion_weight=body_motion_weight,
        damage_action_rate_weight=damage_action_rate_weight,
        damage_angular_rate_weight=damage_angular_rate_weight,
        damage_joint_accel_weight=damage_joint_accel_weight,
        damage_flight_weight=damage_flight_weight,
        retention_curriculum=retention_curriculum,
        pair_level=pair_level,
        limb_stage=limb_stage,
    )
    ancestry = 0.0
    parent = None
    if resume:
        learner, parent = load_checkpoint(resume, mode)
        ancestry = float(parent["cumulative_training_seconds"])
        if ancestry + seconds > allowance + 0.01:
            raise ValueError(
                f"Resume exceeds {allowance}-second lineage budget ({ancestry:.2f}+{seconds})"
            )
    else:
        learner = Policy(mode)
    if initial_std is not None:
        if not 0 < initial_std <= 1:
            raise ValueError("Initial exploration std must be in (0, 1]")
        with torch.no_grad():
            learner.log_std.fill_(float(np.log(initial_std)))
    if neutralize_validity:
        # Healthy pretraining never changes these bits: their normalized inputs
        # are exactly zero and the corresponding random weights are untrained.
        # Neutral initialization avoids an arbitrary 10-sigma reaction when a
        # new joint disappears; subsequent PPO updates learn these weights.
        slots = [j for j in range(12) if neutralize_validity == "all" or j % 3 != 2]
        columns = [45 + j for j in slots]
        if not torch.all(learner.mean[columns] == 1):
            raise ValueError(
                "Only previously constant validity channels may be neutralized"
            )
        with torch.no_grad():
            learner.actor[0].weight[:, columns] = 0
            learner.critic[0].weight[:, columns] = 0
    actor = copy.deepcopy(learner).to(actor_device)
    learner = learner.to(device)
    teacher = None
    if reference:
        teacher, _ = load_checkpoint(reference)
        if teacher.mode != "blind" or mode != "blind":
            raise ValueError("Reference retention currently supports reactive policies")
        teacher = teacher.to(actor_device).eval().requires_grad_(False)
    single_teacher = None
    if single_reference:
        single_teacher, _ = load_checkpoint(single_reference)
        if single_teacher.mode != "blind" or mode != "blind":
            raise ValueError("Single-damage retention requires reactive policies")
        single_teacher = single_teacher.to(actor_device).eval().requires_grad_(False)
    front_teacher = None
    if front_reference:
        front_teacher, _ = load_checkpoint(front_reference)
        if front_teacher.mode != "blind":
            raise ValueError("Limb references must use reactive policies")
        front_teacher = front_teacher.to(actor_device).eval().requires_grad_(False)
    opt = torch.optim.Adam(
        learner.parameters(), lr=learning_rate, fused=device in ("cuda", "mps")
    )
    normal_rng = np.random.default_rng(seed + 12345)
    shapes = {
        "obs": (OBS_DIM,),
        "ctx": (CONTEXT_DIM,),
        "extra": (4,),
        "act": (12,),
        "logp": (),
        "val": (),
        "rew": (),
        "alive": (),
    }
    if mode == "history":
        shapes["hist"] = (HISTORY, OBS_DIM)
    if teacher is not None:
        shapes.update(reference=(12,), healthy=())
    if healthy_style:
        shapes.update(style_mask=(12,))
    if single_teacher is not None:
        shapes.update(single_reference=(12,), single=())
    buf = {
        k: np.empty((horizon, num_envs, *shape), np.float32)
        for k, shape in shapes.items()
    }
    source = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    config = {
        "seed": seed,
        "mode": mode,
        "hidden": learner.hidden,
        "num_envs": num_envs,
        "bodies": bodies,
        "terrain": terrain,
        "reward_profile": reward_profile,
        "support_weight": support_weight,
        "stride_weight": stride_weight,
        "balance_weight": balance_weight,
        "body_motion_weight": body_motion_weight,
        "damage_action_rate_weight": damage_action_rate_weight,
        "damage_angular_rate_weight": damage_angular_rate_weight,
        "damage_joint_accel_weight": damage_joint_accel_weight,
        "damage_flight_weight": damage_flight_weight,
        "front_reference_scale": front_reference_scale,
        "symmetry_weight": symmetry_weight,
        "retention_curriculum": retention_curriculum,
        "pair_level": pair_level,
        "limb_stage": limb_stage,
        "neutralize_validity": neutralize_validity,
        "initial_std": initial_std,
        "front_reference_checkpoint": str(
            Path(front_reference).resolve().relative_to(ROOT)
        )
        if front_reference
        else None,
        "front_reference_sha256": hashlib.sha256(
            Path(front_reference).read_bytes()
        ).hexdigest()
        if front_reference
        else None,
        "single_reference_checkpoint": str(
            Path(single_reference).resolve().relative_to(ROOT)
        )
        if single_reference
        else None,
        "single_reference_sha256": hashlib.sha256(
            Path(single_reference).read_bytes()
        ).hexdigest()
        if single_reference
        else None,
        "single_reference_weight": single_reference_weight,
        "reference_checkpoint": str(Path(reference).resolve().relative_to(ROOT))
        if reference
        else None,
        "reference_sha256": hashlib.sha256(Path(reference).read_bytes()).hexdigest()
        if reference
        else None,
        "reference_reward_weight": reference_reward_weight,
        "reference_loss_weight": reference_loss_weight,
        "damage_healthy_reward_weight": damage_healthy_reward_weight,
        "damage_healthy_loss_weight": damage_healthy_loss_weight,
        "healthy_style_teacher_encoding": "nominal missing channels, intact validity; surviving feedback unchanged"
        if healthy_style
        else None,
        "learning_rate": learning_rate,
        "normalization_frozen": teacher is not None,
        "body_environment_counts": {g.body.name: g.n for g in env.groups},
        "support_rule": "terminal foot or designated distal stump; other link-ground contact penalized",
        "randomized_motor_strength": env.randomize_strength,
        "epochs": epochs,
        "horizon": horizon,
        "budget_seconds": seconds,
        "total_allowance_seconds": allowance,
        "extension_reason": extension_reason,
        "ancestry_seconds": ancestry,
        "device": device,
        "actor_device": actor_device,
        "source_commit": source,
        "source_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "physics": "CPU MuJoCo/mjbatch",
        "mujoco": mujoco.__version__,
        "contact_profile": "firm",
        "threads": threads,
        "physics_timestep_s": env.timestep,
        "control_timestep_s": 0.02,
        "setup_seconds": time.perf_counter() - setup_start,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    @torch.no_grad()
    def infer(obs, ctx, hist, extra):
        values = [torch.as_tensor(x, device=actor_device) for x in (obs, ctx, extra)]
        h = torch.as_tensor(hist, device=actor_device) if mode == "history" else None
        mean, value = actor(values[0], values[1], h, values[2])
        return mean.cpu().numpy(), value.cpu().numpy()

    def save(name, elapsed):
        saved = {
            "state": {k: v.detach().cpu() for k, v in learner.state_dict().items()},
            "mode": mode,
            "hidden": learner.hidden,
            "config": config,
            "cumulative_training_seconds": ancestry + elapsed,
            "training_seconds": elapsed,
            "parent": str(resume) if resume else None,
            "iterations": iteration,
            "transitions": transitions,
        }
        torch.save(saved, output / name)

    iteration = 0
    transitions = 0
    records = []
    start = time.perf_counter()
    deadline = start + min(seconds, allowance - ancestry) - 0.8
    save("initial.pt", 0)
    while time.perf_counter() < deadline:
        rollout_start = time.perf_counter()
        metrics = []
        falls = 0
        completed = 0
        for t in range(horizon):
            obs, ctx = env.obs(), env.context.copy()
            extra = np.concatenate((env.vel, env.pos[:, 2:3]), 1).astype(np.float32)
            history = env.history.copy() if mode == "history" else None
            mean, val = infer(obs, ctx, history, extra)
            if teacher is not None:
                with torch.no_grad():
                    target_obs = teacher_observation(obs) if healthy_style else obs
                    target, _ = teacher(
                        torch.as_tensor(target_obs, device=actor_device)
                    )
                target = target.cpu().numpy()
            if single_teacher is not None:
                with torch.no_grad():
                    single_target, _ = single_teacher(
                        torch.as_tensor(obs, device=actor_device)
                    )
                single_target = single_target.cpu().numpy()
                if front_teacher is not None:
                    with torch.no_grad():
                        front_target, _ = front_teacher(
                            torch.as_tensor(obs, device=actor_device)
                        )
                    # Training-only targets: both FR removals use the successful
                    # focused reference; other missing limbs use the earlier one.
                    single_target = np.where(
                        (ctx[:, 1] == 0)[:, None],
                        front_target.cpu().numpy(),
                        single_target,
                    )
            log_std = actor.log_std.detach().cpu().numpy()
            noise = normal_rng.standard_normal(mean.shape, dtype=np.float32)
            actions = mean + np.exp(log_std) * noise
            logp = log_density(noise, log_std)
            reward, done, fell, terms = env.step(actions)
            if teacher is not None:
                # Read health after stepping so a fault at this step immediately
                # disables imitation. No strength/geometry labels enter the actor.
                healthy = healthy_mask(env.context).astype(np.float32)
                reward += reference_reward_weight * reference_bonus(
                    actions, target, healthy
                )
            if healthy_style:
                active_style = style_mask(obs)
                reward += damage_healthy_reward_weight * style_bonus(
                    actions, target, active_style
                )
            raw_reward = float(reward.mean())
            timeout = done & ~fell
            if timeout.any():
                ex = np.concatenate((env.vel, env.pos[:, 2:3]), 1).astype(np.float32)
                _, nv = infer(env.obs(), env.context, env.history, ex)
                reward[timeout] += 0.99 * nv[timeout]
            vals = {
                "obs": obs,
                "ctx": ctx,
                "extra": extra,
                "act": actions,
                "logp": logp,
                "val": val,
                "rew": reward,
                "alive": ~done,
            }
            if history is not None:
                vals["hist"] = history
            if teacher is not None:
                vals.update(reference=target, healthy=healthy)
            if healthy_style:
                vals.update(style_mask=active_style)
            if single_teacher is not None:
                vals.update(
                    single_reference=single_target,
                    single=single_damage_mask(env.context).astype(np.float32)
                    * np.where(env.context[:, 1] == 0, front_reference_scale, 1),
                )
            for k, v in vals.items():
                buf[k][t] = v
            metrics.append([terms["track"].mean(), terms["speed"].mean(), raw_reward])
            falls += fell.sum()
            completed += done.sum()
            env.reset(np.flatnonzero(done))
        rollout_seconds = time.perf_counter() - rollout_start
        # A rollout can be discarded at the deadline; never start a long update
        # after the budget is spent. One in-progress minibatch may overshoot slightly.
        if time.perf_counter() >= deadline:
            break
        batch = {k: torch.as_tensor(v, device=device) for k, v in buf.items()}
        ex = np.concatenate((env.vel, env.pos[:, 2:3]), 1).astype(np.float32)
        _, last = infer(env.obs(), env.context, env.history, ex)
        value = torch.cat((batch["val"], torch.as_tensor(last, device=device)[None]), 0)
        adv = torch.zeros_like(batch["rew"])
        carry = 0.0
        for t in reversed(range(horizon)):
            delta = batch["rew"][t] + 0.99 * batch["alive"][t] * value[t + 1] - value[t]
            carry = delta + 0.99 * 0.95 * batch["alive"][t] * carry
            adv[t] = carry
        returns = (adv + batch["val"]).flatten()
        advantages = adv.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        flat = {k: v.reshape((-1, *v.shape[2:])) for k, v in batch.items()}
        update_start = time.perf_counter()
        for _ in range(epochs):
            for ids in torch.randperm(num_envs * horizon, device=device).chunk(4):
                if time.perf_counter() >= deadline:
                    break
                mean, v = learner(
                    flat["obs"][ids],
                    flat["ctx"][ids],
                    flat["hist"][ids] if mode == "history" else None,
                    flat["extra"][ids],
                )
                lp = log_density(
                    (flat["act"][ids] - mean) / learner.log_std.exp(), learner.log_std
                )
                ratio = (lp - flat["logp"][ids]).exp()
                objective = torch.minimum(
                    ratio * advantages[ids], ratio.clamp(0.8, 1.2) * advantages[ids]
                )
                loss = (
                    -objective.mean()
                    + 0.5 * (v - returns[ids]).square().mean()
                    - 0.005 * learner.log_std.sum()
                )
                if mode == "history":
                    loss += (
                        0.5
                        * (learner.estimate(flat["hist"][ids]) - flat["ctx"][ids])
                        .square()
                        .mean()
                    )
                if teacher is not None:
                    loss += reference_loss_weight * reference_loss(
                        mean, flat["reference"][ids], flat["healthy"][ids]
                    )
                if healthy_style:
                    loss += damage_healthy_loss_weight * style_loss(
                        mean, flat["reference"][ids], flat["style_mask"][ids]
                    )
                if single_teacher is not None:
                    loss += single_reference_weight * reference_loss(
                        mean, flat["single_reference"][ids], flat["single"][ids]
                    )
                if symmetry_weight:
                    loss += symmetry_weight * mirror_loss(
                        learner, flat["obs"][ids], mean
                    )
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    learner.log_std.clamp_(-2, 0)
        if teacher is None:
            learner.absorb(flat["obs"])
        actor.load_state_dict(learner.state_dict())
        iteration += 1
        transitions += num_envs * horizon
        elapsed = time.perf_counter() - start
        track, speed, reward = np.mean(metrics, axis=0)
        row = {
            "iteration": iteration,
            "elapsed_s": elapsed,
            "transitions": transitions,
            "track": float(track),
            "speed_mps": float(speed),
            "reward": float(reward),
            "fall_fraction": float(falls / max(completed, 1)),
            "rollout_s": rollout_seconds,
            "update_s": time.perf_counter() - update_start,
        }
        records.append(row)
        if iteration % 10 == 0 or iteration == 1:
            print(json.dumps(row), flush=True)
        if iteration % 25 == 0:
            save("latest.pt", elapsed)
            if retention_curriculum or limb_stage:
                save(f"iteration_{iteration:04d}.pt", elapsed)
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start
    save("policy.pt", elapsed)
    report = {
        **config,
        "training_seconds": elapsed,
        "cumulative_training_seconds": ancestry + elapsed,
        "transitions": transitions,
        "iterations": iteration,
        "progress": records,
        "checkpoint_sha256": hashlib.sha256(
            (output / "policy.pt").read_bytes()
        ).hexdigest(),
    }
    (output / "training.json").write_text(json.dumps(report, indent=2) + "\n")
    env.close()
    print(
        json.dumps(
            {
                "finished": str(output / "policy.pt"),
                "seconds": elapsed,
                "iterations": iteration,
            }
        ),
        flush=True,
    )
    return report
