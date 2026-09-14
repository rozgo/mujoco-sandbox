"""Bounded PPO pilot on the accepted velocity-command flight plant.

One unchanged graph actor, all 78 sampled controls, no teacher control in PPO.
One small clean PID replay batch accompanies the first actor update per rollout.
"""

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch import nn

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo import advantages, joint_log_probability, motor_distribution
from embodied_fly.ppo_critic import StandardizedValueNetwork, fit_critic
from embodied_fly.ppo_step import bounded_step, motor_kl
from embodied_fly.ppo_timing import physical_timescales, recurrent_forward
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import HoverReward, evaluate_hover, start_states
from embodied_fly.velocity_imitation import sequence_loss
from embodied_fly.velocity_motor import SCHEMA, observation
from embodied_fly.wing_position import wing_actuators
from embodied_fly.wing_readout_ppo import (
    freeze_upstream,
    motor_features,
    readout_action,
    readout_replay,
    teacher_features,
)


class HoverCritic(nn.Module):
    """Training-only causal value: observation, descending memory, reward filter, height.

    Height reveals the failure margin only to the critic. No targets, teacher
    state or future observations enter this network or the deployed actor.
    """

    def __init__(self, actor):
        super().__init__()
        size = actor.observation_size + len(actor.descending_ids) + 7
        self.network = StandardizedValueNetwork(
            nn.Linear(size, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, 1)
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def features(self, actor, obs, memory, reward, env):
        aux = np.column_stack((reward.mean(), env.fields["qpos"][:, 2]))
        return torch.cat(
            (
                obs,
                memory[actor.descending_ids].T.detach(),
                torch.as_tensor(aux, dtype=torch.float32, device=obs.device),
            ),
            dim=-1,
        ).detach()


def replay(actor, data, state, a, b, active, log_std, gradients):
    logps, actions = [], []
    context = torch.ones(data["obs"].shape[1], dtype=torch.long, device=state.device)
    for t in range(a, b):
        result = recurrent_forward(actor, data["obs"][t], state, context, 1, gradients)
        distribution = motor_distribution(result, active, log_std, 0.0005)
        logps.append(
            joint_log_probability(
                result, distribution, data["latent"][t], context, motor_only=True
            )
        )
        actions.append(result.action)
        state = actor.reset_worlds(result.state, data["done"][t])
    return torch.stack(logps), torch.stack(actions), state


@torch.no_grad()
def replay_audit(actor, data, states, sequence, active, log_std):
    maximum_action_error = maximum_logp_error = 0.0
    errors = []
    for chunk, initial in enumerate(states):
        a, b = chunk * sequence, (chunk + 1) * sequence
        logp, actions, _ = replay(actor, data, initial, a, b, active, log_std, False)
        maximum_action_error = max(
            maximum_action_error, float((actions - data["mean_action"][a:b]).abs().max())
        )
        maximum_logp_error = max(
            maximum_logp_error, float((logp - data["logp"][a:b]).abs().max())
        )
        errors.append((logp - data["logp"][a:b]).double())
    delta = torch.cat(errors)
    aggregate_kl = float((delta.expm1() - delta).mean())
    report = {
        "maximum_action_error": maximum_action_error,
        "maximum_log_probability_error": maximum_logp_error,
        "frames": data["logp"].numel(),
        "reset_events": int(data["done"].sum()),
        "roundoff_aggregate_kl": aggregate_kl,
        "mean_absolute_log_probability_error": float(delta.abs().mean()),
        "tolerances": {"action": 2e-6, "maximum_logp": 0.01, "aggregate_kl": 1e-6},
        "passed": maximum_action_error < 2e-6
        and maximum_logp_error < 0.01
        and aggregate_kl < 1e-6,
    }
    if not report["passed"]:
        raise RuntimeError(f"Recurrent replay differs before optimization: {report}")
    return report


def train(args):
    if args.horizon % args.sequence or args.worlds % 8 or min(args.seconds, args.lr) <= 0:
        raise ValueError(
            "Positive budget/LR, eight-way world split and complete sequences required"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    critic_rng = np.random.default_rng(args.seed ^ 0xC8171C)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    old_reward = parent.get("ppo_recipe", {}).get("reward", {})
    old_vertical_weight = old_reward.get("vertical_tracking_rate", 2.0)
    old_objective = old_reward.get("velocity_objective", "separate")
    changing_objective = args.resume_ppo and old_objective != args.velocity_objective
    changing_vertical = args.resume_ppo and old_vertical_weight != args.vertical_reward_weight
    changing_reward = changing_vertical or changing_objective
    if bool(args.adapt_vertical_reward) != changing_vertical:
        raise ValueError(
            "A changed vertical reward requires explicit --adapt-vertical-reward on resume"
        )
    if bool(args.adapt_velocity_objective) != changing_objective:
        raise ValueError(
            "Changed velocity objective requires explicit --adapt-velocity-objective on resume"
        )
    if changing_vertical and changing_objective:
        raise ValueError("Change the velocity objective or vertical weight separately")
    if parent.get("observation_schema") != SCHEMA or not actor.motor_only:
        raise ValueError("Requires the preserved velocity-command motor actor")
    if not args.resume_ppo and (
        sha256(args.checkpoint)
        != "c4a87e93a9fffb4679c84a6bca42716bc2e89365d48be97d9ec37780df2779e4"
    ):
        raise ValueError("This first pilot must start from the approved 5.6-second checkpoint")
    if args.resume_ppo and (
        not args.wing_readout_only
        or not parent.get("ppo_recipe", {}).get("wing_readout_only")
        or old_reward.get("horizontal_velocity_scale_cm_s", old_reward["velocity_scale_cm_s"])
        != args.horizontal_reward_scale
    ):
        raise ValueError("Continuation requires the same recorded wing-readout PPO recipe")
    actor.train()
    if args.wing_readout_only:
        freeze_upstream(actor)
    initial = {name: p.detach().cpu().clone() for name, p in actor.named_parameters()}
    env = environment(args.worlds, args.threads)
    if physical_contract(env.model) != parent["physical_contract"]:
        raise ValueError("Physical contract changed")
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    dataset_report = json.loads((args.dataset / "report.json").read_text())
    if (
        dataset_report["physical_contract"] != parent["physical_contract"]
        or not dataset_report["passed"]
    ):
        raise ValueError("Imitation anchor must use the validated matching teacher")
    for name in ("observations", "actions"):
        if sha256(args.dataset / f"{name}.npy") != dataset_report[f"{name}_sha256"]:
            raise ValueError("Demonstration checksum differs")
    # Only the first, uninterrupted two-second hover of training episodes 0..7.
    demo_obs = torch.as_tensor(
        np.load(args.dataset / "observations.npy", mmap_mode="r")[:8, :1000].copy(),
        device=device,
    )
    demo_actions = torch.as_tensor(
        np.load(args.dataset / "actions.npy", mmap_mode="r")[:8, :1000].copy(), device=device
    )
    assert torch.count_nonzero(demo_obs[:, :, 375:379]) == 0
    cached_demo = teacher_features(actor, demo_obs) if args.wing_readout_only else None
    wings = torch.as_tensor(wing_actuators(env.model), device=device)
    starts = start_states(args.dataset, range(8))
    world_episode = np.arange(args.worlds) % 8
    commands = np.zeros((args.worlds, 4), np.float32)
    reward_fn = HoverReward(
        env, args.horizontal_reward_scale, args.vertical_reward_weight, args.velocity_objective
    )

    def reset(ids):
        env.reset(ids, state={k: v[world_episode[ids]] for k, v in starts.items()})
        reward_fn.reset(ids)

    reset(np.arange(args.worlds))
    critic = HoverCritic(actor).to(device)
    active = torch.ones(78, dtype=torch.bool, device=device)
    log_std = nn.Parameter(torch.full((78,), math.log(0.003), device=device))
    log_std.requires_grad_(not args.wing_readout_only)
    parameters = [p for p in actor.parameters() if p.requires_grad]
    if log_std.requires_grad:
        parameters.append(log_std)
    # This is an explicit imitation->PPO optimizer transition, no old Adam moments.
    optimizer = torch.optim.Adam(parameters, lr=args.lr, eps=1e-5)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
    if args.resume_ppo:
        critic.load_state_dict(parent["critic_state_dict"], strict=True)
        optimizer.load_state_dict(parent["optimizer_state_dict"])
        value_optimizer.load_state_dict(parent["value_optimizer_state_dict"])
        with torch.no_grad():
            log_std.copy_(parent["log_std"].to(device))
        rng.bit_generator.state = json.loads(parent["sampler_state_json"])
        torch.set_rng_state(parent["torch_rng_state"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state(parent["cuda_rng_state"].cpu(), device)
    gamma, gae_lambda = math.exp(-0.002 / 2), math.exp(-0.002 / 0.25)
    recipe = {
        "wing_readout_only": args.wing_readout_only,
        "trainable_actor_parameters": sum(
            p.numel() for p in actor.parameters() if p.requires_grad
        ),
        "fixed_exploration": args.wing_readout_only,
        "resumed_ppo_optimizer_and_critic": args.resume_ppo,
        "worlds": args.worlds,
        "physics_threads": args.threads,
        "physics_backend": "CPU MuJoCo/mjbatch",
        "learning_device": str(device),
        "physics_hz": 1000,
        "control_hz": 500,
        "horizon": args.horizon,
        "sequence": args.sequence,
        "actor_epochs": 2,
        "critic_epochs": 4,
        "actor_lr": args.lr,
        "critic_lr": 3e-4,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip": 0.2,
        "target_kl": 0.02,
        "post_update_kl_backtracking": args.bounded_updates,
        "gradient_norm_cap": 1,
        "initial_tanh_latent_std": 0.003,
        "minimum_std": 0.0005,
        "entropy_bonus": 0,
        "imitation_weight": 1.0,
        "imitation_loss": "wing MSE + 0.1 nonwing MSE",
        "imitation_frequency": "8 sequences x 64 target steps, once on first actor minibatch per rollout",
        "imitation_context_steps": 64,
        "imitation_split": "episodes 0..7, initial 2-second hover only",
        "optimizer_transition": "fresh PPO Adam and fresh critic; all learned actor weights retained",
        "critic_warmup_rollouts": 1,
        "episode_seconds": 10,
        "recurrent_carry": "after updates reconstruct final sequence with new weights; older boundary state is detached",
        "teacher_control_share": 0,
        "additional_actuator_noise": False,
        "command": [0, 0, 0, 0],
        "reward": reward_fn.recipe,
        "timescales": physical_timescales(
            0.002, gamma, gae_lambda, args.horizon, args.sequence
        ),
    }
    if args.wing_readout_only:
        recipe.update(
            imitation_context_steps="full frozen-core history from episode start",
            recurrent_carry="exact: upstream weights frozen; retain actual live neural state",
            optimizer_transition="fresh PPO Adam for existing wing readout only; fresh critic; fixed exploration",
        )
    if args.resume_ppo:
        recipe.update(
            optimizer_transition="restore actor Adam, critic weights/Adam, fixed exploration and actor sampling RNG; reset physical episodes; critic sample shuffle restarts from declared seed",
            critic_warmup_rollouts=0,
        )
    if changing_reward:
        recipe.update(
            reward_transition={
                "vertical_tracking_rate_before": old_vertical_weight,
                "vertical_tracking_rate_after": args.vertical_reward_weight,
                "other_reward_terms_changed": False,
                "critic_adaptation": "one new-reward rollout, critic updates only; retain critic weights/Adam/calibration",
            },
            critic_warmup_rollouts=1,
        )
        if changing_objective:
            recipe["reward_transition"] = {
                "velocity_objective_before": old_objective,
                "velocity_objective_after": args.velocity_objective,
                "reward_before": old_reward,
                "reward_after": reward_fn.recipe,
                "other_reward_terms_changed": False,
                "critic_adaptation": "one new-reward rollout, critic updates only; retain critic weights/Adam/calibration",
            }
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    evaluations = {}
    eval_seconds = 0.0
    if not args.smoke:
        for label, teacher in (("pid", True), ("parent", False)):
            if args.baseline_dir:
                source = args.baseline_dir / label
                baseline_training = json.loads((args.baseline_dir / "report.json").read_text())
                if label == "parent":
                    if baseline_training["checkpoint_sha256"] == sha256(args.checkpoint):
                        source = args.baseline_dir / "final"
                    elif baseline_training["parent_checkpoint_sha256"] != sha256(
                        args.checkpoint
                    ):
                        raise ValueError(
                            "Cached parent does not correspond to this training parent"
                        )
                report = json.loads((source / "report.json").read_text())
                if report["physical_contract"] != parent["physical_contract"]:
                    raise ValueError("Cached pre-update evaluation physical contract differs")
                for case in report["cases"]:
                    if sha256(source / case["file"]) != case["sha256"]:
                        raise ValueError("Cached physical capture checksum mismatch")
                shutil.copytree(source, args.output / label)
                report = dict(report, reused_preupdate_capture=True)
            else:
                report = evaluate_hover(
                    actor, parent, args.dataset, args.output / label, device, teacher=teacher
                )
                eval_seconds += report["total_wall_seconds"]
            evaluations[label] = report
            print(json.dumps({"evaluation": label, "cases": report["cases"]}), flush=True)
    memory = actor.initial_state(args.worlds)
    obs = torch.as_tensor(observation(env, commands), device=device)
    counters = {
        "rollouts": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "transitions": 0,
        "imitation_presentations": 0,
        "collection_seconds": 0.0,
        "optimization_seconds": 0.0,
        "imitation_seconds": 0.0,
        "critic_seconds": 0.0,
        "memory_refresh_seconds": 0.0,
    }
    progress, episodes, snapshots = [], [], []
    episode_returns = np.zeros(args.worlds)
    audit = gradient_audit = None
    audit_seconds = checkpoint_seconds = 0.0
    started_utc = utc_now()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    training_start = time.perf_counter()

    def training_seconds():
        return (
            time.perf_counter()
            - training_start
            - audit_seconds
            - checkpoint_seconds
            - during_eval_seconds
        )

    def save(name, measured):
        nonlocal checkpoint_seconds
        begin = time.perf_counter()
        excluded = {
            "state_dict",
            "optimizer_state_dict",
            "sampler_state_json",
            "torch_rng_state",
            "cuda_rng_state",
            "imitation_recipe",
        }
        checkpoint = {k: v for k, v in parent.items() if k not in excluded}
        checkpoint.update(
            state_dict={k: v.detach().cpu() for k, v in actor.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),
            value_optimizer_state_dict=value_optimizer.state_dict(),
            critic_state_dict=critic.state_dict(),
            log_std=log_std.detach().cpu(),
            training_updates=parent["training_updates"] + counters["actor_updates"],
            cumulative_training_seconds=parent["cumulative_training_seconds"] + measured,
            parent_checkpoint_sha256=sha256(args.checkpoint),
            source_commit=provenance["source_commit"],
            method="Velocity hover PPO with light PID imitation anchor",
            ppo_recipe=recipe,
            ppo_counters=dict(counters),
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device).cpu()
            if device.type == "cuda"
            else torch.empty(0),
            sampler_state_json=json.dumps(rng.bit_generator.state),
        )
        torch.save(checkpoint, args.output / name)
        snapshots.append(
            {"file": name, "sha256": sha256(args.output / name), "training_seconds": measured}
        )
        checkpoint_seconds += time.perf_counter() - begin

    during_eval_seconds = 0.0
    midpoint_done = False
    with (args.output / "progress.jsonl").open("x") as log:
        while training_seconds() < args.seconds:
            begin = time.perf_counter()
            buffers = {
                k: []
                for k in (
                    "obs",
                    "latent",
                    "logp",
                    "mean_action",
                    "value",
                    "reward",
                    "done",
                    "features",
                )
            }
            states, term_sum = [], {}
            with torch.no_grad():
                for t in range(args.horizon):
                    if t % args.sequence == 0:
                        states.append(memory.clone())
                    features = critic.features(actor, obs, memory, reward_fn, env)
                    value = critic.network(features).squeeze(-1)
                    output = actor(obs, memory)
                    distribution = motor_distribution(output, active, log_std, 0.0005)
                    latent = distribution.sample()
                    logp = joint_log_probability(
                        output, distribution, latent, output.activity, motor_only=True
                    )
                    env.step(latent.tanh().cpu().numpy())
                    reward, failed, terms = reward_fn()
                    episode_returns += reward
                    timeout = env.ages >= 5000
                    done = failed | timeout
                    next_obs = torch.as_tensor(observation(env, commands), device=device)
                    reward_tensor = torch.as_tensor(reward, device=device)
                    if np.any(timeout & ~failed):
                        terminal_features = critic.features(
                            actor, next_obs, output.state, reward_fn, env
                        )
                        reward_tensor += (
                            gamma
                            * critic.network(terminal_features).squeeze(-1)
                            * torch.as_tensor(timeout & ~failed, device=device)
                        )
                    items = {
                        "obs": obs,
                        "latent": latent,
                        "logp": logp,
                        "mean_action": output.action,
                        "value": value,
                        "reward": reward_tensor,
                        "done": torch.as_tensor(done, device=device),
                        "features": features,
                    }
                    if args.wing_readout_only:
                        items["motor_features"] = motor_features(actor, output.state)
                        buffers.setdefault("motor_features", [])
                    for key, item in items.items():
                        buffers[key].append(item)
                    for key, term in terms.items():
                        term_sum[key] = (
                            term_sum.get(key, 0.0) + float(term.mean()) / args.horizon
                        )
                    ids = np.flatnonzero(done)
                    for i in ids:
                        episodes.append(
                            {
                                "world": int(i),
                                "start_episode": int(world_episode[i]),
                                "seconds": float(env.ages[i] * 0.002),
                                "failed": bool(failed[i]),
                                "return": float(episode_returns[i]),
                            }
                        )
                    memory = actor.reset_worlds(output.state, items["done"])
                    if len(ids):
                        reset(ids)
                        episode_returns[ids] = 0
                        next_obs = torch.as_tensor(observation(env, commands), device=device)
                    obs = next_obs
                final_features = critic.features(actor, obs, memory, reward_fn, env)
                final_value = critic.network(final_features).squeeze(-1)
            data = {k: torch.stack(v) for k, v in buffers.items()}
            old_log_std = log_std.detach().clone()
            del buffers
            adv, returns = advantages(
                data["reward"], data["value"], final_value, data["done"], gamma, gae_lambda
            )
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
            synchronize(device)
            counters["collection_seconds"] += time.perf_counter() - begin
            counters["transitions"] += args.horizon * args.worlds
            if audit is None:
                begin = time.perf_counter()
                audit = replay_audit(actor, data, states, args.sequence, active, log_std)
                if args.wing_readout_only:
                    cached_logp, cached_action, _ = readout_replay(
                        actor, data, 0, args.horizon, active, log_std
                    )
                    audit["cached_max_action_error"] = float(
                        (cached_action - data["mean_action"]).abs().max()
                    )
                    audit["cached_max_logp_error"] = float(
                        (cached_logp - data["logp"]).abs().max()
                    )
                    if (
                        audit["cached_max_action_error"] > 2e-6
                        or audit["cached_max_logp_error"] > 0.01
                    ):
                        raise RuntimeError(f"Cached readout replay mismatch: {audit}")
                    del cached_logp, cached_action
                synchronize(device)
                audit_seconds += time.perf_counter() - begin
                (args.output / "replay_audit.json").write_text(
                    json.dumps(audit, indent=2) + "\n"
                )
                print(json.dumps({"replay_audit": audit}), flush=True)
            begin = time.perf_counter()
            losses, kls, imitation_losses, step_checks = [], [], [], []
            stop = False
            if args.bounded_updates:
                for group in optimizer.param_groups:
                    group["lr"] = args.lr
            # Let the new critic see one rollout before it drives actor updates.
            for _ in range(
                0 if counters["rollouts"] < recipe["critic_warmup_rollouts"] else 2
            ):
                for chunk in rng.permutation(len(states)):
                    a, b = chunk * args.sequence, (chunk + 1) * args.sequence
                    if args.wing_readout_only:
                        new_logp, means_before, _ = readout_replay(
                            actor, data, a, b, active, log_std
                        )
                    else:
                        new_logp, means_before, _ = replay(
                            actor, data, states[chunk].detach(), a, b, active, log_std, True
                        )
                    log_ratio = new_logp - data["logp"][a:b]
                    ratio = log_ratio.exp()
                    kl = ((ratio - 1) - log_ratio).mean()
                    kls.append(float(kl.detach()))
                    analytic_before = (
                        motor_kl(
                            data["mean_action"][a:b],
                            means_before.detach(),
                            old_log_std,
                            log_std,
                        )
                        if args.bounded_updates
                        else 0
                    )
                    if not torch.isfinite(kl) or kl.detach() > 0.02 or analytic_before > 0.015:
                        stop = True
                        break
                    physical_loss = torch.maximum(
                        -adv[a:b] * ratio, -adv[a:b] * ratio.clamp(0.8, 1.2)
                    ).mean()
                    optimizer.zero_grad(set_to_none=True)
                    physical_loss.backward()
                    if gradient_audit is None:
                        gradient_audit = {
                            name: float(p.grad.norm())
                            for name, p in actor.named_parameters()
                            if p.requires_grad and p.grad is not None
                        }
                        if not all(
                            p.grad is not None
                            and torch.isfinite(p.grad).all()
                            and p.grad.norm() > 0
                            for p in (
                                actor.wing_residual.parameters()
                                if args.wing_readout_only
                                else actor.core.parameters()
                            )
                        ):
                            raise RuntimeError(
                                "Physical PPO gradients do not reach the recurrent core"
                            )
                    if not losses:
                        anchor_begin = time.perf_counter()
                        starts_demo = rng.integers(64, 936, size=8)
                        starts_demo[0] = 0
                        indices = starts_demo[None] + np.arange(-64, 64)[:, None]
                        valid = torch.as_tensor(indices >= 0, device=device)
                        ix = torch.as_tensor(indices.clip(0), device=device)
                        wi = torch.arange(8, device=device)[None]
                        if args.wing_readout_only:
                            predicted = readout_action(actor, cached_demo[wi, ix[64:]])
                            target = demo_actions[wi, ix[64:]]
                            anchor = (
                                (predicted[..., wings] - target[..., wings]).square().mean()
                            )
                        else:
                            anchor, _, _, _ = sequence_loss(
                                actor,
                                demo_obs[wi, ix],
                                demo_actions[wi, ix[64:]],
                                valid,
                                64,
                                wings,
                                True,
                            )
                        anchor.backward()
                        imitation_losses.append(float(anchor.detach()))
                        counters["imitation_presentations"] += 512
                        synchronize(device)
                        counters["imitation_seconds"] += time.perf_counter() - anchor_begin
                    nn.utils.clip_grad_norm_(parameters, 1, error_if_nonfinite=True)
                    if args.bounded_updates:

                        def measure(
                            data=data,
                            state=states[chunk],
                            a=a,
                            b=b,
                            old_log_std=old_log_std,
                        ):
                            if args.wing_readout_only:
                                _, means, _ = readout_replay(
                                    actor, data, a, b, active, log_std
                                )
                                return motor_kl(
                                    data["mean_action"][a:b], means, old_log_std, log_std
                                )
                            _, means, _ = replay(
                                actor,
                                data,
                                state,
                                a,
                                b,
                                active,
                                log_std,
                                False,
                            )
                            return motor_kl(
                                data["mean_action"][a:b], means, old_log_std, log_std
                            )

                        check = bounded_step(optimizer, parameters, measure)
                        step_checks.append(check)
                        if not check["accepted"]:
                            stop = True
                            break
                    else:
                        optimizer.step()
                    losses.append(float(physical_loss.detach()))
                    counters["actor_updates"] += 1
                if stop:
                    break
            synchronize(device)
            counters["optimization_seconds"] += time.perf_counter() - begin
            begin = time.perf_counter()
            fitted = fit_critic(
                critic,
                value_optimizer,
                data["features"],
                returns,
                args.sequence,
                4,
                critic_rng,
                shuffle=True,
            )
            prediction = fitted.pop("predictions")
            fitted.pop("losses")
            fitted["explained_variance"] = float(
                1
                - (returns - prediction).var(unbiased=False)
                / returns.var(unbiased=False).clamp_min(1e-8)
            )
            counters["critic_updates"] += fitted["updates"]
            synchronize(device)
            counters["critic_seconds"] += time.perf_counter() - begin
            begin = time.perf_counter()
            if not args.wing_readout_only:
                with torch.no_grad():
                    _, _, memory = replay(
                        actor,
                        data,
                        states[-1],
                        args.horizon - args.sequence,
                        args.horizon,
                        active,
                        log_std,
                        False,
                    )
            synchronize(device)
            counters["memory_refresh_seconds"] += time.perf_counter() - begin
            counters["rollouts"] += 1
            row = {
                "rollout": counters["rollouts"],
                "training_seconds": training_seconds(),
                "transitions": counters["transitions"],
                "actor_updates": counters["actor_updates"],
                "minibatch_updates": len(losses),
                "kl_stop": stop,
                "maximum_kl": max(kls, default=0),
                "mean_policy_loss": float(np.mean(losses)) if losses else None,
                "imitation_loss": imitation_losses,
                "post_update_checks": step_checks,
                "actor_lr": optimizer.param_groups[0]["lr"],
                "reward_rates": term_sum,
                "critic": fitted,
                "completed_episodes": len(episodes),
                "recent_mean_episode_seconds": float(
                    np.mean([e["seconds"] for e in episodes[-32:]])
                )
                if episodes
                else None,
                "live_mean_height_mm": float(env.fields["qpos"][:, 2].mean() * 10),
                "std_mean": float(log_std.detach().exp().mean()),
            }
            progress.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
            del data, states, prediction, returns, adv
            if not midpoint_done and training_seconds() >= args.seconds / 2:
                measured = training_seconds()
                save("midpoint_actor.pt", measured)
                if not args.smoke:
                    begin = time.perf_counter()
                    evaluations["midpoint"] = evaluate_hover(
                        actor, parent, args.dataset, args.output / "midpoint", device
                    )
                    during_eval_seconds += time.perf_counter() - begin
                    print(
                        json.dumps(
                            {
                                "evaluation": "midpoint",
                                "cases": evaluations["midpoint"]["cases"],
                            }
                        ),
                        flush=True,
                    )
                midpoint_done = True
    synchronize(device)
    measured = training_seconds()
    save("final_actor.pt", measured)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    if not args.smoke:
        evaluations["final"] = evaluate_hover(
            actor, parent, args.dataset, args.output / "final", device
        )
        eval_seconds += evaluations["final"]["total_wall_seconds"]
    changes = {
        name: float((p.detach().cpu() - initial[name]).norm())
        for name, p in actor.named_parameters()
    }
    report = {
        "provenance": provenance,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "requested_training_seconds": args.seconds,
        "training_wall_seconds": measured,
        "cumulative_training_seconds": parent["cumulative_training_seconds"] + measured,
        "parent_checkpoint_sha256": sha256(args.checkpoint),
        "checkpoint_sha256": sha256(args.output / "final_actor.pt"),
        "physical_contract": parent["physical_contract"],
        "recipe": recipe,
        "counters": counters,
        "snapshots": snapshots,
        "progress": progress,
        "episodes": episodes,
        "evaluations": evaluations,
        "replay_audit": audit,
        "physical_gradient_audit": gradient_audit,
        "parameter_changes_l2": changes,
        "fixed_graph_sha256": parent["graph_sha256"],
        "peak_cuda_bytes": peak,
        "setup_seconds": setup_seconds,
        "replay_audit_seconds": audit_seconds,
        "checkpoint_seconds": checkpoint_seconds,
        "evaluation_seconds": eval_seconds + during_eval_seconds,
        "physics_steps": counters["transitions"] * 2,
        "aggregate_simulated_seconds": counters["transitions"] * 0.002,
        "smoke_only": args.smoke,
        "training_complete": True,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "completed": True,
                "training_seconds": measured,
                "counters": counters,
                "final_evaluation": evaluations.get("final"),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--horizon", type=int, default=512)
    p.add_argument("--sequence", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--seed", type=int, default=151101)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--baseline-dir", type=Path)
    p.add_argument("--bounded-updates", action="store_true")
    p.add_argument("--wing-readout-only", action="store_true")
    p.add_argument("--resume-ppo", action="store_true")
    p.add_argument("--horizontal-reward-scale", type=float, default=0.5)
    p.add_argument("--vertical-reward-weight", type=float, default=2.0)
    p.add_argument("--adapt-vertical-reward", action="store_true")
    p.add_argument("--velocity-objective", choices=("separate", "vector"), default="separate")
    p.add_argument("--adapt-velocity-objective", action="store_true")
    train(p.parse_args())
