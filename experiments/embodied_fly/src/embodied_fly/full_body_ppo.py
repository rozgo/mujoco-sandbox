"""Final bounded PPO attempt through one full-body decoder and live MaleCNS.

Cached raw neural features reproduce conditional action likelihoods. No future
body/neural trajectory is held fixed in a differentiable physical objective.
"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.full_body_decoder import (
    decode_features,
    freeze_for_decoder_training,
    motor_features,
)
from embodied_fly.full_body_guidance_evaluate import promotion, summarize
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo import advantages, joint_log_probability, motor_distribution
from embodied_fly.ppo_critic import fit_critic
from embodied_fly.ppo_step import bounded_step, motor_kl
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import HoverReward, evaluate_hover, start_states
from embodied_fly.velocity_hover_ppo import HoverCritic, cached_replay_errors, replay_audit
from embodied_fly.velocity_motor import observation


def conditional_replay(actor, raw_motor, latent, active, log_std):
    shape = raw_motor.shape[:-1]
    actions = decode_features(actor, raw_motor.reshape(-1, raw_motor.shape[-1]))
    output = SimpleNamespace(action=actions)
    dist = motor_distribution(output, active, log_std, 0.0005)
    logp = joint_log_probability(output, dist, latent.reshape(-1, 78), None, motor_only=True)
    return logp.reshape(shape), actions.reshape(*shape, 78)


def gates(summary, baseline):
    return promotion(summary, baseline)


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    source = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng, critic_rng = (
        np.random.default_rng(args.seed),
        np.random.default_rng(args.seed ^ 0xC8171C),
    )
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    freeze_for_decoder_training(actor)
    actor.train()
    initial = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    reduced = (
        parent["physical_contract"].get("force_law_version")
        == "wing_motion_reduced_coupling_v6"
    )
    env = environment(32, 16, reduced_coupling=reduced)
    if physical_contract(env.model) != parent["physical_contract"]:
        raise ValueError("Training plant does not match checkpoint")
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    baseline_report = json.loads((args.baseline / "report.json").read_text())
    if baseline_report["physical_contract"] != parent["physical_contract"]:
        raise ValueError("Learning comparison must use the same physical plant")
    baseline = summarize(baseline_report)
    original_report = json.loads((args.original / "report.json").read_text())
    original = summarize(original_report)
    for folder, report in ((args.baseline, baseline_report), (args.original, original_report)):
        for case in report["cases"]:
            if sha256(folder / case["file"]) != case["sha256"]:
                raise ValueError("Changed baseline capture")
    starts = start_states(args.dataset, range(8))
    worlds = np.arange(32) % 8
    commands = np.zeros((32, 4), np.float32)
    reward_fn = HoverReward(env, 0.5, 2.0, "vector")
    critic = HoverCritic(actor).to(device)
    parameters = list(actor.motor_decoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=3e-6, eps=1e-5)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4, eps=1e-5)
    active = torch.ones(78, dtype=torch.bool, device=device)
    log_std = torch.full((78,), math.log(0.0015), device=device)
    gamma, lam = math.exp(-0.002 / 2), math.exp(-0.002 / 0.25)

    def reset(ids):
        env.reset(ids, state={k: v[worlds[ids]] for k, v in starts.items()})
        reward_fn.reset(ids)

    reset(np.arange(32))
    memory = actor.initial_state(32)
    env.batch.forward()
    obs = torch.as_tensor(observation(env, commands), device=device)
    recipe = {
        "method": "On-policy PPO, one shared full-body decoder",
        "worlds": 32,
        "physics_threads": 16,
        "physics_hz": 1000,
        "control_hz": 500,
        "physics_backend": "CPU MuJoCo/mjbatch",
        "learning_device": str(device),
        "horizon": 512,
        "sequence": 64,
        "actor_epochs": 2,
        "critic_epochs": 4,
        "lr": 3e-6,
        "critic_lr": 3e-4,
        "target_kl": 0.005,
        "clip": 0.2,
        "post_step_kl_scope": "entire collected rollout, all 78 channels",
        "gamma": gamma,
        "gae_lambda": lam,
        "std": 0.0015,
        "reward": reward_fn.recipe,
        "training_cap_seconds": args.seconds,
        "fresh_actor_optimizer": True,
        "fresh_critic_and_optimizer": True,
        "critic_warmup_rollouts": 1,
        "trainable_actor_parameters": sum(p.numel() for p in parameters),
        "trainable_scope": "all shared decoder parameters, including all 78 rows and normalization",
        "upstream_frozen": True,
        "teacher_control_share": 0,
        "imitation_loss": False,
        "body_part_masks": False,
        "force_or_action_filters_added": False,
        "initialization": "cold starts from episodes 0..7; no warm recovery bank",
        "observation_timing": "refresh before actor, matching physical evaluation",
        "conditional_replay": "raw motor neurons; upstream weights frozen; new live experience each rollout",
        "source_checkpoint_sha256": sha256(args.checkpoint),
        "physical_contract": parent["physical_contract"],
    }
    (args.output / "recipe.json").write_text(json.dumps(recipe, indent=2) + "\n")
    counters = {
        "rollouts": 0,
        "transitions": 0,
        "actor_updates": 0,
        "critic_updates": 0,
        "collection_seconds": 0.0,
        "actor_seconds": 0.0,
        "critic_seconds": 0.0,
    }
    snapshots, episodes, evaluations = [], [], []
    episode_returns = np.zeros(32)
    setup_seconds = time.perf_counter() - setup_start
    excluded_seconds = 0.0
    began = time.perf_counter()
    audit = None
    stop_reason = "training budget complete"
    checkpoints_at = [t for t in (300.0, 600.0, 1200.0) if t < args.seconds] + [args.seconds]
    if args.smoke:
        checkpoints_at = [args.seconds]

    def elapsed():
        synchronize(device)
        return time.perf_counter() - began - excluded_seconds

    def save(label, measured):
        excluded = {
            "state_dict",
            "optimizer_state_dict",
            "value_optimizer_state_dict",
            "critic_state_dict",
            "sampler_state_json",
            "torch_rng_state",
            "cuda_rng_state",
            "imitation_recipe",
        }
        saved = {k: v for k, v in parent.items() if k not in excluded}
        saved.update(
            state_dict={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
            optimizer_state_dict=optimizer.state_dict(),
            value_optimizer_state_dict=value_optimizer.state_dict(),
            critic_state_dict=critic.state_dict(),
            log_std=log_std.cpu(),
            parent_checkpoint_sha256=sha256(args.checkpoint),
            source_commit=source["source_commit"],
            method=recipe["method"],
            ppo_recipe=recipe,
            ppo_counters=dict(counters),
            training_updates=parent.get("training_updates", 0) + counters["actor_updates"],
            cumulative_training_seconds=parent.get("cumulative_training_seconds", 0)
            + measured,
            torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(device).cpu()
            if device.type == "cuda"
            else torch.empty(0),
            sampler_state_json=json.dumps(rng.bit_generator.state),
            critic_sampler_state_json=json.dumps(critic_rng.bit_generator.state),
        )
        path = args.output / f"{label}_actor.pt"
        torch.save(saved, path)
        row = {
            "label": label,
            "checkpoint": path.as_posix(),
            "sha256": sha256(path),
            "training_seconds": measured,
        }
        snapshots.append(row)
        return row, saved

    with (args.output / "progress.jsonl").open("x") as log:
        while elapsed() < args.seconds:
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
                    "motor",
                )
            }
            states, terms_mean = [], {}
            with torch.no_grad():
                for t in range(512):
                    if t % 64 == 0:
                        states.append(memory.clone())
                    features = critic.features(actor, obs, memory, reward_fn, env)
                    value = critic.network(features).squeeze(-1)
                    output = actor(obs, memory)
                    dist = motor_distribution(output, active, log_std, 0.0005)
                    latent = dist.sample()
                    logp = joint_log_probability(output, dist, latent, None, motor_only=True)
                    env.step(latent.tanh().cpu().numpy())
                    env.batch.forward()
                    rewards, failed, terms = reward_fn()
                    episode_returns += rewards
                    timeout = env.ages >= 5000
                    done = failed | timeout
                    next_obs = torch.as_tensor(observation(env, commands), device=device)
                    rew = torch.as_tensor(rewards, device=device)
                    if np.any(timeout & ~failed):
                        terminal = critic.features(
                            actor, next_obs, output.state, reward_fn, env
                        )
                        rew += (
                            gamma
                            * critic.network(terminal).squeeze(-1)
                            * torch.as_tensor(timeout & ~failed, device=device)
                        )
                    items = {
                        "obs": obs,
                        "latent": latent,
                        "logp": logp,
                        "mean_action": output.action,
                        "value": value,
                        "reward": rew,
                        "done": torch.as_tensor(done, device=device),
                        "features": features,
                        "motor": motor_features(actor, output.state),
                    }
                    for key, item in items.items():
                        buffers[key].append(item)
                    for key, term in terms.items():
                        terms_mean[key] = terms_mean.get(key, 0.0) + float(term.mean()) / 512
                    ids = np.flatnonzero(done)
                    for i in ids:
                        episodes.append(
                            {
                                "world": int(i),
                                "episode": int(worlds[i]),
                                "seconds": float(env.ages[i] * 0.002),
                                "failed": bool(failed[i]),
                                "total_reward": float(episode_returns[i]),
                            }
                        )
                    memory = actor.reset_worlds(output.state, items["done"])
                    if len(ids):
                        reset(ids)
                        episode_returns[ids] = 0
                        env.batch.forward()
                        next_obs = torch.as_tensor(observation(env, commands), device=device)
                    obs = next_obs
                final_value = critic.network(
                    critic.features(actor, obs, memory, reward_fn, env)
                ).squeeze(-1)
            data = {k: torch.stack(v) for k, v in buffers.items()}
            del buffers
            adv, returns = advantages(
                data["reward"], data["value"], final_value, data["done"], gamma, lam
            )
            adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
            synchronize(device)
            counters["collection_seconds"] += time.perf_counter() - begin
            counters["transitions"] += 512 * 32
            if not torch.isfinite(data["obs"]).all() or not torch.isfinite(returns).all():
                raise RuntimeError("Nonfinite live collection")
            if audit is None:
                begin = time.perf_counter()
                audit = replay_audit(actor, data, states, 64, active, log_std)
                with torch.no_grad():
                    lp, act = conditional_replay(
                        actor, data["motor"], data["latent"], active, log_std
                    )
                    audit.update(
                        cached_replay_errors(act, data["mean_action"], lp, data["logp"])
                    )
                (args.output / "replay_audit.json").write_text(
                    json.dumps(audit, indent=2) + "\n"
                )
                if (
                    audit["cached_max_action_error"] > 2e-6
                    or audit["cached_max_logp_error"] > 0.02
                    or audit["cached_roundoff_aggregate_kl"] > 4e-6
                ):
                    raise RuntimeError(f"Full-body conditional replay mismatch: {audit}")
                excluded_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            checks, losses = [], []
            for group in optimizer.param_groups:
                group["lr"] = 3e-6

            @torch.no_grad()
            def all_kl(data=data):
                acts = decode_features(actor, data["motor"].flatten(0, 1)).reshape(-1, 32, 78)
                return motor_kl(data["mean_action"], acts, log_std, log_std)

            stop_update = False
            for _ in range(0 if counters["rollouts"] == 0 else 2):
                for chunk in rng.permutation(8):
                    a, b = int(chunk) * 64, (int(chunk) + 1) * 64
                    new_lp, _ = conditional_replay(
                        actor, data["motor"][a:b], data["latent"][a:b], active, log_std
                    )
                    log_ratio = new_lp - data["logp"][a:b]
                    ratio = log_ratio.exp()
                    if all_kl() > 0.00375:
                        stop_update = True
                        break
                    loss = torch.maximum(
                        -adv[a:b] * ratio, -adv[a:b] * ratio.clamp(0.8, 1.2)
                    ).mean()
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                    check = bounded_step(optimizer, parameters, all_kl, limit=0.005)
                    checks.append(check)
                    if not check["accepted"]:
                        stop_update = True
                        break
                    counters["actor_updates"] += 1
                    losses.append(float(loss.detach()))
                if stop_update:
                    break
            synchronize(device)
            counters["actor_seconds"] += time.perf_counter() - begin
            begin = time.perf_counter()
            fitted = fit_critic(
                critic,
                value_optimizer,
                data["features"],
                returns,
                64,
                4,
                critic_rng,
                shuffle=True,
            )
            predicted = fitted.pop("predictions")
            fitted.pop("losses")
            fitted["explained_variance"] = float(
                1
                - (returns - predicted).var(unbiased=False)
                / returns.var(unbiased=False).clamp_min(1e-8)
            )
            counters["critic_updates"] += fitted["updates"]
            synchronize(device)
            counters["critic_seconds"] += time.perf_counter() - begin
            counters["rollouts"] += 1
            row = dict(counters) | {
                "training_seconds": elapsed(),
                "reward_rates": terms_mean,
                "policy_loss": float(np.mean(losses)) if losses else None,
                "steps": checks,
                "critic": fitted,
                "completed_episodes": len(episodes),
                "recent_mean_episode_seconds": float(
                    np.mean([e["seconds"] for e in episodes[-32:]])
                )
                if episodes
                else None,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            if counters["rollouts"] % 5 == 0:
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "rollouts",
                                "transitions",
                                "training_seconds",
                                "actor_updates",
                                "recent_mean_episode_seconds",
                            )
                        }
                    ),
                    flush=True,
                )
            if checkpoints_at and elapsed() >= checkpoints_at[0]:
                at = checkpoints_at.pop(0)
                measured = elapsed()
                begin = time.perf_counter()
                label = f"step_{round(at):04d}"
                snapshot, saved = save(label, measured)
                if not args.smoke:
                    report = evaluate_hover(
                        actor, saved, args.dataset, args.output / label, device
                    )
                    summary = summarize(report)
                    learning_gates, original_gates = (
                        gates(summary, baseline),
                        gates(summary, original),
                    )
                    ev = dict(
                        snapshot,
                        capture=(args.output / label).as_posix(),
                        summary=summary,
                        versus_transfer=learning_gates,
                        versus_original=original_gates,
                        passed=all(learning_gates.values()) and all(original_gates.values()),
                    )
                    evaluations.append(ev)
                    print(json.dumps({"physical_evaluation": ev}), flush=True)
                    (args.output / "evaluations.json").write_text(
                        json.dumps(evaluations, indent=2) + "\n"
                    )
                    if (
                        summary["complete"] == 0
                        or summary["velocity_rms_mm_s"] > 1.25 * baseline["velocity_rms_mm_s"]
                    ):
                        stop_reason = "predeclared physical regression stop"
                excluded_seconds += time.perf_counter() - begin
                if stop_reason != "training budget complete":
                    break
            del data, states, predicted
        measured = elapsed()
        if not snapshots or snapshots[-1]["training_seconds"] + 0.5 < measured:
            save("final", measured)
    changed = {
        k: not torch.equal(initial[k], v.detach().cpu()) for k, v in actor.state_dict().items()
    }
    if any(v for k, v in changed.items() if not k.startswith("motor_decoder.")):
        raise RuntimeError("Frozen upstream actor changed")
    output_rows = int(
        (actor.motor_decoder[3].weight.detach().cpu() != initial["motor_decoder.3.weight"])
        .any(1)
        .sum()
    )
    eligible = [e for e in evaluations if e["passed"]]
    report = {
        "provenance": source,
        "completed_utc": utc_now(),
        "stop_reason": stop_reason,
        "training_seconds": measured,
        "setup_seconds": setup_seconds,
        "excluded_audit_evaluation_io_seconds": excluded_seconds,
        "counters": counters,
        "recipe": recipe,
        "evaluations": evaluations,
        "snapshots": snapshots,
        "selected": min(eligible, key=lambda e: e["summary"]["velocity_rms_mm_s"])
        if eligible
        else None,
        "baseline": baseline,
        "original": original,
        "upstream_exactly_unchanged": True,
        "output_rows_changed": output_rows,
        "changed_tensors": changed,
        "peak_cuda_memory_allocated": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
    }
    (args.output / "episodes.json").write_text(json.dumps(episodes, indent=2) + "\n")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "completed": report["stop_reason"],
                "training_seconds": measured,
                "selected": report["selected"],
                "output_rows_changed": output_rows,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "graph", "dataset", "baseline", "original", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--seconds", type=float, default=1200)
    p.add_argument("--seed", type=int, default=415001)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke", action="store_true")
    run(p.parse_args())
