"""Frozen mean-versus-sampled hover and full-graph recurrent replay audit.

No optimizer or teacher. Independent worlds share one frozen actor and the
accepted physical model. Recurrent replay uses recorded observations/actions;
it does not advance or overwrite physical worlds.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.hover_only import HoverBalancedReward, HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo import joint_log_probability, motor_distribution
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def audit(args):
    if args.worlds < 2 or args.worlds % 2 or args.seconds <= 0:
        raise ValueError("An even world count and positive capture duration are required")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance, setup_start = evidence(), time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    env = FlyBatch(
        args.worlds,
        args.threads,
        actor.sensor_extension_size,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
    )
    assert physical_contract(env.model) == parent["physical_contract"]
    tasks = HoverOnlyTasks(env, args.seed)
    reward_fn = HoverBalancedReward(env)
    state = actor.initial_state(args.worlds)
    active = torch.ones(actor.action_size, dtype=torch.bool, device=device)
    log_std = parent["log_std"].to(device)
    floor = parent["config"]["minimum_noise"]
    sampled = np.arange(args.worlds) % 2 == 1
    first_failure = np.full(args.worlds, np.nan)
    returns = np.zeros(args.worlds)
    rows, replay = [], []
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    start = time.perf_counter()
    for step in range(round(args.seconds / env.control_dt)):
        observation = torch.as_tensor(env.observation(), device=device)
        with torch.no_grad():
            output = actor(observation, state, sample_activity=True)
            distribution = motor_distribution(output, active, log_std, floor)
            latent = distribution.sample()
            logp = joint_log_probability(
                output, distribution, latent, output.activity, motor_only=True
            )
            command = output.action.cpu().numpy().copy()
            command[sampled] = latent.tanh().cpu().numpy()[sampled]
            if step < 128:
                replay.append(
                    (
                        observation.detach(),
                        latent.detach(),
                        output.activity.detach(),
                        logp.detach(),
                    )
                )
            state = output.state
        previous = env.previous_action.copy()
        env.step(command)
        reward, failed, _ = reward_fn(previous)
        time_now = (step + 1) * env.control_dt
        first_failure[np.isnan(first_failure) & failed] = time_now
        # Stop accumulating each world's return at its first physical failure.
        # Continue motion recording without resets, retaining the failed path.
        returns += np.where(np.isnan(first_failure) | (first_failure == time_now), reward, 0)
        rows.append(
            {
                "position_cm": env.fields["qpos"][:, :3].copy(),
                "velocity_cm_s": env.fields["qvel"][:, :3].copy(),
                "action": command.copy(),
                "mean_action": output.action.cpu().numpy().copy(),
                "lift_bodyweights": env.wing_forces.lift.copy() / env.body_weight,
                "forbidden_bodyweights": env.forbidden_peak.copy() / env.body_weight,
            }
        )
    synchronize(device)
    capture_seconds = time.perf_counter() - start
    arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
    np.savez_compressed(args.output / "capture.npz", **arrays)
    results = []
    for i in range(args.worlds):
        pos = arrays["position_cm"][:, i]
        retained = np.arange(len(pos)) * env.control_dt >= 1
        result = {
            "world": i,
            "control": "sampled" if sampled[i] else "mean",
            "first_failure_seconds": None
            if np.isnan(first_failure[i])
            else float(first_failure[i]),
            "return_until_first_failure_or_end": float(returns[i]),
            "position_peak_mm": float(np.linalg.norm(pos - tasks.start[i], axis=1).max() * 10),
            "height_span_after_one_second_mm": float(np.ptp(pos[retained, 2]) * 10)
            if retained.any()
            else None,
            "mean_lift_bodyweights": float(arrays["lift_bodyweights"][:, i].mean()),
            "applied_noise_rms": float(
                np.sqrt(np.mean((arrays["action"][:, i] - arrays["mean_action"][:, i]) ** 2))
            ),
        }
        results.append(result)
    # Compare the likelihood of the SAME samples under the unchanged full graph.
    # Enable gradient recording, as training does, but perform no backward/update.
    replay_start = time.perf_counter()
    memory = actor.initial_state(args.worlds)
    differences = []
    for observation, latent, activity, old in replay:
        result = actor(observation, memory, activity_override=activity)
        dist = motor_distribution(result, active, log_std, floor)
        new = joint_log_probability(result, dist, latent, activity, motor_only=True)
        differences.append((new - old).detach())
        # Forward state values continue; detaching saves memory for this no-update
        # numeric audit. This does not test long-sequence gradients.
        memory = result.state.detach()
    delta = torch.stack(differences)
    synchronize(device)
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "capture_sha256": sha256(args.output / "capture.npz"),
        "config": {
            "worlds": args.worlds,
            "threads": args.threads,
            "seed": args.seed,
            "seconds": args.seconds,
            "noise_floor": floor,
        },
        "physical_contract": physical_contract(env.model),
        "setup_seconds": setup_seconds,
        "capture_seconds": capture_seconds,
        "physical_transitions": len(rows) * args.worlds,
        "optimizer_updates": 0,
        "physical_resets_during_capture": 0,
        "same_actor_all_worlds": True,
        "results": results,
        "replay": {
            "seconds": time.perf_counter() - replay_start,
            "actions_per_world": len(replay),
            "maximum_absolute_log_probability_difference": float(delta.abs().max()),
            "mean_approximate_kl": float((delta.exp() - 1 - delta).mean()),
            "scope": "Frozen recurrent forward values with gradients enabled; no backward or optimization",
        },
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("capture_seconds", "replay", "results")}))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--seed", type=int, default=120201)
    parser.add_argument("--device", default="cuda")
    audit(parser.parse_args())
