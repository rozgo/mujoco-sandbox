"""Collect actual full-body actor histories for local model-guided learning."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.full_body_decoder import motor_features
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment, pre_row
from embodied_fly.velocity_hover import failures, start_states
from embodied_fly.velocity_motor import observation
from embodied_fly.world_data import (
    layout,
    physical_features,
    physical_metrics,
    split_for_episode,
)


@torch.no_grad()
def run(args):
    begin = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    if actor.wing_residual is not None or not actor.shared_decoder_hidden:
        raise ValueError("Only the shared full-body actor may supply this experience")
    episodes = tuple(range(8))  # Existing split: five training, three validation.
    env = environment(len(episodes), len(episodes))
    if physical_contract(env.model) != checkpoint["physical_contract"]:
        raise ValueError("Canonical physics mismatch")
    env.reset(np.arange(env.n), state=start_states(args.dataset, episodes))
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    indices = layout(env.model)
    commands = np.zeros((env.n, 4), np.float32)
    state = actor.initial_state(env.n)
    actor.eval()
    steps = round(args.seconds * 500)
    arrays = {
        "motor": np.empty((env.n, steps, len(actor.motor_ids)), np.float32),
        "actions": np.empty((env.n, steps, actor.action_size), np.float32),
        "features": np.empty(
            (env.n, steps, env.model.nq - 3 + env.model.nv + env.model.na + 1), np.float32
        ),
        "metrics": np.empty((env.n, steps, 30), np.float32),
        "qpos": np.empty((env.n, steps, env.model.nq), np.float64),
    }
    first_failure = np.full(env.n, np.nan)
    synchronization_error = 0.0
    setup = time.perf_counter() - begin
    synchronize(device)
    start = time.perf_counter()
    for step in range(steps):
        row = pre_row(env, commands, step * 0.002, 0, refresh=True)
        result = actor(torch.as_tensor(observation(env, commands), device=device), state)
        state = result.state
        motor = motor_features(actor, state)
        action = result.action.cpu().numpy()
        if step % 500 == 0:
            synchronization_error = max(
                synchronization_error,
                float((actor.motor_decoder(motor) - result.action).abs().max()),
            )
        arrays["motor"][:, step] = motor.cpu().numpy()
        arrays["actions"][:, step] = action
        arrays["features"][:, step] = physical_features(row["qpos"], row["qvel"], row["act"])
        arrays["metrics"][:, step] = physical_metrics(row["qpos"], row["qvel"], indices)
        arrays["qpos"][:, step] = row["qpos"]
        env.step(action)
        new = failures(env) & np.isnan(first_failure)
        first_failure[new] = (step + 1) * 0.002
    synchronize(device)
    capture = time.perf_counter() - start
    start = time.perf_counter()
    records = []
    for i, episode in enumerate(episodes):
        file = args.output / f"episode_{episode:02d}.npz"
        stop = (
            steps if np.isnan(first_failure[i]) else max(0, round(first_failure[i] * 500) - 1)
        )
        np.savez_compressed(file, **{k: v[i, :stop] for k, v in arrays.items()})
        records.append(
            {
                "episode": episode,
                "split": split_for_episode(episode),
                "file": file.name,
                "sha256": sha256(file),
                "samples": stop,
                "first_failure_seconds": None
                if np.isnan(first_failure[i])
                else float(first_failure[i]),
            }
        )
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "physical_contract": checkpoint["physical_contract"],
        "feature_layer": "raw motor-neuron latent state after current brain tick",
        "maximum_cached_action_reconstruction_error": synchronization_error,
        "training_episodes": [0, 2, 3, 4, 5],
        "validation_episodes": [1, 6, 7],
        "withheld_physical_evaluation_episodes": [8, 9],
        "records": records,
        "worlds": env.n,
        "physics_threads": env.n,
        "physics_hz": 1000,
        "control_hz": 500,
        "observation_timing": "refresh before actor",
        "commands": [0, 0, 0, 0],
        "captured_transitions": steps * env.n,
        "simulated_seconds": args.seconds * env.n,
        "setup_seconds": setup,
        "capture_seconds": capture,
        "serialization_seconds": time.perf_counter() - start,
        "total_wall_seconds": time.perf_counter() - begin,
        "pid_control_share": 0,
        "policy_updates": 0,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
