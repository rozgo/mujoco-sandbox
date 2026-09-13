"""Verify the complete graph's neutral sensory extension before optimization."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.brain import EmbodiedBrain, initialize_extended_actor, load_malecns
from embodied_fly.evaluate import load_actor
from embodied_fly.observations import append_wing_velocity, wing_velocity_indices
from embodied_fly.provenance import evidence, sha256, utc_now


@torch.no_grad()
def verify(args):
    if args.output.exists():
        raise FileExistsError("Preserve the previous migration report")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    parent, checkpoint = load_actor(args.checkpoint, args.graph, device)
    if parent.observation_size != 383:
        raise ValueError("Expected the preserved 383-input actor")
    graph, sensory, descending, motor = load_malecns(args.graph)
    child = (
        EmbodiedBrain(graph, sensory, descending, motor, 389, 78, parent.internal_steps, 6)
        .to(device)
        .eval()
    )
    initialize_extended_actor(child, checkpoint["state_dict"])
    model = mujoco.MjModel.from_binary_path(str(args.capture.parent / "model.mjb"))
    with np.load(args.capture) as data:
        observations = data["observation"].copy()
        augmented = append_wing_velocity(
            observations, data["qvel"], wing_velocity_indices(model)
        )
    offsets = np.arange(args.sequences) * args.steps
    if offsets[-1] + args.steps > len(observations):
        raise ValueError("Capture shorter than requested independent sequences")
    reports = []
    for scale in (1.0, 0.1):
        old_memory, new_memory = (
            parent.initial_state(args.sequences),
            child.initial_state(args.sequences),
        )
        maximum = {"action": 0.0, "state": 0.0, "utility_scores": 0.0}
        for t in range(args.steps):
            old = parent(
                torch.as_tensor(observations[t + offsets], device=device),
                old_memory,
                time_scale=scale,
            )
            new = child(
                torch.as_tensor(augmented[t + offsets], device=device),
                new_memory,
                time_scale=scale,
            )
            for field, previous in maximum.items():
                error = float((getattr(old, field) - getattr(new, field)).abs().max())
                maximum[field] = max(previous, error)
            old_memory, new_memory = old.state, new.state
        reports.append(
            {
                "neural_time_scale": scale,
                "maximum_absolute_difference": maximum,
                "exactly_equal": not any(maximum.values()),
            }
        )
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "parent_checkpoint_sha256": sha256(args.checkpoint),
        "capture_sha256": sha256(args.capture),
        "graph_sha256": checkpoint["graph_sha256"],
        "device": str(device),
        "parallel_neural_sequences": args.sequences,
        "steps_per_sequence": args.steps,
        "old_observation_size": 383,
        "new_observation_size": 389,
        "added_trainable_parameters": child.sensor_extension.weight.numel(),
        "live_physics_worlds": 0,
        "optimization_performed": False,
        "results": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if not all(r["exactly_equal"] for r in reports):
        raise RuntimeError("Migrated actor differs before learning; inspect retained report")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("graph", "checkpoint", "capture", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--sequences", type=int, default=8)
    verify(parser.parse_args())
