"""Distinguish recurrent-memory drift from physical feedback distribution shift."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now


@torch.no_grad()
def diagnose(args):
    if args.output.exists():
        raise FileExistsError("Preserve previous diagnostic reports")
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, _ = load_actor(args.checkpoint, args.graph, device)
    with np.load(args.episode, allow_pickle=False) as data:
        observations, targets = data["observation"], data["action"]
        target_activity = data["activity"]
    manifest = json.loads((args.episode.parent / "manifest.json").read_text())
    names = [a["name"] for a in manifest["environment"]["actuation"]]
    errors, predictions, state_std, activities = [], [], [], []
    memory = actor.initial_state(1)
    for observation, target in zip(observations, targets):
        result = actor(torch.as_tensor(observation[None], device=device), memory)
        memory = result.state
        prediction = result.action[0].cpu().numpy()
        predictions.append(prediction)
        errors.append((prediction - target) ** 2)
        state_std.append(float(memory.std()))
        activities.append(int(result.activity[0]))
    errors, predictions = np.array(errors), np.array(predictions)
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "episode_sha256": sha256(args.episode),
        "case": f"teacher-replay {args.episode.stem}",
        "full_sequence_mse": float(errors.mean()),
        "first_32_frames_mse": float(errors[:32].mean()),
        "last_500_frames_mse": float(errors[-500:].mean()),
        "neural_state_std_first_last": [state_std[0], state_std[-1]],
        "activity_accuracy": float(np.mean(np.asarray(activities) == target_activity)),
        "rest_fraction": float(np.mean(np.asarray(activities) == 0)),
        "first_activity_mismatch_frame": next(
            (i for i, (a, b) in enumerate(zip(activities, target_activity)) if a != b), None
        ),
        "groups": {},
    }
    for group in ("wing", "antenna", "labrum", "coxa", "tibia", "femur", "tarsus", "head"):
        indices = [i for i, name in enumerate(names) if group in name]
        report["groups"][group] = {
            "mse": float(errors[:, indices].mean()),
            "prediction_rms": float(np.sqrt((predictions[:, indices] ** 2).mean())),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    diagnose(parser.parse_args())
