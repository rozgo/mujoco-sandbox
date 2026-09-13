"""Causal frozen-actor replay for fitting the existing nonlinear motor decoder.

Features are actual motor-cell states, never raw observations or teacher state.
Variable-length episodes retain their entire history, including stop/resume.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.body import CONTROL_DT
from embodied_fly.evaluate import load_actor
from embodied_fly.observations import (
    append_wing_angles,
    append_wing_velocity,
    wing_angle_indices,
    wing_velocity_indices,
)
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def episode_split(ids, seed=1193):
    ids = np.asarray(ids, dtype=np.int64)
    if len(ids) < 4 or len(np.unique(ids)) != len(ids):
        raise ValueError("Need at least four distinct complete episodes")
    validation = np.random.default_rng(seed).permutation(ids)[: max(1, len(ids) // 4)]
    return np.array([i for i in ids if i not in validation]), np.sort(validation)


def replay_group(actor, episodes, device, time_scale):
    """Each column has independent persistent memory; padding is never recorded."""
    memory = actor.initial_state(len(episodes))
    lengths = np.array([len(e["observation"]) for e in episodes])
    for frame in range(int(lengths.max())):
        observation = np.stack(
            [e["observation"][min(frame, len(e["observation"]) - 1)] for e in episodes]
        )
        result = actor(
            torch.as_tensor(observation, device=device), memory, time_scale=time_scale
        )
        memory = result.state
        active = np.flatnonzero(frame < lengths)
        yield (
            active,
            frame,
            memory[actor.motor_ids].T[active].cpu().numpy().copy(),
            result.action[active].cpu().numpy().copy(),
        )


@torch.no_grad()
def collect(args):
    if args.worlds < 1:
        raise ValueError("Positive replay batch size required")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    if actor.sensor_extension_size != 12:
        raise ValueError("Expected the declared 395-input actor")
    manifest = json.loads((args.data / "manifest.json").read_text())
    wing_motion = manifest.get("physical_preset") == "wing_motion"
    if wing_motion and args.role != "flight":
        raise ValueError("Wing-motion corpus is a flight task")
    if manifest["control_hz"] != (5000 if args.role == "flight" and not wing_motion else 500):
        raise ValueError("Corpus role and physical observation clock disagree")
    if sha256(args.data / "model.mjb") != manifest["model_sha256"]:
        raise ValueError("Capture model hash mismatch")
    model = mujoco.MjModel.from_binary_path(str(args.data / "model.mjb"))
    wings = np.array([i for i in range(model.nu) if "wing_" in model.actuator(i).name])
    episodes, rejected = [], []
    for e in manifest["episodes"]:
        path = args.data / f"episode_{e['episode']:03d}.npz"
        if sha256(path) != e["state_sha256"]:
            raise ValueError("Capture state hash mismatch")
        if e["failure"] or e.get("physical_failure") or e["final_upright"] < 0.5:
            rejected.append(e["episode"])
            continue
        with np.load(path, allow_pickle=False) as c:
            if c["observation"].shape[-1] != 383:
                raise ValueError("Expected legacy capture with measured wing extension")
            observation = append_wing_velocity(
                c["observation"], c["qvel"], wing_velocity_indices(model)
            )
            observation = append_wing_angles(observation, c["qpos"], wing_angle_indices(model))
            if not np.isfinite(observation).all() or not np.isfinite(c["action"]).all():
                raise ValueError("Nonfinite replay data")
            episodes.append(
                {"id": e["episode"], "observation": observation, "action": c["action"].copy()}
            )
    training, validation = episode_split([e["id"] for e in episodes])
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    replay_started = time.perf_counter()
    rows = {k: [] for k in ("motor", "parent_action", "teacher_action", "episode", "frame")}
    for offset in range(0, len(episodes), args.worlds):
        group = episodes[offset : offset + args.worlds]
        for active, frame, motor, parent in replay_group(
            actor, group, device, 1 / manifest["control_hz"] / CONTROL_DT
        ):
            rows["motor"].append(motor)
            rows["parent_action"].append(parent)
            rows["teacher_action"].append(
                np.stack([group[i]["action"][frame] for i in active])
            )
            rows["episode"].append(np.array([group[i]["id"] for i in active], np.int64))
            rows["frame"].append(np.full(len(active), frame, np.int64))
    synchronize(device)
    replay_seconds = time.perf_counter() - replay_started
    saving = time.perf_counter()
    packed = {k: np.concatenate(v) for k, v in rows.items()}
    np.savez_compressed(
        args.output / "features.npz",
        **packed,
        training_episodes=training,
        validation_episodes=validation,
        wing_channels=wings,
    )
    report = {
        "schema": "frozen-motor-cell-features-v1",
        "provenance": provenance,
        "completed_utc": utc_now(),
        "role": args.role,
        "checkpoint_sha256": sha256(args.checkpoint),
        "graph_sha256": checkpoint["graph_sha256"],
        "data_manifest_sha256": sha256(args.data / "manifest.json"),
        "model_sha256": manifest["model_sha256"],
        "cache_sha256": sha256(args.output / "features.npz"),
        "control_hz": manifest["control_hz"],
        "physical_preset": manifest.get(
            "physical_preset", "flight" if args.role == "flight" else "walking"
        ),
        "neural_device": str(device),
        "parallel_neural_sequences": min(args.worlds, len(episodes)),
        "episode_lengths": {str(e["id"]): len(e["action"]) for e in episodes},
        "rejected_episodes": rejected,
        "training_episode_ids": training.tolist(),
        "validation_episode_ids": validation.tolist(),
        "frames": len(packed["episode"]),
        "motor_features": packed["motor"].shape[1],
        "physical_transitions_collected": 0,
        "new_deployed_parameters": 0,
        "feature_source": "Actual motor-cell states after the frozen graph actor's causal update",
        "parent_action_source": "Same frozen actor's full 78 outputs on those histories",
        "teacher_action_source": "Captured labels; never supplied to the actor or feature vector",
        "limitations": "Offline replay uses actual recorded previous actions. No autonomous stability or biological causality claim.",
        "setup_seconds": setup_seconds,
        "frozen_actor_replay_seconds": replay_seconds,
        "packing_and_save_seconds": time.perf_counter() - saving,
        "total_wall_seconds": time.perf_counter() - started,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("checkpoint", "graph", "data", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--role", choices=("flight", "ground"), required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    collect(parser.parse_args())
