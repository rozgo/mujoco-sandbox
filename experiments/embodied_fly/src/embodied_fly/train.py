"""Timed recurrent behavioral cloning pilot; internal-core learning is audited.

Supervised warm start, not PPO. Checkpoints include learned utility/motor interfaces
and internal neural parameters. No success claim follows from imitation loss alone.
"""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from embodied_fly.body import CONTROL_DT
from embodied_fly.brain import EmbodiedBrain, initialize_extended_actor, load_malecns
from embodied_fly.observations import (
    append_height,
    append_wing_angles,
    append_wing_velocity,
    wing_angle_indices,
    wing_velocity_indices,
)
from embodied_fly.provenance import evidence, sha256, utc_now


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_episodes(
    path, wing_velocity_inputs=False, wing_angle_inputs=False, height_inputs=False
):
    if height_inputs and not wing_angle_inputs:
        raise ValueError("Height inputs require complete wing feedback")
    if wing_angle_inputs and not wing_velocity_inputs:
        raise ValueError("Wing angles require velocity extension in the observation schema")
    manifest = json.loads((path / "manifest.json").read_text())
    rejected = {
        e["episode"]
        for e in manifest["episodes"]
        if e["failure"] or e.get("physical_failure") or e["final_upright"] < 0.5
    }
    episodes = []
    model = (
        mujoco.MjModel.from_binary_path(str(path / "model.mjb"))
        if wing_velocity_inputs
        else None
    )
    indices = wing_velocity_indices(model) if model is not None else None
    angle_indices = wing_angle_indices(model) if wing_angle_inputs else None
    for file in sorted(path.glob("episode_*.npz")):
        if int(file.stem.split("_")[-1]) in rejected:
            continue
        with np.load(file, allow_pickle=False) as data:
            episode = {key: data[key].copy() for key in ("observation", "action", "activity")}
            if wing_velocity_inputs:
                if episode["observation"].shape[-1] != 383:
                    raise ValueError(
                        "Augmentation requires the declared 383-input legacy data"
                    )
                episode["observation"] = append_wing_velocity(
                    episode["observation"], data["qvel"], indices
                )
            if wing_angle_inputs:
                episode["observation"] = append_wing_angles(
                    episode["observation"], data["qpos"], angle_indices
                )
            if height_inputs:
                if "requested_height_cm" in data:
                    requested = data["requested_height_cm"]
                elif manifest.get("physical_preset") == "wing_motion":
                    item = next(
                        e
                        for e in manifest["episodes"]
                        if e["episode"] == int(file.stem.split("_")[-1])
                    )
                    # Legacy corpus records the declared target in its reference report.
                    reference = json.loads((path / item["report"]).read_text())
                    requested = reference["initial_height_cm"]
                elif manifest.get("role") == "flight":
                    raise ValueError("Flight corpus must declare its requested height")
                else:
                    requested = 0.0  # explicit ground command
                episode["observation"] = append_height(
                    episode["observation"], data["qpos"], requested
                )
            episodes.append(episode)
    if len(episodes) < 4:
        raise ValueError("Need at least four separate episodes for train/validation split")
    return episodes


def sample(episodes, rng, length, worlds, device, reset_start=False):
    choices = rng.integers(len(episodes), size=worlds)
    sequences = []
    for choice in choices:
        episode = episodes[choice]
        if len(episode["action"]) < length:
            raise ValueError("Episode shorter than recurrent context")
        start = 0 if reset_start else rng.integers(len(episode["action"]) - length + 1)
        sequences.append(
            {key: value[start : start + length] for key, value in episode.items()}
        )
    return {
        key: torch.as_tensor(
            np.stack([s[key] for s in sequences], axis=1),
            device=device,
            dtype=torch.long if key == "activity" else torch.float32,
        )
        for key in sequences[0]
    }


def weighted_motor_loss(prediction, target, flight, wing_channels, wing_weight, ground_weight):
    """Task identity is independent of clock: walking and wing flight both use 500 Hz."""
    per_world = (prediction - target).square().mean(-1)
    if wing_weight:
        per_world = per_world + wing_weight * flight * (
            prediction[:, wing_channels] - target[:, wing_channels]
        ).square().mean(-1)
    weights = torch.where(flight.bool(), 1.0, ground_weight)
    return (weights * per_world).mean()


def configure_optimizer(brain, args, parent, sensory_migrated):
    """Change the sensory learning rate while preserving existing Adam moments."""
    multiplier = args.sensor_lr_multiplier
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Sensory learning-rate multiplier must be finite and positive")
    if multiplier != 1 and not brain.sensor_extension_size:
        raise ValueError("A sensory learning-rate multiplier requires an extension")
    parameters = [p for p in brain.parameters() if p.requires_grad]
    sensor = brain.sensor_extension.weight if brain.sensor_extension_size else None
    grouped = bool(parent and parent.get("optimizer_sensor_grouped")) and not sensory_migrated
    if grouped:
        parameter_groups = [
            {"params": [p for p in parameters if p is not sensor]},
            {"params": [sensor]},
        ]
    else:
        parameter_groups = parameters
    optimizer = torch.optim.Adam(parameter_groups, lr=args.lr)
    resumed = parent is not None and "optimizer_state_dict" in parent and not sensory_migrated
    if resumed:
        optimizer.load_state_dict(parent["optimizer_state_dict"])
    if not grouped and multiplier != 1:
        # Moving the same Parameter object between groups preserves its existing
        # Adam step and moments. Do this after loading the original group layout.
        optimizer.param_groups[0]["params"] = [p for p in parameters if p is not sensor]
        optimizer.add_param_group({"params": [sensor]})
        grouped = True
    optimizer.param_groups[0]["lr"] = args.lr
    optimizer.param_groups[0]["role"] = "shared_actor"
    if grouped:
        optimizer.param_groups[1]["lr"] = args.lr * multiplier
        optimizer.param_groups[1]["role"] = "sensory_extension"
    return optimizer, resumed, grouped


def train(args):
    if not 0 <= args.reset_fraction <= 1:
        raise ValueError("Reset fraction must be between zero and one")
    if not np.isfinite(args.ground_loss_weight) or args.ground_loss_weight <= 0:
        raise ValueError("Ground retention weight must be finite and positive")
    args.height_inputs = getattr(args, "height_inputs", False)
    setup_start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    run_evidence = evidence()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    training, validation, splits, data_manifests = [], [], [], []
    clock_groups = {}
    wing_channels = None
    for dataset in [args.data, *args.additional_data]:
        episodes = load_episodes(
            dataset, args.wing_velocity_inputs, args.wing_angle_inputs, args.height_inputs
        )
        manifest = json.loads((dataset / "manifest.json").read_text())
        hz = manifest.get("control_hz", manifest.get("environment", {}).get("control_hz", 500))
        preset = manifest.get(
            "physical_preset", manifest.get("environment", {}).get("physical_preset")
        )
        flight = preset in ("flight", "wing_motion") or manifest.get("role") == "flight"
        if preset is None and manifest.get("role") is None:
            flight = hz == 5000  # legacy aerodynamic corpora predate explicit roles
        for episode in episodes:
            episode["flight_task"] = np.full(len(episode["action"]), flight, np.float32)
        time_scale = 1 / hz / CONTROL_DT
        if not np.isfinite(time_scale) or time_scale <= 0:
            raise ValueError("Invalid dataset action clock")
        group = clock_groups.setdefault(time_scale, {"training": [], "validation": []})
        if args.wing_loss_weight:
            channels = np.array(
                [
                    i
                    for i, a in enumerate(manifest["environment"]["actuation"])
                    if "wing_" in a["name"]
                ]
            )
            if len(channels) != 6 or (
                wing_channels is not None and not np.array_equal(wing_channels, channels)
            ):
                raise ValueError("Dataset wing channel ordering mismatch")
            wing_channels = channels
        validation_indices = set(
            np.random.default_rng(1193)
            .permutation(len(episodes))[: max(1, len(episodes) // 4)]
            .tolist()
        )
        training.extend(e for i, e in enumerate(episodes) if i not in validation_indices)
        validation.extend(e for i, e in enumerate(episodes) if i in validation_indices)
        group["training"].extend(
            e for i, e in enumerate(episodes) if i not in validation_indices
        )
        group["validation"].extend(
            e for i, e in enumerate(episodes) if i in validation_indices
        )
        manifest_hash = sha256(dataset / "manifest.json")
        data_manifests.append(manifest_hash)
        splits.append(
            {
                "manifest_sha256": manifest_hash,
                "validation_indices": sorted(validation_indices),
                "control_hz": hz,
                "neural_time_scale": time_scale,
                "height_reference_report_sha256": {
                    item["report"]: sha256(dataset / item["report"])
                    for item in manifest["episodes"]
                    if "report" in item
                }
                if args.height_inputs and preset == "wing_motion"
                else {},
                "episode_file_sha256": {
                    file.name: sha256(file) for file in sorted(dataset.glob("episode_*.npz"))
                },
            }
        )
    graph, sensory, descending, motor = load_malecns(args.graph)
    observation_size = episodes[0]["observation"].shape[1]
    action_size = episodes[0]["action"].shape[1]
    brain = EmbodiedBrain(
        graph,
        sensory,
        descending,
        motor,
        observation_size,
        action_size,
        internal_steps=args.internal_steps,
        sensor_extension_size=14
        if args.height_inputs
        else 12
        if args.wing_angle_inputs
        else 6
        if args.wing_velocity_inputs
        else 0,
    ).to(device)
    all_observations = np.concatenate([e["observation"] for e in training])
    parent = None
    sensory_migrated = False
    if args.resume:
        parent = torch.load(args.resume, map_location="cpu", weights_only=True)
        if parent["graph_sha256"] != sha256(args.graph / "weights.npz"):
            raise ValueError("Resume graph differs from checkpoint")
        if parent.get("graph_metadata_sha256") is not None and parent[
            "graph_metadata_sha256"
        ] != sha256(args.graph / "brain.npz"):
            raise ValueError("Resume neuron metadata differs from checkpoint")
        for name in ("sensory_ids", "descending_ids", "motor_ids"):
            if not torch.equal(parent["state_dict"][name], getattr(brain, name).cpu()):
                raise ValueError("Resume neuron routing differs from graph metadata")
        if parent["config"]["internal_steps"] != args.internal_steps:
            raise ValueError("Resume must preserve the recurrent architecture")
        sensory_migrated = initialize_extended_actor(brain, parent["state_dict"])
    else:
        brain.observation_mean.copy_(torch.from_numpy(all_observations.mean(0)).to(device))
        brain.observation_std.copy_(
            torch.from_numpy(all_observations.std(0)).to(device).clamp_min(0.05)
        )
    if args.freeze_core:
        brain.core.requires_grad_(False)
    core_initial = {n: p.detach().clone() for n, p in brain.core.named_parameters()}
    optimizer, optimizer_resumed, optimizer_sensor_grouped = configure_optimizer(
        brain, args, parent, sensory_migrated
    )
    fixed_validation = {
        scale: sample(
            group["validation"],
            np.random.default_rng(9044),
            args.burnin + args.sequence,
            args.worlds,
            device,
        )
        for scale, group in clock_groups.items()
    }
    reset_validation = {
        scale: sample(
            group["validation"],
            np.random.default_rng(9045),
            args.sequence,
            args.worlds,
            device,
            reset_start=True,
        )
        for scale, group in clock_groups.items()
    }

    @torch.no_grad()
    def validate(batch, burnin=args.burnin, time_scale=1.0):
        brain.eval()
        state = brain.initial_state(args.worlds)
        losses = []
        for t in range(burnin + args.sequence):
            result = brain(batch["observation"][t], state, time_scale=time_scale)
            state = result.state
            if t >= burnin:
                losses.append(F.mse_loss(result.action, batch["action"][t]))
        brain.train()
        return float(torch.stack(losses).mean())

    initial_by_clock = {s: validate(b, time_scale=s) for s, b in fixed_validation.items()}
    initial_reset_by_clock = {s: validate(b, 0, s) for s, b in reset_validation.items()}
    initial_validation = float(np.mean(list(initial_by_clock.values())))
    initial_reset_validation = float(np.mean(list(initial_reset_by_clock.values())))
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    optimization_started_utc = utc_now()
    started = time.perf_counter()
    updates = 0
    reset_updates = 0
    examples = 0
    grad_audit = None
    last_log = started
    max_memory = 0
    clock_updates = dict.fromkeys(clock_groups, 0)
    scales = list(clock_groups)
    with (args.output / "progress.jsonl").open("w") as log:
        while time.perf_counter() - started < args.seconds:
            reset_start = rng.random() < args.reset_fraction if args.reset_fraction else False
            burnin = 0 if reset_start else args.burnin
            time_scale = scales[rng.integers(len(scales))] if len(scales) > 1 else scales[0]
            clock_updates[time_scale] += 1
            reset_updates += int(reset_start)
            batch = sample(
                clock_groups[time_scale]["training"],
                rng,
                burnin + args.sequence,
                args.worlds,
                device,
                reset_start=reset_start,
            )
            state = brain.initial_state(args.worlds)
            with torch.no_grad():
                for t in range(burnin):
                    state = brain(batch["observation"][t], state, time_scale=time_scale).state
            optimizer.zero_grad(set_to_none=True)
            motor_losses, utility_losses = [], []
            for t in range(burnin, burnin + args.sequence):
                result = brain(batch["observation"][t], state, time_scale=time_scale)
                state = result.state
                flight = batch["flight_task"][t]
                motor_losses.append(
                    weighted_motor_loss(
                        result.action,
                        batch["action"][t],
                        flight,
                        wing_channels,
                        args.wing_loss_weight,
                        args.ground_loss_weight,
                    )
                )
                utility_losses.append(
                    (
                        F.cross_entropy(
                            result.utility_logits, batch["activity"][t], reduction="none"
                        )
                        * torch.where(flight.bool(), 1.0, args.ground_loss_weight)
                    ).mean()
                )
            motor_loss = torch.stack(motor_losses).mean()
            utility_loss = torch.stack(utility_losses).mean()
            loss = motor_loss + 0.02 * utility_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            if grad_audit is None:
                grad_audit = {}
                for name, p in brain.core.named_parameters():
                    grad_audit[name] = (
                        None
                        if p.grad is None
                        else {
                            "finite": bool(torch.isfinite(p.grad).all()),
                            "l2": float(p.grad.norm()),
                            "nonzero_cells": int((p.grad != 0).sum()),
                        }
                    )
                if not args.freeze_core and not all(
                    a is not None and a["finite"] and a["l2"] > 0 for a in grad_audit.values()
                ):
                    raise RuntimeError(f"Core is not receiving valid gradients: {grad_audit}")
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            updates += 1
            examples += args.sequence * args.worlds
            synchronize(device)
            now = time.perf_counter()
            if now - last_log >= 10 or updates == 1:
                record = {
                    "update": updates,
                    "seconds": now - started,
                    "motor_mse": float(motor_loss.detach()),
                    "utility_ce": float(utility_loss.detach()),
                    "supervised_examples": examples,
                    "reset_start_updates": reset_updates,
                    "neural_time_scale": time_scale,
                    "ground_loss_weight": args.ground_loss_weight,
                }
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
                last_log = now
    synchronize(device)
    training_seconds = time.perf_counter() - started
    optimization_completed_utc = utc_now()
    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated(device)
    evaluation_start = time.perf_counter()
    final_by_clock = {s: validate(b, time_scale=s) for s, b in fixed_validation.items()}
    final_reset_by_clock = {s: validate(b, 0, s) for s, b in reset_validation.items()}
    final_validation = float(np.mean(list(final_by_clock.values())))
    final_reset_validation = float(np.mean(list(final_reset_by_clock.values())))
    synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    changes = {
        name: {
            "l2": float((p.detach() - core_initial[name]).norm()),
            "changed_cells": int((p != core_initial[name]).sum()),
        }
        for name, p in brain.core.named_parameters()
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    # Paths are runtime details, omitted from portable/public checkpoint metadata.
    for key in ("graph", "data", "output", "additional_data", "resume"):
        config.pop(key)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in brain.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer_sensor_grouped": optimizer_sensor_grouped,
        "observation_size": observation_size,
        "action_size": action_size,
        "sensor_extension_size": brain.sensor_extension_size,
        "config": config,
        "source_commit": source,
        "graph_sha256": hashlib.sha256((args.graph / "weights.npz").read_bytes()).hexdigest(),
        "graph_metadata_sha256": sha256(args.graph / "brain.npz"),
        "data_manifest_sha256": hashlib.sha256(
            (args.data / "manifest.json").read_bytes()
        ).hexdigest(),
        "all_data_manifest_sha256": data_manifests,
        "parent_checkpoint_sha256": sha256(args.resume) if args.resume else None,
        "method": "recurrent behavioral cloning; utility warm-start labels rest/explore",
    }
    torch.save(checkpoint, args.output / "actor.pt")
    report = {
        "provenance": run_evidence,
        "optimization_started_utc": optimization_started_utc,
        "optimization_completed_utc": optimization_completed_utc,
        "source_commit": source,
        "config": config,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "neurons": graph.shape[0],
        "connections": graph.nnz,
        "trainable_parameters": sum(p.numel() for p in brain.parameters() if p.requires_grad),
        "core_parameters": sum(p.numel() for p in brain.core.parameters()),
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "updates": updates,
        "reset_start_updates": reset_updates,
        "supervised_examples": examples,
        "unique_training_frames": len(all_observations),
        "parallel_neural_sequences": args.worlds,
        "dataset_splits": splits,
        "parent_checkpoint_sha256": checkpoint["parent_checkpoint_sha256"],
        "optimizer_resumed": optimizer_resumed,
        "optimizer_sensor_grouped": optimizer_sensor_grouped,
        "optimizer_learning_rates": {
            group["role"]: group["lr"] for group in optimizer.param_groups
        },
        "sensory_extension_migrated": sensory_migrated,
        "optimizer_reset_reason": "new sensory parameters; preserve actor function, initialize fresh Adam"
        if sensory_migrated
        else None,
        "normalization": "retained from parent" if parent else "training frames only",
        "live_physics_worlds_during_imitation": 0,
        "initial_validation_motor_mse": initial_validation,
        "final_validation_motor_mse": final_validation,
        "initial_reset_validation_motor_mse": initial_reset_validation,
        "final_reset_validation_motor_mse": final_reset_validation,
        "peak_cuda_allocated_bytes": max_memory,
        "core_gradient_audit": grad_audit,
        "core_parameter_changes": changes,
        "clock_updates": clock_updates,
        "clock_sampling": "equal probability per control rate; homogeneous recurrent minibatches; one shared actor",
        "task_loss_weighting": "explicit flight/ground identity per example; independent of control rate",
        "initial_validation_by_neural_time_scale": initial_by_clock,
        "final_validation_by_neural_time_scale": final_by_clock,
        "initial_reset_validation_by_neural_time_scale": initial_reset_by_clock,
        "final_reset_validation_by_neural_time_scale": final_reset_by_clock,
        "checkpoint_sha256": hashlib.sha256(
            (args.output / "actor.pt").read_bytes()
        ).hexdigest(),
        "physical_success": "Not established by this supervised training report",
        "completed_utc": utc_now(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--additional-data", type=Path, action="append", default=[])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reset-fraction", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=8)
    parser.add_argument("--burnin", type=int, default=8)
    parser.add_argument("--internal-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--sensor-lr-multiplier", type=float, default=1)
    parser.add_argument("--seed", type=int, default=18001)
    parser.add_argument("--freeze-core", action="store_true")
    parser.add_argument("--wing-loss-weight", type=float, default=0)
    parser.add_argument("--ground-loss-weight", type=float, default=1)
    parser.add_argument("--wing-velocity-inputs", action="store_true")
    parser.add_argument("--wing-angle-inputs", action="store_true")
    parser.add_argument("--height-inputs", action="store_true")
    train(parser.parse_args())
