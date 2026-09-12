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

import numpy as np
import torch
from torch.nn import functional as F

from embodied_fly.brain import EmbodiedBrain, load_malecns


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def load_episodes(path):
    manifest = json.loads((path / "manifest.json").read_text())
    rejected = {
        e["episode"]
        for e in manifest["episodes"]
        if e["failure"] or e.get("physical_failure") or e["final_upright"] < 0.5
    }
    episodes = []
    for file in sorted(path.glob("episode_*.npz")):
        if int(file.stem.split("_")[-1]) in rejected:
            continue
        with np.load(file, allow_pickle=False) as data:
            episodes.append(
                {key: data[key].copy() for key in ("observation", "action", "activity")}
            )
    if len(episodes) < 4:
        raise ValueError("Need at least four separate episodes for train/validation split")
    return episodes


def sample(episodes, rng, length, worlds, device):
    choices = rng.integers(len(episodes), size=worlds)
    sequences = []
    for choice in choices:
        episode = episodes[choice]
        if len(episode["action"]) < length:
            raise ValueError("Episode shorter than recurrent context")
        start = rng.integers(len(episode["action"]) - length + 1)
        sequences.append(
            {key: value[start : start + length] for key, value in episode.items()}
        )
    return {
        key: torch.as_tensor(np.stack([s[key] for s in sequences], axis=1), device=device)
        for key in sequences[0]
    }


def train(args):
    setup_start = time.perf_counter()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    episodes = load_episodes(args.data)
    validation_indices = set(
        np.random.default_rng(1193)
        .permutation(len(episodes))[: max(1, len(episodes) // 4)]
        .tolist()
    )
    training = [e for i, e in enumerate(episodes) if i not in validation_indices]
    validation = [e for i, e in enumerate(episodes) if i in validation_indices]
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
    ).to(device)
    all_observations = np.concatenate([e["observation"] for e in training])
    brain.observation_mean.copy_(torch.from_numpy(all_observations.mean(0)).to(device))
    brain.observation_std.copy_(
        torch.from_numpy(all_observations.std(0)).to(device).clamp_min(0.05)
    )
    if args.freeze_core:
        brain.core.requires_grad_(False)
    core_initial = {n: p.detach().clone() for n, p in brain.core.named_parameters()}
    optimizer = torch.optim.Adam(
        (p for p in brain.parameters() if p.requires_grad), lr=args.lr
    )
    fixed_validation = sample(
        validation,
        np.random.default_rng(9044),
        args.burnin + args.sequence,
        args.worlds,
        device,
    )

    @torch.no_grad()
    def validate():
        brain.eval()
        state = brain.initial_state(args.worlds)
        losses = []
        for t in range(args.burnin + args.sequence):
            result = brain(fixed_validation["observation"][t], state)
            state = result.state
            if t >= args.burnin:
                losses.append(F.mse_loss(result.action, fixed_validation["action"][t]))
        brain.train()
        return float(torch.stack(losses).mean())

    initial_validation = validate()
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    args.output.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    started = time.perf_counter()
    updates = 0
    examples = 0
    grad_audit = None
    last_log = started
    max_memory = 0
    with (args.output / "progress.jsonl").open("w") as log:
        while time.perf_counter() - started < args.seconds:
            batch = sample(training, rng, args.burnin + args.sequence, args.worlds, device)
            state = brain.initial_state(args.worlds)
            with torch.no_grad():
                for t in range(args.burnin):
                    state = brain(batch["observation"][t], state).state
            optimizer.zero_grad(set_to_none=True)
            motor_losses, utility_losses = [], []
            for t in range(args.burnin, args.burnin + args.sequence):
                result = brain(batch["observation"][t], state)
                state = result.state
                motor_losses.append(F.mse_loss(result.action, batch["action"][t]))
                utility_losses.append(
                    F.cross_entropy(result.utility_logits, batch["activity"][t])
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
                    "motor_mse": float(motor_loss),
                    "utility_ce": float(utility_loss),
                    "supervised_examples": examples,
                }
                print(json.dumps(record), flush=True)
                log.write(json.dumps(record) + "\n")
                log.flush()
                last_log = now
    synchronize(device)
    training_seconds = time.perf_counter() - started
    if device.type == "cuda":
        max_memory = torch.cuda.max_memory_allocated(device)
    evaluation_start = time.perf_counter()
    final_validation = validate()
    synchronize(device)
    evaluation_seconds = time.perf_counter() - evaluation_start
    changes = {
        name: {
            "l2": float((p - core_initial[name]).norm()),
            "changed_cells": int((p != core_initial[name]).sum()),
        }
        for name, p in brain.core.named_parameters()
    }
    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    # Paths are runtime details, omitted from portable/public checkpoint metadata.
    for key in ("graph", "data", "output"):
        config.pop(key)
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in brain.state_dict().items()},
        "observation_size": observation_size,
        "action_size": action_size,
        "config": config,
        "source_commit": source,
        "graph_sha256": hashlib.sha256((args.graph / "weights.npz").read_bytes()).hexdigest(),
        "data_manifest_sha256": hashlib.sha256(
            (args.data / "manifest.json").read_bytes()
        ).hexdigest(),
        "method": "recurrent behavioral cloning; utility warm-start labels rest/explore",
    }
    torch.save(checkpoint, args.output / "actor.pt")
    report = {
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
        "supervised_examples": examples,
        "unique_training_frames": len(all_observations),
        "parallel_neural_sequences": args.worlds,
        "validation_indices_in_accepted_episodes": sorted(validation_indices),
        "live_physics_worlds_during_imitation": 0,
        "initial_validation_motor_mse": initial_validation,
        "final_validation_motor_mse": final_validation,
        "peak_cuda_allocated_bytes": max_memory,
        "core_gradient_audit": grad_audit,
        "core_parameter_changes": changes,
        "checkpoint_sha256": hashlib.sha256(
            (args.output / "actor.pt").read_bytes()
        ).hexdigest(),
        "physical_success": "Not established by this supervised training report",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--worlds", type=int, default=8)
    parser.add_argument("--sequence", type=int, default=8)
    parser.add_argument("--burnin", type=int, default=8)
    parser.add_argument("--internal-steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--seed", type=int, default=18001)
    parser.add_argument("--freeze-core", action="store_true")
    train(parser.parse_args())
