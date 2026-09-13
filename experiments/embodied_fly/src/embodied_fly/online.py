"""Online recurrent braking correction with explicit frozen walking retention.

Training-only targets: inherited braking teacher at zero command, frozen parent
graph actor for moving commands. Deploy only the learned student actor.
"""

import argparse
import json
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from torch.nn import functional as F

from embodied_fly.batch import FlyBatch
from embodied_fly.body import CONTROL_DT
from embodied_fly.braking import BrakingTeacher
from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def train(args):
    if (
        min(
            args.worlds,
            args.threads,
            args.sequence,
            args.seconds,
            args.lr,
            args.retention_weight,
        )
        <= 0
    ):
        raise ValueError(
            "Worlds, threads, sequence, duration and learning rate must be positive"
        )
    if not 0 <= args.initial_teacher_mix <= 1:
        raise ValueError("Initial teacher mixture must be between zero and one")
    args.output.mkdir(parents=True, exist_ok=False)
    setup_started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    actor, parent = load_actor(args.resume, args.graph, device)
    anchor, _ = load_actor(args.resume, args.graph, device)
    anchor.requires_grad_(False)
    # Same measured graph can share immutable device buffers between copies.
    anchor.core.adjacency = actor.core.adjacency
    anchor.core.transpose = actor.core.transpose
    actor.train()
    env = FlyBatch(args.worlds, args.threads, actor.sensor_extension_size)
    teacher = BrakingTeacher(env, args.teacher, device)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    initial_core = {n: p.detach().clone() for n, p in actor.core.named_parameters()}
    cases = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0.5, 0, 0],
            [1, 0, 0],
            [2, 0, 0],
            [1.5, 0, 0.75],
            [1.5, 0, -0.75],
        ],
        np.float32,
    )
    env.command[:] = cases[np.arange(args.worlds) % len(cases)]
    env.reset(np.arange(args.worlds), yaw=rng.uniform(-0.2, 0.2, args.worlds))
    initial_commands = env.command.copy()
    memory = actor.initial_state(args.worlds)
    anchor_memory = anchor.initial_state(args.worlds)
    trace = deque(maxlen=128)
    episodes = []
    updates = transitions = switches = 0
    gradient_audit = None
    collection_seconds = optimization_seconds = 0.0
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_started
    started_utc = utc_now()
    started = time.perf_counter()
    failure = None
    try:
        while time.perf_counter() - started < args.seconds:
            forward_started = time.perf_counter()
            memory = memory.detach()
            motor_losses, utility_losses, holding_errors, moving_errors = [], [], [], []
            # Explicit assistance fades to zero halfway through this short pilot.
            mix = args.initial_teacher_mix * max(
                0, 1 - (time.perf_counter() - started) / (args.seconds * 0.5)
            )
            for _ in range(args.sequence):
                observation = torch.as_tensor(env.observation(), device=device)
                stationary = torch.as_tensor(
                    np.linalg.norm(env.command, axis=1) < 1e-6, device=device
                )
                with torch.no_grad():
                    reference = anchor(observation, anchor_memory)
                    anchor_memory = reference.state
                    braking = teacher.act()
                    target = torch.where(stationary[:, None], braking, reference.action)
                result = actor(observation, memory)
                memory = result.state
                error = (result.action - target).square().mean(-1)
                # Balance stop and moving examples within each simultaneous batch.
                groups = []
                if stationary.any():
                    holding_errors.append(float(error[stationary].mean().detach()))
                    groups.append(error[stationary].mean())
                if (~stationary).any():
                    moving_errors.append(float(error[~stationary].mean().detach()))
                    groups.append(args.retention_weight * error[~stationary].mean())
                motor_losses.append(torch.stack(groups).mean())
                utility_losses.append(
                    F.cross_entropy(result.utility_logits, (~stationary).long())
                )
                action = ((1 - mix) * result.action.detach() + mix * target).cpu().numpy()
                env.step(action)
                transitions += args.worlds
                trace.append(
                    {
                        "qpos": env.fields["qpos"].copy(),
                        "qvel": env.fields["qvel"].copy(),
                        "activation": env.fields["act"].copy(),
                        "ctrl": env.fields["ctrl"].copy(),
                        "action": env.previous_action.copy(),
                        "utility": result.utility_scores.detach().cpu().numpy(),
                        "command": env.command.copy(),
                    }
                )
                upright = env.fields["xmat"][:, env.template.thorax_id, 8]
                fallen = (upright < 0.5) | (env.fields["qpos"][:, 2] < 0.06)
                done = fallen | (env.ages >= round(2 / CONTROL_DT))
                ids = np.flatnonzero(done)
                for i in ids:
                    trace_info = None
                    if fallen[i]:
                        window = list(trace)[-min(int(env.ages[i]), len(trace)) :]
                        file = args.output / f"failure_{len(episodes):05d}.npz"
                        np.savez_compressed(
                            file, **{k: np.stack([f[k][i] for f in window]) for k in window[0]}
                        )
                        trace_info = {
                            "file": file.name,
                            "sha256": sha256(file),
                            "frames": len(window),
                        }
                    episodes.append(
                        {
                            "initial_command": initial_commands[i].tolist(),
                            "final_command": env.command[i].tolist(),
                            "failed": bool(fallen[i]),
                            "simulated_seconds": float(env.ages[i] * CONTROL_DT),
                            "teacher_mix_at_end": mix,
                            "failure_trace": trace_info,
                        }
                    )
                env.reset(ids, yaw=rng.uniform(-0.2, 0.2, len(ids)))
                env.command[ids] = cases[rng.integers(len(cases), size=len(ids))]
                initial_commands[ids] = env.command[ids]
                reset = torch.as_tensor(done, device=device)
                memory = actor.reset_worlds(memory, reset)
                anchor_memory = anchor.reset_worlds(anchor_memory, reset)
                # Half the worlds switch at one second while memory/physics persist.
                changing = np.flatnonzero(
                    (env.ages == round(1 / CONTROL_DT)) & (np.arange(args.worlds) % 2 == 0)
                )
                was_stationary = np.linalg.norm(env.command[changing], axis=1) < 1e-6
                env.command[changing] = 0
                env.command[changing, 0] = was_stationary.astype(float)
                switches += len(changing)
            loss = torch.stack(motor_losses).mean() + 0.02 * torch.stack(utility_losses).mean()
            synchronize(device)
            collection_seconds += time.perf_counter() - forward_started
            backward_started = time.perf_counter()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite online imitation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if gradient_audit is None:
                gradient_audit = {
                    n: {
                        "l2": float(p.grad.norm()),
                        "finite": bool(torch.isfinite(p.grad).all()),
                        "nonzero_cells": int((p.grad != 0).sum()),
                    }
                    for n, p in actor.core.named_parameters()
                }
                if not all(a["finite"] and a["l2"] > 0 for a in gradient_audit.values()):
                    raise RuntimeError("No valid internal-cell learning gradient")
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
            optimizer.step()
            updates += 1
            synchronize(device)
            optimization_seconds += time.perf_counter() - backward_started
            if updates == 1 or updates % 10 == 0:
                row = {
                    "updates": updates,
                    "seconds": time.perf_counter() - started,
                    "transitions": transitions,
                    "command_switches": switches,
                    "teacher_mix": mix,
                    "motor_loss": float(torch.stack(motor_losses).mean().detach()),
                    "utility_ce": float(torch.stack(utility_losses).mean().detach()),
                    "holding_mse": float(np.mean(holding_errors)) if holding_errors else None,
                    "moving_retention_mse": float(np.mean(moving_errors))
                    if moving_errors
                    else None,
                    "completed_episodes": len(episodes),
                    "falls": sum(e["failed"] for e in episodes),
                }
                with (args.output / "progress.jsonl").open("a") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        np.savez_compressed(args.output / "error_snapshot.npz", **env.fields)
        raise
    finally:
        synchronize(device)
        elapsed = time.perf_counter() - started
        config = {k: v for k, v in vars(args).items() if not isinstance(v, Path)}
        config["internal_steps"] = actor.internal_steps
        checkpoint = {
            "state_dict": {k: v.detach().cpu() for k, v in actor.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "observation_size": actor.observation_size,
            "sensor_extension_size": actor.sensor_extension_size,
            "action_size": actor.action_size,
            "config": config,
            "graph_sha256": parent["graph_sha256"],
            "graph_metadata_sha256": sha256(args.graph / "brain.npz"),
            "source_commit": provenance["source_commit"],
            "parent_checkpoint_sha256": sha256(args.resume),
            "method": "online recurrent imitation with inherited braking and frozen parent walking retention",
        }
        torch.save(checkpoint, args.output / "actor.pt")
        report = {
            "provenance": provenance,
            "config": config,
            "started_utc": started_utc,
            "completed_utc": utc_now(),
            "setup_seconds": setup_seconds,
            "training_wall_seconds": elapsed,
            "collection_and_supervised_forward_seconds": collection_seconds,
            "backward_optimizer_seconds": optimization_seconds,
            "updates": updates,
            "transitions": transitions,
            "aggregate_simulated_seconds": transitions * CONTROL_DT,
            "parallel_worlds": args.worlds,
            "command_switches_without_reset": switches,
            "episodes": episodes,
            "core_gradient_audit": gradient_audit,
            "core_changes": {
                n: float((p.detach() - initial_core[n]).norm())
                for n, p in actor.core.named_parameters()
            },
            "physics": "native MuJoCo CPU / mjbatch",
            "neural_device": str(device),
            "neural_device_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU",
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
            "checkpoint_sha256": sha256(args.output / "actor.pt"),
            "model_sha256": sha256(args.output / "model.mjb"),
            "parent_checkpoint_sha256": sha256(args.resume),
            "braking_teacher_sha256": sha256(args.teacher),
            "optimizer_initialization": "new imitation Adam; actor and normalization inherited",
            "execution": "mixture of student and training-only target, linearly faded to pure student halfway",
            "deployed_actor_count": 1,
            "teacher_required_at_evaluation": False,
            "failure": failure,
            "physical_success": "Independent teacher-free evaluation required",
        }
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("graph", "resume", "teacher", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--sequence", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--initial-teacher-mix", type=float, default=0.5)
    parser.add_argument("--retention-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=38001)
    train(parser.parse_args())
