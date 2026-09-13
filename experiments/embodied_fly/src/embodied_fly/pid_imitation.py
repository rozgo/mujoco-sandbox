"""Timed PID-to-student motor imitation on the accepted physical hover plant.

The PID supplies supervised targets and an explicitly annealed share of executed
commands during training only. The checkpoint is the unchanged graph actor; it
contains no controller, phase input, action mask or physical helper.
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.physical_contract import physical_contract
from embodied_fly.pid_hover import HoverPID, PIDConfig
from embodied_fly.ppo_timing import recurrent_forward
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.wing_position import wing_actuators


class BatchPID:
    """Use the accepted single-world PID verbatim on read-only scratch states."""

    def __init__(self, env, tasks):
        self.env = env
        self.controllers = [
            HoverPID(env.template, tasks.air_action, tasks.start[i]) for i in range(env.n)
        ]

    def reset(self, ids):
        for i in ids:
            self.controllers[i].integral[:] = 0

    def act(self):
        e, actions = self.env, []
        for i, controller in enumerate(self.controllers):
            for k in ("qpos", "qvel", "act", "ctrl"):
                getattr(e.template.data, k)[:] = e.fields[k][i]
            e.template.data.time = e.ages[i] * e.control_dt
            mujoco.mj_forward(e.template.model, e.template.data)
            controller.target[:2] = e.requested_xy_cm[i]
            controller.target[2] = e.requested_height_cm[i]
            actions.append(controller.act())
        return np.asarray(actions, dtype=np.float32)


def teacher_fraction(progress):
    """First half expert-driven, next 30% handoff, final 20% student-driven."""
    return float(np.clip((0.8 - progress) / 0.3, 0, 1))


def imitation_loss(action, target, wings):
    other = np.setdiff1d(np.arange(action.shape[-1]), wings)
    wing = (action[:, wings] - target[:, wings]).square().mean()
    posture = (action[:, other] - target[:, other]).square().mean()
    return wing + 0.1 * posture, wing, posture


def train(args):
    if min(args.worlds, args.sequence, args.seconds, args.lr, args.episode_seconds) <= 0:
        raise ValueError("Positive training settings required")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance, started = evidence(), time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    actor, parent = load_actor(args.resume, args.graph, device)
    if not actor.motor_only or actor.sensor_extension_size != 16:
        raise ValueError("PID imitation requires the existing motor-only 399-input actor")
    env = FlyBatch(
        args.worlds,
        args.threads,
        16,
        preset="wing_position",
        wing_response="instant",
        physics_hz=1000,
    )
    assert physical_contract(env.model) == parent["physical_contract"]
    tasks = HoverOnlyTasks(env, args.seed)
    teacher = BatchPID(env, tasks)
    wings = wing_actuators(env.model)
    original = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    actor.train()
    parameters = [p for p in actor.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    memory = actor.initial_state(args.worlds)
    context = torch.ones(args.worlds, dtype=torch.long, device=device)
    # Different episode lengths create different observed wing phases. Half the
    # worlds keep exact nominal starts; the rest receive small reset-only offsets.
    horizon = np.zeros(args.worlds, dtype=int)
    episode_teacher_sum = np.zeros(args.worlds)

    def reset(ids):
        if not len(ids):
            return
        tasks.reset(ids)
        shifted = ids[ids % 2 == 1]
        env.fields["qpos"][shifted, 2] += rng.uniform(-0.02, 0.02, len(shifted))
        env.fields["qvel"][shifted, 2] += rng.uniform(-0.5, 0.5, len(shifted))
        env.batch.forward(ids)
        env.mean_sensors[ids] = env.fields["sensordata"][ids]
        teacher.reset(ids)
        episode_teacher_sum[ids] = 0
        horizon[ids] = np.rint(
            rng.uniform(0.8, 1, len(ids)) * args.episode_seconds / env.control_dt
        ).astype(int)

    reset(np.arange(env.n))
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    config = {k: v for k, v in parent["config"].items()}
    config.update(
        pid_imitation={
            "seconds": args.seconds,
            "worlds": args.worlds,
            "sequence": args.sequence,
            "lr": args.lr,
            "seed": args.seed,
            "episode_seconds": args.episode_seconds,
            "teacher_schedule": "1 for first50%, linear to0 by80%, then0",
            "wing_loss_weight": 1,
            "nonwing_posture_weight": 0.1,
        }
    )

    def save(name):
        child = {
            k: v
            for k, v in parent.items()
            if k
            not in (
                "optimizer_state_dict",
                "critic_state_dict",
                "value_optimizer_state_dict",
                "imitation_optimizer_state_dict",
            )
        }
        child.update(
            state_dict={k: v.detach().cpu() for k, v in actor.state_dict().items()},
            config=config,
            parent_checkpoint_sha256=sha256(args.resume),
            source_commit=provenance["source_commit"],
            method="PID-supervised recurrent motor imitation; teacher absent at deployment",
            imitation_optimizer_state_dict=optimizer.state_dict(),
        )
        torch.save(child, args.output / name)

    synchronize(device)
    setup_seconds = time.perf_counter() - started
    started_utc, started = utc_now(), time.perf_counter()
    updates = transitions = student_transitions = 0
    equivalent_teacher_transitions = 0.0
    collection_seconds = optimization_seconds = 0.0
    episodes, batches = [], []
    gradient_audit, supervised_checkpoint = None, False
    with (args.output / "progress.jsonl").open("x") as log:
        while time.perf_counter() - started < args.seconds:
            progress = (time.perf_counter() - started) / args.seconds
            mixture = teacher_fraction(progress)
            if mixture < 1 and not supervised_checkpoint:
                save("teacher_stage.pt")
                supervised_checkpoint = True
            memory = memory.detach()
            losses, wing_losses, posture_losses, rows = [], [], [], []
            begin = time.perf_counter()
            for _ in range(args.sequence):
                observation = env.observation()
                target_np = teacher.act()
                target = torch.as_tensor(target_np, device=device)
                result = recurrent_forward(
                    actor,
                    torch.as_tensor(observation, device=device),
                    memory,
                    context,
                    1.0,
                    True,
                )
                memory = result.state
                loss, wing_loss, posture_loss = imitation_loss(result.action, target, wings)
                losses.append(loss)
                wing_losses.append(wing_loss.detach())
                posture_losses.append(posture_loss.detach())
                student = result.action.detach().cpu().numpy()
                executed = mixture * target_np + (1 - mixture) * student
                rows.append(
                    {
                        "observation": observation.copy(),
                        "teacher_action": target_np.copy(),
                        "student_action": student.copy(),
                        "executed_action": executed.copy(),
                        "teacher_fraction": np.full(env.n, mixture),
                        "qpos": env.fields["qpos"].copy(),
                        "qvel": env.fields["qvel"].copy(),
                        "age_actions": env.ages.copy(),
                    }
                )
                env.step(executed)
                rows[-1]["post_qpos"] = env.fields["qpos"].copy()
                rows[-1]["post_qvel"] = env.fields["qvel"].copy()
                rows[-1]["forbidden_bodyweights"] = env.forbidden_peak.copy() / env.body_weight
                episode_teacher_sum += mixture
                transitions += env.n
                student_transitions += env.n if mixture == 0 else 0
                equivalent_teacher_transitions += mixture * env.n
                failed = (
                    (env.fields["qpos"][:, 2] < 0.5)
                    | (env.fields["xmat"][:, env.template.thorax_id, 8] < 0.5)
                    | (env.forbidden_peak / env.body_weight > 0.1)
                )
                done = failed | (env.ages >= horizon)
                for i in np.flatnonzero(done):
                    episodes.append(
                        {
                            "world": int(i),
                            "failed": bool(failed[i]),
                            "seconds": float(env.ages[i] * env.control_dt),
                            "teacher_fraction_at_end": mixture,
                            "teacher_fraction_mean": float(
                                episode_teacher_sum[i] / env.ages[i]
                            ),
                            "entire_episode_student_only": bool(episode_teacher_sum[i] == 0),
                            "trace_file": f"batch_{updates:05d}.npz",
                            "trace_end_frame": len(rows) - 1,
                        }
                    )
                reset(np.flatnonzero(done))
                memory = actor.reset_worlds(memory, torch.as_tensor(done, device=device))
            synchronize(device)
            collection_seconds += time.perf_counter() - begin
            begin = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite imitation loss")
            loss.backward()
            if gradient_audit is None:
                gradient_audit = {
                    k: {
                        "l2": float(p.grad.norm()),
                        "finite": bool(torch.isfinite(p.grad).all()),
                    }
                    for k, p in actor.core.named_parameters()
                }
                assert all(v["finite"] and v["l2"] > 0 for v in gradient_audit.values())
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
            synchronize(device)
            optimization_seconds += time.perf_counter() - begin
            file = args.output / f"batch_{updates:05d}.npz"
            np.savez_compressed(file, **{k: np.stack([r[k] for r in rows]) for k in rows[0]})
            batches.append(
                {
                    "file": file.name,
                    "sha256": sha256(file),
                    "actions": args.sequence * args.worlds,
                    "teacher_fraction": mixture,
                }
            )
            updates += 1
            row = {
                "elapsed_seconds": time.perf_counter() - started,
                "updates": updates,
                "transitions": transitions,
                "teacher_fraction": mixture,
                "wing_mse": float(torch.stack(wing_losses).mean()),
                "posture_mse": float(torch.stack(posture_losses).mean()),
                "completed_episodes": len(episodes),
                "failed_episodes": sum(e["failed"] for e in episodes),
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
    synchronize(device)
    training_seconds = time.perf_counter() - started
    save("actor.pt")
    fixed, changed = [], []
    for key, value in actor.state_dict().items():
        assert torch.isfinite(value).all()
        (fixed if torch.equal(value.cpu(), original[key]) else changed).append(key)
    assert all(
        k in fixed
        for k in original
        if k.startswith(("utility_head.", "intention_encoder.", "observation_"))
        or k.endswith("_ids")
    )
    report = {
        "provenance": provenance,
        "training_started_utc": started_utc,
        "completed_utc": utc_now(),
        "setup_seconds": setup_seconds,
        "training_wall_seconds": training_seconds,
        "collection_seconds": collection_seconds,
        "optimization_seconds": optimization_seconds,
        "trace_write_and_loop_overhead_seconds": training_seconds
        - collection_seconds
        - optimization_seconds,
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "teacher_stage_checkpoint_sha256": sha256(args.output / "teacher_stage.pt")
        if supervised_checkpoint
        else None,
        "parent_checkpoint_sha256": sha256(args.resume),
        "physical_contract": physical_contract(env.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "pid_config": asdict(PIDConfig()),
        "config": config["pid_imitation"],
        "updates": updates,
        "transitions": transitions,
        "aggregate_simulated_seconds": transitions * env.control_dt,
        "student_only_transitions": student_transitions,
        "equivalent_teacher_action_share": equivalent_teacher_transitions / transitions,
        "physics_hz": 1000,
        "control_hz": 500,
        "worlds": env.n,
        "threads": args.threads,
        "neural_device": str(device),
        "neural_device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else str(device),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "actor_parameters": sum(p.numel() for p in actor.parameters()),
        "core_gradient_audit": gradient_audit,
        "fixed_tensors": fixed,
        "changed_tensors": changed,
        "optimizer_initialization": "Fresh imitation Adam; PPO/critic optimizer states removed for next stage",
        "teacher_has_clock_and_integral": True,
        "actor_receives_teacher_clock_or_integral": False,
        "all_motor_outputs_learned": True,
        "executed_body_force_override": False,
        "episodes": episodes,
        "traces": batches,
        "scope": "Training-only PID imitation with action handoff; independent actor-only evaluation required",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--worlds", type=int, default=32)
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--sequence", type=int, default=64)
    p.add_argument("--episode-seconds", type=float, default=4)
    p.add_argument("--lr", type=float, default=0.0001)
    p.add_argument("--seed", type=int, default=120401)
    train(p.parse_args())
