"""Paired, frozen-actor exploration diagnostic; no optimization or live resets."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.evaluate import load_actor
from embodied_fly.motor_focus import MotorTasks, TASKS
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def paired_actions(mean, noise, standard_normal):
    """The PPO tanh-Normal sampling rule, with an exact deterministic control."""
    noise = torch.as_tensor(noise, device=mean.device, dtype=mean.dtype)[:, None]
    if not torch.isfinite(noise).all() or (noise < 0).any():
        raise ValueError("Finite nonnegative noise required")
    latent = torch.atanh(mean.clamp(-0.9999, 0.9999))
    sampled = (latent + noise * standard_normal).tanh()
    return torch.where(noise == 0, mean, sampled)


def parameter_hash(actor):
    digest = hashlib.sha256()
    for name, value in actor.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    device = torch.device(args.device)
    torch.set_num_threads(4)
    actor, checkpoint = load_actor(args.resume, args.graph, device)
    actor.eval().requires_grad_(False)
    original = parameter_hash(actor)
    levels = np.asarray(args.noise, dtype=float)
    if not np.isfinite(levels).all() or (levels < 0).any() or 0 not in levels:
        raise ValueError("Include a deterministic control and nonnegative noise levels")
    worlds = 3 * len(levels)
    env = FlyBatch(worlds, min(worlds, 16), 14, preset="wing_position")
    if checkpoint["physical_contract"] != physical_contract(env.model):
        raise ValueError("Checkpoint physical contract mismatch")
    if not actor.motor_only:
        raise ValueError("Motor-only actor required")
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    noise = np.repeat(levels, 3)
    setup = time.perf_counter() - started
    results, files = [], {}
    started = time.perf_counter()
    for seed in args.seeds:
        tasks = MotorTasks(env, seed)
        # Match complete initial physical states, commands, and targets across noise.
        paired = np.tile(np.arange(3), len(levels))
        state = {k: env.fields[k][paired].copy() for k in ("qpos", "qvel", "act", "ctrl")}
        target_height = env.requested_height_cm[paired].copy()
        heading = tasks.heading[paired].copy()
        env.reset(np.arange(worlds), state=state)
        env.command[:] = 0
        env.command[:, 0] = paired == 1
        env.requested_height_cm[:] = target_height
        env.needs[:] = 0
        initial = env.fields["qpos"].copy()
        assert all(
            np.array_equal(initial[:3], initial[i : i + 3]) for i in range(0, worlds, 3)
        )
        generator = torch.Generator(device=device).manual_seed(seed + 101)
        memory = actor.initial_state(worlds)
        trace = []
        metrics = []
        first_failure = np.full(worlds, np.nan)
        with torch.no_grad():
            for step in range(round(args.seconds / env.control_dt)):
                observation = env.observation()
                output = actor(torch.as_tensor(observation, device=device), memory)
                memory = output.state
                # Common random numbers across paired conditions; independent cases.
                epsilon = torch.randn(
                    (3, actor.action_size), generator=generator, device=device
                )
                action = paired_actions(output.action, noise, epsilon.repeat(len(levels), 1))
                trace.append(
                    {
                        "qpos": env.fields["qpos"].copy(),
                        "qvel": env.fields["qvel"].copy(),
                        "activation": env.fields["act"].copy(),
                        "observation": observation.copy(),
                        "action": action.cpu().numpy(),
                        "mean_action": output.action.cpu().numpy(),
                    }
                )
                env.step(trace[-1]["action"])
                failed = tasks.failed()
                first_failure[np.isnan(first_failure) & failed] = (step + 1) * env.control_dt
                target = initial[:, :3].copy()
                distance = (step + 1) * env.control_dt * env.command[:, 0]
                target[:, 0] += distance * np.cos(heading)
                target[:, 1] += distance * np.sin(heading)
                velocity = env.velocity()
                metrics.append(
                    {
                        "height_cm": env.fields["qpos"][:, 2].copy(),
                        "upright": env.fields["xmat"][:, env.template.thorax_id, 8].copy(),
                        "root_error_cm": np.linalg.norm(
                            env.fields["qpos"][:, :3] - target, axis=1
                        ),
                        "speed_error_cm_s": velocity[:, 3] - env.command[:, 0],
                        "yaw_rad_s": velocity[:, 2].copy(),
                        "forbidden_support": env.forbidden_peak.copy() / env.body_weight,
                    }
                )
        path = args.output / f"seed_{seed}.npz"
        arrays = {key: np.stack([row[key] for row in trace]) for key in trace[0]}
        measured = {key: np.stack([row[key] for row in metrics]) for key in metrics[0]}
        np.savez_compressed(
            path,
            **arrays,
            **measured,
            initial_qpos=initial,
            noise=noise,
            requested_height_cm=target_height,
            heading=heading,
        )
        files[path.name] = sha256(path)
        for i in range(worlds):
            rms = lambda key: float(np.sqrt(np.mean(measured[key][:, i] ** 2)))
            result = {
                "seed": seed,
                "task": TASKS[paired[i]],
                "noise": float(noise[i]),
                "stable": bool(np.isnan(first_failure[i])),
                "first_failure_seconds": None
                if np.isnan(first_failure[i])
                else float(first_failure[i]),
                "root_rmse_mm": 10 * rms("root_error_cm"),
                "speed_rmse_cm_s": rms("speed_error_cm_s"),
                "yaw_rmse_rad_s": rms("yaw_rad_s"),
                "minimum_height_cm": float(measured["height_cm"][:, i].min()),
                "minimum_upright": float(measured["upright"][:, i].min()),
                "maximum_forbidden_support": float(measured["forbidden_support"][:, i].max()),
                "action_noise_rms": float(
                    np.sqrt(
                        np.mean((arrays["action"][:, i] - arrays["mean_action"][:, i]) ** 2)
                    )
                ),
                "forward_distance_cm": float(
                    (env.fields["qpos"][i, :2] - initial[i, :2])
                    @ np.array([np.cos(heading[i]), np.sin(heading[i])])
                ),
            }
            results.append(result)
            print(json.dumps(result), flush=True)
    synchronize(device)
    final = parameter_hash(actor)
    assert final == original
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.resume),
        "physical_contract": checkpoint["physical_contract"],
        "actor_parameters_sha256_before": original,
        "actor_parameters_sha256_after": final,
        "weights_unchanged": final == original,
        "optimizer_updates": 0,
        "live_resets": 0,
        "teacher_present": False,
        "critic_present": False,
        "noise_rule": "PPO pre-tanh independent Gaussian; common standard samples in paired worlds",
        "initial_states_matched": True,
        "noise_levels": levels.tolist(),
        "seeds": args.seeds,
        "seconds_per_case": args.seconds,
        "parallel_worlds": worlds,
        "transitions": len(metrics) * worlds * len(args.seeds),
        "physics_hz": 5000,
        "control_hz": 500,
        "setup_seconds": setup,
        "evaluation_and_capture_seconds": time.perf_counter() - started,
        "warning_count": int(env.fields["warning"].sum()),
        "files": files,
        "results": results,
        "scope": "Development diagnostic of exploration, not a policy acceptance evaluation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[98103, 98113, 98123])
    parser.add_argument("--noise", type=float, nargs="+", default=[0, 0.003, 0.01])
    run(parser.parse_args())
