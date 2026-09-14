"""Frozen-policy noise comparison on the exact PPO collection observation path.

No optimizer, teacher, resets after initialization, or changes to the force law.
Common Gaussian draws pair the half/current-noise conditions across 32 worlds.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.ppo import motor_distribution
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.velocity_demonstrations import environment, post_row, pre_row
from embodied_fly.velocity_hover import HoverReward, metrics, start_states
from embodied_fly.velocity_motor import observation
from embodied_fly.wing_position import wing_actuators

EPISODES = (0, 1, 2, 3, 6, 7, 8, 9)


def sampled_action(output, log_std, scale, generator):
    """Use PPO's tanh-normal distribution, or its deployed mean when scale is zero."""
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("Noise scale must be finite and nonnegative")
    active = torch.ones(output.action.shape[-1], dtype=torch.bool, device=log_std.device)
    dist = motor_distribution(output, active, log_std, 0.0005)
    noise = torch.randn(
        dist.loc.shape, generator=generator, device=dist.loc.device, dtype=dist.loc.dtype
    )
    return output.action if scale == 0 else (dist.loc + scale * dist.scale * noise).tanh()


@torch.no_grad()
def audit(args):
    begin = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    actor.eval()
    initial = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    env = environment(32, 16)
    assert physical_contract(env.model) == parent["physical_contract"]
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    episodes = np.tile(EPISODES, 4)
    resets = start_states(args.dataset, episodes)
    commands = np.zeros((env.n, 4), np.float32)
    log_std = parent["log_std"].to(device)
    wing_ids = wing_actuators(env.model)
    sweep_addresses = env.model.jnt_qposadr[env.model.actuator_trnid[wing_ids[[0, 3]], 0]]
    evaluations = {}
    setup_seconds = time.perf_counter() - begin
    for scale in (0.0, 0.5, 1.0):
        label = {0.0: "zero", 0.5: "half", 1.0: "current"}[scale]
        root = args.output / label
        root.mkdir()
        env.reset(np.arange(env.n), state=resets)
        memory = actor.initial_state(env.n)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        reward_fn = HoverReward(env, horizontal_scale=2.0)
        first_failure = np.full(env.n, np.nan)
        returns = np.zeros(env.n)
        traces, samples = [], []
        synchronize(device)
        capture_start = time.perf_counter()
        for step in range(5000):
            # Same post-step sensor timing as PPO collection, deliberately no forward().
            row = pre_row(env, commands, step * 0.002, 0, refresh=False)
            row["angular_velocity"] = env.velocity()[:, :3].copy()
            output = actor(torch.as_tensor(observation(env, commands), device=device), memory)
            action = sampled_action(output, log_std, scale, generator).cpu().numpy()
            memory = output.state
            row["action"] = action
            env.step(action)
            row.update(post_row(env))
            reward, failed, _ = reward_fn()
            returns += reward * np.isnan(first_failure)
            first_failure[failed & np.isnan(first_failure)] = (step + 1) * 0.002
            # Four predeclared standard starts, first replicate only, for video replay.
            selected = [0, 1, 6, 7]
            traces.append({k: v[selected].copy() for k, v in row.items()})
            samples.append(
                {
                    k: row[k].copy()
                    for k in (
                        "time",
                        "measured_velocity",
                        "angular_velocity",
                        "post_position",
                        "upright",
                    )
                }
                | {
                    "qpos": row["qpos"][:, :3].copy(),
                    "sweeps": row["qpos"][:, sweep_addresses].copy(),
                }
            )
        synchronize(device)
        capture_seconds = time.perf_counter() - capture_start
        arrays = {k: np.stack([r[k] for r in samples]) for k in samples[0]}
        np.savez_compressed(root / "all_worlds.npz", **arrays)
        cases = []
        for world, episode in enumerate(episodes):
            failure = None if np.isnan(first_failure[world]) else float(first_failure[world])
            own = {k: v[:, world] for k, v in arrays.items()}
            case = {
                "world": world,
                "episode": int(episode),
                "replicate": world // 8,
                "return_before_failure": float(returns[world]),
                **metrics(own, failure, 10),
            }
            stop = 5000 if failure is None else round(failure * 500)
            if stop > 500:
                sweep = own["sweeps"][500:stop]
                freq = np.fft.rfftfreq(len(sweep), d=0.002)
                band = (freq >= 1) & (freq <= 100)
                fft = np.abs(np.fft.rfft(sweep - sweep.mean(0), axis=0))
                case["sweep_peak_hz_left_right"] = freq[band][
                    np.argmax(fft[band], axis=0)
                ].tolist()
            if world in selected:
                index = selected.index(world)
                trace = {k: np.stack([r[k][index] for r in traces]) for k in traces[0]}
                file = root / f"episode_{episode:02d}.npz"
                np.savez_compressed(file, **trace)
                case.update(file=file.name, sha256=sha256(file))
            cases.append(case)
        survivors = [c for c in cases if c["survived_ten_seconds"]]
        evaluations[label] = {
            "scale": scale,
            "latent_std_mean": float(log_std.exp().mean()) * scale,
            "capture_seconds": capture_seconds,
            "cases": cases,
            "survived": len(survivors),
            "worlds": env.n,
            "mean_airborne_seconds": float(np.mean([c["airborne_seconds"] for c in cases])),
            "mean_return": float(returns.mean()),
            "survivor_mean_total_rms_mm_s": float(
                np.mean([c["velocity_rms_mm_s"] for c in survivors])
            )
            if survivors
            else None,
            "survivor_mean_climb_mm": float(
                np.mean([c["final_displacement_mm"][2] for c in survivors])
            )
            if survivors
            else None,
            "all_worlds_sha256": sha256(root / "all_worlds.npz"),
        }
        (root / "report.json").write_text(json.dumps(evaluations[label], indent=2) + "\n")
        print(
            json.dumps({k: v for k, v in evaluations[label].items() if k != "cases"}),
            flush=True,
        )
        del traces, samples, arrays
    unchanged = all(torch.equal(v, initial[k]) for k, v in actor.state_dict().items())
    assert unchanged
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "weights_unchanged": unchanged,
        "physical_contract": parent["physical_contract"],
        "graph_sha256": parent["graph_sha256"],
        "model_sha256": sha256(args.output / "model.mjb"),
        "seed": args.seed,
        "worlds_per_condition": 32,
        "episodes": list(EPISODES),
        "replicates": 4,
        "control_hz": 500,
        "physics_hz": 1000,
        "seconds_per_flight": 10,
        "observation_timing": "PPO collection: physics-step return; no extra forward",
        "scope": "Frozen actor; common random draws per world; no weight updates or physical rescue. Four noise replicates per start are not independent trained seeds.",
        "failure_metrics": "Before first failure only. Compare complete-flight RMS among survivors separately from survival; short failed trajectories must not win by low RMS.",
        "evaluations": evaluations,
        "setup_seconds": setup_seconds,
        "total_wall_seconds": time.perf_counter() - begin,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=151211)
    audit(p.parse_args())
