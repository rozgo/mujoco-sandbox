"""Compare CPU and CUDA physics separately from the NumPy environment bridge."""

import argparse
import gc
import json
import platform
import subprocess
import time
from importlib.metadata import version

import numpy as np
from benchmark_learner import gpu_sample

from adaptive_locomotion.bodies import PRESETS, ROOT
from adaptive_locomotion.env import DogEnv


def run(backend, family, num_envs, seconds=3, repeats=3):
    started = time.perf_counter()
    env = DogEnv(
        num_envs=num_envs,
        seed=9221,
        threads=16,
        physics_backend=backend,
        bodies=[PRESETS["healthy"]] if family == "healthy" else None,
        limb_stage="consolidate" if family == "nine" else None,
        randomize=False,
        faults=False,
        randomize_strength=False,
        reward_profile="walk",
        support_weight=2,
        stride_weight=1,
        balance_weight=5,
        body_motion_weight=1,
        damage_flight_weight=3,
        visible_step_weight=2,
        rear_overlap_weight=3,
    )
    setup = time.perf_counter() - started
    records = []
    try:
        for scope in ("physics_only", "environment_bridge"):
            for repeat in range(repeats):
                env.reset(np.arange(env.n))
                action = np.zeros((env.n, 12), np.float32)
                for _ in range(10):
                    env.step(action)
                if backend == "warp":
                    for g in env.groups:
                        g.batch.upload_controls()
                    env.groups[0].batch.wp.synchronize_device()
                start, steps = time.perf_counter(), 0
                while time.perf_counter() - start < seconds:
                    if scope == "environment_bridge":
                        _, done, _, _ = env.step(action)
                        env.reset(np.flatnonzero(done))
                        steps += 1
                    elif backend == "warp":
                        for _ in range(10):
                            for g in env.groups:
                                g.batch.step_device(env.decimation)
                        env.groups[0].batch.wp.synchronize_device()
                        steps += 10
                    else:

                        def advance(g):
                            g.batch.step(nstep=env.decimation)

                        if env.pool:
                            list(env.pool.map(advance, env.groups))
                        else:
                            advance(env.groups[0])
                        steps += 1
                if backend == "warp":
                    env.groups[0].batch.wp.synchronize_device()
                elapsed = time.perf_counter() - start
                for g in env.groups:
                    if backend == "warp":
                        g.batch.download()
                    assert np.isfinite(g.qpos).all() and np.isfinite(g.qvel).all()
                    assert np.all(np.abs(g.torque) <= g.force_range[:, :, 1] + 1e-4)
                records.append(
                    {
                        "scope": scope,
                        "repeat": repeat,
                        "elapsed_seconds": elapsed,
                        "control_intervals": steps * num_envs,
                        "control_intervals_per_second": steps * num_envs / elapsed,
                        "physics_steps_per_second": steps
                        * num_envs
                        * env.decimation
                        / elapsed,
                    }
                )
        return {
            "physics_backend": backend,
            "family": family,
            "num_envs": num_envs,
            "setup_seconds": setup,
            "capture_compile_seconds": sum(
                getattr(g.batch, "compile_seconds", 0) for g in env.groups
            ),
            "group_world_counts": [g.n for g in env.groups],
            "model_sizes": [[g.model.nq, g.model.nv, g.model.nu] for g in env.groups],
            "physics_timestep_s": env.timestep,
            "control_timestep_s": 0.02,
            "full_collision_and_support_sensors": True,
            "records": records,
            "median_control_intervals_per_second": {
                scope: float(
                    np.median(
                        [
                            r["control_intervals_per_second"]
                            for r in records
                            if r["scope"] == scope
                        ]
                    )
                )
                for scope in ("physics_only", "environment_bridge")
            },
        }
    finally:
        env.close()
        gc.collect()


def main(backend, family, num_envs, label):
    path = ROOT / "outputs/locomotion/warp_backend" / f"{label}.json"
    if path.exists():
        raise ValueError("Existing result must be preserved; choose another label")
    path.parent.mkdir(parents=True, exist_ok=True)
    start_gpu = gpu_sample()
    result = run(backend, family, num_envs)
    result.update(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        python=platform.python_version(),
        mujoco=version("mujoco"),
        mujoco_warp=version("mujoco-warp"),
        warp=version("warp-lang"),
        gpu_before=start_gpu,
        gpu_after=gpu_sample(),
        scope="Standing commands, no policy/learner/reference matching. Physics-only excludes host transfers and rewards. Environment bridge includes transfers, observations, existing rewards and episode resets. This is not PPO throughput.",
    )
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("mjbatch", "warp"), required=True)
    p.add_argument("--family", choices=("healthy", "nine"), required=True)
    p.add_argument("--num-envs", type=int, required=True)
    p.add_argument("--label", required=True)
    main(**vars(p.parse_args()))
