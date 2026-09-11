"""Attribute uncaptured GPU kernel time; not a captured-throughput benchmark."""

import json
from collections import defaultdict

import numpy as np

from adaptive_locomotion.bodies import PRESETS, ROOT
from adaptive_locomotion.env import DogEnv


def main():
    output = ROOT / "outputs/locomotion/warp_backend/kernel_profile.json"
    if output.exists():
        raise ValueError("Preserve the existing profile")
    env = DogEnv(
        num_envs=512,
        bodies=[PRESETS["healthy"]],
        randomize=False,
        faults=False,
        threads=16,
        physics_backend="warp",
    )
    try:
        for _ in range(10):
            env.step(np.zeros((env.n, 12)))
        b = env.groups[0].batch
        b.upload_controls()
        totals = defaultdict(float)
        with b.wp.ScopedDevice(b.device), b.wp.ScopedTimer(
            "uncaptured step", print=False, cuda_filter=b.wp.TIMING_KERNEL
        ) as timer:
            for _ in range(3):
                b.mjw.step(b.m, b.d)
        for item in timer.timing_results:
            totals[item.name] += item.elapsed
        rows = sorted(totals.items(), key=lambda x: x[1], reverse=True)
        result = {
            "steps": 3,
            "worlds": env.n,
            "cuda_kernel_ms": sum(totals.values()),
            "top_kernels_ms": rows[:30],
            "all_kernel_ms": rows,
            "scope": "Uncaptured kernel attribution only. CUDA events include instrumentation; use synchronized captured benchmarks for throughput.",
        }
        output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        env.close()


if __name__ == "__main__":
    main()
