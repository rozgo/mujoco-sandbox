"""Measured device/batch benchmarks, separate from training and startup costs."""

import json
import time

import numpy as np

from .paths import OUTPUTS


def neural(device="cpu", batch=8, ticks=100):
    import sys

    from .paths import NEURAL_DATA, VENDOR

    sys.path.insert(0, str(VENDOR))
    from fly_brain import FlyBrain

    start = time.perf_counter()
    brain = FlyBrain(data=NEURAL_DATA, device=device, batch=batch)
    for _ in range(3):
        brain.step()
    if device == "cuda":
        brain.xp.cuda.Stream.null.synchronize()
    setup = time.perf_counter() - start
    inject = [(brain.cells(["LC4", "LPLC2"], "L"), np.linspace(0, 0.45, batch))]
    count = np.zeros(batch, int)
    start = time.perf_counter()
    for _ in range(ticks):
        fired = brain.step(inject=inject)
        count += np.array([len(fired)] if batch == 1 else [len(x) for x in fired])
    if device == "cuda":
        brain.xp.cuda.Stream.null.synchronize()
    wall = time.perf_counter() - start
    result = {
        "kind": "neural",
        "device": device,
        "batch": batch,
        "ticks": ticks,
        "setup_seconds": setup,
        "wall_seconds": wall,
        "agent_seconds": batch * ticks * 0.02,
        "spikes": count.tolist(),
        "neurons": brain.n,
        "synaptic_connections": len(brain.weights),
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"benchmark_neural_{device}_{batch}.json").write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)


def physics(worlds=16, steps=1000):
    import warp as wp
    from flygym.warp import GPUSimulation

    from .scene import build

    a = build(1)
    start = time.perf_counter()
    sim = GPUSimulation(a.sim.world, worlds, max_contacts=256, max_constraints=1024)
    sim.warmup()
    with wp.ScopedCapture() as capture:
        for _ in range(10):
            sim.step()
    wp.synchronize()
    setup = time.perf_counter() - start
    start = time.perf_counter()
    for _ in range(steps // 10):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    wall = time.perf_counter() - start
    result = {
        "kind": "physics_hold_only",
        "device": "cuda",
        "worlds": worlds,
        "steps": steps,
        "setup_seconds": setup,
        "wall_seconds": wall,
        "world_seconds": worlds * steps * 0.0001,
        "note": "Physics-only diagnostic, not utility training or closed-loop speed.",
    }
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"benchmark_warp_{worlds}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["neural", "physics"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=8)
    args = p.parse_args()
    if args.kind == "neural":
        neural(args.device, args.batch)
    else:
        physics(args.batch)
