"""Paired scenario evaluation with retained failures and an explicit neural ablation."""

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .paths import OUTPUTS
from .utility import handcrafted_weights


def run_case(job):
    from .environment import Habitat

    method, weights, seed, seconds, n_flies, record, neural, device = job
    env = Habitat(
        n_flies,
        seed,
        weights=weights,
        neural=neural,
        device=device,
        vision=True,
        random_spawn=True,
        record=record,
    )
    began = time.perf_counter()
    try:
        report = env.run(seconds)
        report["valid"] = True
    except RuntimeError as error:
        report = env.report()
        report["valid"] = False
        report["error"] = str(error)
    if record:
        env.save(f"evaluation/{method}_{seed}")
    else:
        env.sensors.close()
    report["method"] = method
    report["evaluation_wall_seconds"] = time.perf_counter() - began
    report["physical_out_of_bounds"] = bool(
        np.any(np.abs(env.previous_xyz[:, :2]) > [27, 19])
    )
    report["max_final_height_mm"] = float(env.previous_xyz[:, 2].max())
    return report


def evaluate(
    checkpoint,
    seeds=(3100, 3101, 3102, 3103),
    seconds=16.0,
    flies=1,
    workers=8,
    name="heldout_01",
    device="cpu",
):
    root = OUTPUTS / name
    if root.exists():
        raise ValueError(f"Refusing to overwrite {root}")
    root.mkdir(parents=True)
    weights = np.load(checkpoint)["weights"]
    methods = [
        ("untrained", np.zeros_like(weights), True),
        ("handcrafted", handcrafted_weights(), True),
        ("learned", weights, True),
        ("learned_without_neural", weights, False),
    ]
    jobs = [
        (method, w, int(seed), seconds, flies, False, neural, device)
        for method, w, neural in methods
        for seed in seeds
    ]
    start = time.perf_counter()
    if workers == 1:
        results = [run_case(job) for job in jobs]
    else:
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(run_case, jobs, chunksize=1))
    summary = {}
    for method, _, _ in methods:
        rows = [r for r in results if r["method"] == method]
        summary[method] = {
            "mean_reward": float(np.mean([r["reward"] for r in rows])),
            "survivors": sum(r["alive"] for r in rows),
            "agents_tested": len(rows) * flies,
            "numerical_failures": sum(not r["valid"] for r in rows),
            "out_of_bounds_trials": sum(r["physical_out_of_bounds"] for r in rows),
            "food_intake": sum(n["food_intake"] for r in rows for n in r["needs"]),
            "water_intake": sum(n["water_intake"] for r in rows for n in r["needs"]),
            "action_seconds": np.sum(
                [np.sum(r["action_seconds"], axis=0) for r in rows], axis=0
            ).tolist(),
        }
    report = {
        "seeds": seeds,
        "seconds_per_trial": seconds,
        "flies_per_world": flies,
        "workers": workers,
        "neural_device": device,
        "physics": "native MuJoCo CPU",
        "wall_seconds": time.perf_counter() - start,
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "summary": summary,
        "trials": results,
        "interpretation": "Same utility weights across methods learned/ablation; ablation removes the neural readout and steering contribution. Utility training is separate from this measurement.",
    }
    (root / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "trials"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--seeds", type=int, nargs="+", default=[3100, 3101, 3102, 3103])
    p.add_argument("--seconds", type=float, default=16)
    p.add_argument("--flies", type=int, default=1)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default="heldout_01")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    evaluate(a.checkpoint, a.seeds, a.seconds, a.flies, a.workers, a.name, a.device)
