"""CEM learns one utility matrix from full embodied episode outcomes."""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime

import numpy as np

from .paths import OUTPUTS
from .utility import ACTIONS, FEATURES, handcrafted_weights


def initialize(ready):
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    from .environment import Habitat

    env = Habitat(1, seed=999, vision=True, neural=True)
    env.step()
    env.sensors.close()
    ready.put(os.getpid())


def evaluate(job):
    from .environment import Habitat

    weights, seed, seconds, n_flies = job
    start = time.perf_counter()
    env = Habitat(
        n_flies, seed, weights=weights, neural=True, vision=True, random_spawn=True
    )
    try:
        report = env.run(seconds)
        report = {
            k: report[k]
            for k in (
                "seed",
                "n_flies",
                "reward",
                "alive",
                "needs",
                "resources",
                "events",
                "action_seconds",
                "warnings",
            )
        }
        report["valid"] = True
    except RuntimeError as exc:
        report = {
            "seed": seed,
            "n_flies": n_flies,
            "reward": -100.0,
            "alive": 0,
            "valid": False,
            "error": str(exc),
        }
    finally:
        env.sensors.close()
    report["wall_seconds"] = time.perf_counter() - start
    return report


def train(
    seconds=300,
    workers=12,
    population=12,
    episode_seconds=10.0,
    n_flies=1,
    name="cem_01",
    max_generations=20,
):
    root = OUTPUTS / name
    if root.exists():
        raise ValueError(f"Refusing to overwrite existing training run {root}")
    root.mkdir(parents=True)
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    config = {
        "algorithm": "diagonal CEM",
        "seed": 4701,
        "workers": workers,
        "population": population,
        "elites": max(3, population // 4),
        "episode_seconds": episode_seconds,
        "flies_per_world": n_flies,
        "paired_scenarios_per_candidate": 2,
        "training_budget_seconds": seconds,
        "max_generations": max_generations,
        "physics": "native MuJoCo CPU",
        "neural": "MaleCNS CPU per worker",
        "vision": True,
        "parameters": 72,
        "features": FEATURES,
        "actions": ACTIONS,
        "source_commit": source,
        "started_utc": datetime.now(UTC).isoformat(),
        "initialization": "handcrafted weights + Gaussian search noise",
        "note": "Utility parameters learn; anatomy, walking skill and connectome weights stay frozen.",
    }
    (root / "config.json").write_text(json.dumps(config, indent=2))
    rng = np.random.default_rng(config["seed"])
    mean = handcrafted_weights().astype(float)
    std = np.full_like(mean, 0.8)
    history = []
    context = mp.get_context("spawn")
    ready = context.Queue()
    setup_start = time.perf_counter()
    with ProcessPoolExecutor(
        workers, mp_context=context, initializer=initialize, initargs=(ready,)
    ) as pool:
        warm = [pool.submit(os.getpid) for _ in range(workers)]
        ready_pids = [ready.get(timeout=180) for _ in range(workers)]
        for f in warm:
            f.result()
        setup = time.perf_counter() - setup_start
        print(
            json.dumps(
                {
                    "phase": "ready",
                    "workers": len(set(ready_pids)),
                    "setup_seconds": setup,
                }
            ),
            flush=True,
        )
        start = time.perf_counter()
        generation = 0
        while time.perf_counter() - start < seconds and generation < max_generations:
            began = time.perf_counter()
            candidates = rng.normal(mean, std, size=(population, *mean.shape))
            candidates[0] = mean
            seeds = [1100 + generation * 2, 1101 + generation * 2]
            jobs = [
                (w, int(seed), episode_seconds, n_flies)
                for w in candidates
                for seed in seeds
            ]
            results = list(pool.map(evaluate, jobs, chunksize=1))
            scores = (
                np.array([r["reward"] for r in results])
                .reshape(population, 2)
                .mean(axis=1)
            )
            elite_indices = np.argsort(scores)[-config["elites"] :]
            elites = candidates[elite_indices]
            mean = 0.3 * mean + 0.7 * elites.mean(axis=0)
            std = np.maximum(0.12, 0.3 * std + 0.7 * elites.std(axis=0))
            record = {
                "generation": generation,
                "seeds": seeds,
                "mean_score": float(scores.mean()),
                "best_score": float(scores.max()),
                "center_score": float(scores[0]),
                "scores": scores.tolist(),
                "generation_seconds": time.perf_counter() - began,
                "training_seconds": time.perf_counter() - start,
                "numerical_failures": sum(not r["valid"] for r in results),
                "episodes": results,
            }
            history.append(record)
            np.savez_compressed(
                root / f"generation_{generation:02}.npz",
                weights=mean,
                std=std,
                candidates=candidates,
                scores=scores,
            )
            (root / f"generation_{generation:02}.json").write_text(
                json.dumps(record, indent=2)
            )
            print(
                json.dumps(
                    {k: v for k, v in record.items() if k not in ("episodes", "scores")}
                ),
                flush=True,
            )
            generation += 1
        training = time.perf_counter() - start
    checkpoint = root / "utility.npz"
    np.savez_compressed(checkpoint, weights=mean, features=FEATURES, actions=ACTIONS)
    episodes = sum(len(r["episodes"]) for r in history)
    report = {
        "config": config,
        "setup_seconds": setup,
        "training_seconds": training,
        "generations": generation,
        "episodes": episodes,
        "aggregate_simulated_seconds": episodes * episode_seconds * n_flies,
        "utility_transitions": round(episodes * episode_seconds * n_flies / 0.02),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "completed_utc": datetime.now(UTC).isoformat(),
        "numerical_failures": sum(r["numerical_failures"] for r in history),
    }
    (root / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--population", type=int, default=12)
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--flies", type=int, default=1)
    p.add_argument("--name", default="cem_01")
    p.add_argument("--generations", type=int, default=20)
    a = p.parse_args()
    train(
        a.seconds,
        a.workers,
        a.population,
        a.episode_seconds,
        a.flies,
        a.name,
        a.generations,
    )
