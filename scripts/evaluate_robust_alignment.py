"""Evaluate alignment policies against disturbance level and record training compute.

Policies: the robust policy trained under the disturbance layer, the policy
trained without disturbances (align_v3) and the scripted yardstick, each on the
200 predetermined evaluation seeds at fixed disturbance levels. Training curves
and compute come from the PufferLib dashboard logs and resource samples copied
from the GPU host into outputs/neural_insertion/puffer. Writes
docs/neural_insertion/ROBUST_ALIGN_RESULTS.json.

Usage: uv run --locked python scripts/evaluate_robust_alignment.py [--robust WEIGHTS] [--output PATH]
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from sixlegs.neural_insertion.align_env import EVALUATION_SEEDS, NATIVE, evaluate, scripted  # noqa: E402
from sixlegs.neural_insertion.align_policy import evaluate_checkpoint  # noqa: E402
from sixlegs.neural_insertion.training_log import boxes, history, summary  # noqa: E402

LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0)  # scale v2: 1 nominal, 2 stress
LOGS = ROOT/"outputs/neural_insertion/puffer"
# Learned checkpoints evaluated, by name: the final policy of each completed run.
WEIGHTS = {"robust": ROOT/"assets/neural_insertion/align_v6_policy.bin",
           "run5": ROOT/"assets/neural_insertion/align_v5_policy.bin",
           "undisturbed": ROOT/"assets/neural_insertion/align_v3_policy.bin"}
# Run -> (dashboard log, resource samples, what changed, outcome). Runs 1-3 predate the logger.
RUNS = {
    "align_v1": ("align_v1_diverged.log", None, "first run, no disturbances",
                 "stopped at 22.2 M steps: reward used unclipped actions"),
    "align_v2": ("align_v2.log", None, "reward fix, learning rate 0.015", "stopped at 21.2 M steps: KL divergence"),
    "align_v3": ("align_v3.log", None, "learning rate 0.003", "completed; 100% success without disturbances"),
    "align_v4": ("align_v4.log", "align_v4_resources.csv", "disturbance layer on, level drawn in [0, 1] per episode",
                 "stopped at 48.5 M steps: KL rose from about 0.1 to 14 and collisions returned"),
    "align_v5": ("align_v5.log", "align_v5_resources.csv", "learning rate 0.001",
                 "completed 99.9 M steps; stable, still improving when the learning rate reached zero"),
    "align_v6": ("align_v6.log", "align_v6_resources.csv", "300 M step budget", None),
}
HARDWARE = {"gpu": "NVIDIA GeForce RTX 4090, 24 GiB", "driver": "595.91.07", "cuda": "13.0",
            "cpu": "AMD Ryzen 9 5950X, 16 cores / 32 threads", "host_memory_gib": 62,
            "trainer_threads": 28, "note": "MuJoCo physics runs on CPU threads; the GPU runs the network and PPO update"}


def text(path):
    return path.read_text().strip() if path.exists() else None


def condense(summary_):
    keep = ("success", "collision", "timeout", "vessel_steps", "final_lateral_um", "final_vertical_um",
            "episode_return", "episode_length", "episodes", "mean_success_time_s", "max_success_lateral_um")
    return {k: summary_.get(k) for k in keep}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robust", type=Path, default=WEIGHTS["robust"])
    parser.add_argument("--output", type=Path, default=ROOT/"docs/neural_insertion/ROBUST_ALIGN_RESULTS.json")
    args = parser.parse_args()
    WEIGHTS["robust"] = args.robust
    report = {"description": "Robust alignment: training under the disturbance layer, compute per run, and success "
                             "against disturbance level on the 200 predetermined evaluation seeds.",
              "pufferlib_revision": "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2", "hardware": HARDWARE,
              "core_sha256": hashlib.sha256((NATIVE/"surgical_core.h").read_bytes()).hexdigest(),
              "evaluation_seeds": [EVALUATION_SEEDS[0], EVALUATION_SEEDS[-1]], "levels": list(LEVELS),
              "weights_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in WEIGHTS.items() if v.exists()},
              "runs": {}, "compute": {}, "evaluation": {}}
    for run, (log, res, change, outcome) in RUNS.items():
        path = LOGS/log
        if not path.exists():
            print("missing", path)
            continue
        rows = boxes(path.read_text(errors="replace"))
        report["compute"][run] = summary(rows, LOGS/res if res else None)
        report["runs"][run] = {"change": change, "outcome": outcome or "completed",
                                "commit": text(LOGS/f"{run}.commit"), "started_utc": text(LOGS/f"{run}.start"),
                                "stopped_utc": text(LOGS/f"{run}.end"), "history": history(rows)}
    scratch = ROOT/"outputs/neural_insertion/robust_eval"
    for level in LEVELS:
        for name, weights in WEIGHTS.items():
            if not weights.exists():
                print("missing", weights)
                continue
            s, _ = evaluate_checkpoint(weights, scratch/f"{name}_level{level:.2f}", level=level)
            report["evaluation"].setdefault(name, {})[f"{level:.2f}"] = condense(s)
        s, _ = evaluate(scripted, level=level)
        report["evaluation"].setdefault("scripted", {})[f"{level:.2f}"] = condense(s)
        print(level, {k: v[f"{level:.2f}"]["success"] for k, v in report["evaluation"].items()}, flush=True)
    args.output.write_text(json.dumps(report, indent=1)+"\n")
    print("wrote", args.output)


if __name__ == "__main__":
    main()
