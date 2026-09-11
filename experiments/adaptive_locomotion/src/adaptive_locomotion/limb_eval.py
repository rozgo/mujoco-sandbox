"""Frozen-seed complete-removal evaluations; preserve every failed trial."""

import argparse
import hashlib
import json
from pathlib import Path

from .bodies import ROOT
from .evaluate import rollout
from .limb_loss import LOSS_CASES
from .stride_eval import inspect_stride
from .train import load_checkpoint


def assess(
    checkpoint, output, trials=8, seed=9141, cases=None, gait=True, timestep=0.002
):
    net, saved = load_checkpoint(checkpoint)
    results = []
    for case in cases.split(",") if cases else LOSS_CASES:
        result, _, _ = rollout(net, case, trials=trials, seed=seed, timestep=timestep)
        results.append(result)
        print(
            json.dumps(
                {
                    "checkpoint": Path(checkpoint).name,
                    "case": case,
                    "valid": result["completed_with_allowed_support"],
                    "survived": result["survived"],
                    "distance": result["mean_distance_m"],
                    "trials": trials,
                }
            ),
            flush=True,
        )
    report = {
        "checkpoint": str(Path(checkpoint).resolve().relative_to(ROOT)),
        "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "cumulative_training_seconds": saved["cumulative_training_seconds"],
        "seed": seed,
        "physics_timestep_s": timestep,
        "trials_per_case": trials,
        "cases": results,
    }
    if gait:
        measured = inspect_stride(
            checkpoint, trials=trials, seed=seed, timestep=timestep
        )
        m = measured["summary"]
        gates = {
            "all_healthy_support_valid": next(
                r for r in results if r["case"] == "healthy"
            )["completed_with_allowed_support"]
            == trials,
            "stride_28_to_35cm": m["mean_stride_m"] is not None
            and 0.28 <= m["mean_stride_m"] <= 0.35,
            "speed_within_10_percent": 0.9 * 0.6092188325
            <= m["mean_forward_speed_mps"]
            <= 1.1 * 0.6092188325,
            "duty_gap_at_most_12pp": m["mean_left_right_duty_gap"] <= 0.12,
            "bounce_at_most_11mm": m["base_height_std_m"] <= 0.011,
        }
        measured["checkpoint"] = report["checkpoint"]
        report.update(
            healthy_gait=measured,
            healthy_gates=gates,
            healthy_retained=all(gates.values()),
        )
        print(
            json.dumps(
                {
                    "healthy_retained": report["healthy_retained"],
                    "gates": gates,
                    "gait": m,
                }
            ),
            flush=True,
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=9141)
    p.add_argument("--cases")
    assess(**vars(p.parse_args()))
