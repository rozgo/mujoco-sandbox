"""Paired physical acceptance and healthy gait gates for damage fine-tuning."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .bodies import ROOT
from .evaluate import rollout
from .stride_eval import inspect_stride
from .train import load_checkpoint

SHORT_CASES = ("short_fl", "short_fr", "short_rl", "short_rr")


def assess(checkpoint, output, trials=16, seed=9137, final=False):
    net, saved = load_checkpoint(checkpoint)
    cases = ["healthy", *SHORT_CASES]
    if final:
        cases += ["unseen_short", "unseen_pair", "unseen_weak"]
    outcomes = []
    for case in cases:
        result, _, _ = rollout(net, case, trials=trials, seed=seed)
        outcomes.append(result)
        print(
            json.dumps(
                {
                    "checkpoint": Path(checkpoint).name,
                    "case": case,
                    "valid": result["completed_with_allowed_support"],
                    "trials": trials,
                    "distance": result["mean_distance_m"],
                }
            ),
            flush=True,
        )
    gait = inspect_stride(checkpoint, trials=trials, seed=seed)
    metrics = gait["summary"]
    # Thresholds frozen in DAMAGE_RETENTION.md before training. Speed reference
    # is the independent healthy-gait measurement already published for the parent.
    gates = {
        "all_healthy_trials_support_valid": outcomes[0][
            "completed_with_allowed_support"
        ]
        == trials,
        "stride_28_to_35cm": metrics["mean_stride_m"] is not None
        and 0.28 <= metrics["mean_stride_m"] <= 0.35,
        "speed_within_10_percent": 0.9 * 0.6092188325
        <= metrics["mean_forward_speed_mps"]
        <= 1.1 * 0.6092188325,
        "left_right_duty_gap_at_most_12pp": metrics["mean_left_right_duty_gap"] <= 0.12,
        "body_bounce_at_most_11mm": metrics["base_height_std_m"] <= 0.011,
    }
    damage = outcomes[1:5]
    report = {
        "checkpoint": str(Path(checkpoint).resolve().relative_to(ROOT)),
        "sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "training_seconds": saved["cumulative_training_seconds"],
        "seed": seed,
        "trials_per_case": trials,
        "split": "final" if final else "development",
        "healthy_gates": gates,
        "healthy_retained": all(gates.values()),
        "short_calf_support_valid_rate": sum(
            r["completed_with_allowed_support"] for r in damage
        )
        / (4 * trials),
        "short_calf_mean_distance_m": float(
            np.mean([r["mean_distance_m"] for r in damage])
        ),
        "healthy_gait": gait,
        "cases": outcomes,
    }
    if final:
        report["half_timestep"] = [
            rollout(net, c, trials=8, seed=seed, timestep=0.001)[0]
            for c in ("healthy", *SHORT_CASES)
        ]
        report["damaged_gaits"] = [
            inspect_stride(checkpoint, trials=trials, seed=seed, case=c)
            for c in SHORT_CASES
        ]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("cases", "healthy_gait", "damaged_gaits", "half_timestep")
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=9137)
    parser.add_argument("--final", action="store_true")
    assess(**vars(parser.parse_args()))
