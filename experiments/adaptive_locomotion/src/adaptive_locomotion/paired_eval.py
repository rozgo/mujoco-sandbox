"""Evaluate paired-damage progress with healthy and single-damage retention gates."""

import argparse
import json
from pathlib import Path

from .evaluate import rollout
from .paired import PAIR_CASES, training_pairs
from .retention_eval import assess
from .stride_eval import inspect_stride
from .train import load_checkpoint


def assess_pairs(checkpoint, output, trials=16, seed=9139, final=False):
    report = assess(checkpoint, output, trials=trials, seed=seed)
    net, _ = load_checkpoint(checkpoint)
    names = list(PAIR_CASES) if final else [b.name for b in training_pairs("hard")]
    pairs = []
    for case in [*names, "unseen_weak"]:
        result, _, _ = rollout(net, case, trials=trials, seed=seed)
        pairs.append(result)
        print(
            json.dumps(
                {
                    "case": case,
                    "valid": result["completed_with_allowed_support"],
                    "trials": trials,
                    "distance": result["mean_distance_m"],
                }
            ),
            flush=True,
        )
    report["split"] = "final" if final else "development"
    report["paired_cases"] = pairs[:-1]
    report["motor_retention"] = pairs[-1]
    report["single_damage_retained"] = report["short_calf_support_valid_rate"] == 1.0
    report["motor_skill_retained"] = (
        pairs[-1]["completed_with_allowed_support"] == trials
    )
    report["all_previous_skills_retained"] = all(
        report[k]
        for k in ("healthy_retained", "single_damage_retained", "motor_skill_retained")
    )
    report["trained_pair_support_valid_rate"] = sum(
        p["completed_with_allowed_support"] for p in pairs[:4]
    ) / (4 * trials)
    report["target_pair_support_valid_rate"] = (
        pairs[0]["completed_with_allowed_support"] / trials
    )
    if final:
        report["half_timestep"] = [
            rollout(net, c, trials=8, seed=seed, timestep=0.001)[0]
            for c in ("healthy", "short_fr", "pair_fl_rr_hard")
        ]
        report["target_pair_gait"] = inspect_stride(
            checkpoint, trials=trials, seed=seed, case="pair_fl_rr_hard"
        )
    Path(output).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "checkpoint",
                    "all_previous_skills_retained",
                    "trained_pair_support_valid_rate",
                    "target_pair_support_valid_rate",
                )
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
    parser.add_argument("--seed", type=int, default=9139)
    parser.add_argument("--final", action="store_true")
    assess_pairs(**vars(parser.parse_args()))
