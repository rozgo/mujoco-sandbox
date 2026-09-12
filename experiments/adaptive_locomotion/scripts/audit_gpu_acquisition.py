"""Measure a fresh GPU-trained walker using the existing independent validators."""

import argparse
import json
import time
from pathlib import Path

from adaptive_locomotion.limb_eval import assess
from adaptive_locomotion.stride_eval import inspect_stride


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="healthy")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--seed", type=int, default=9501)
    parser.add_argument("--timestep", type=float, default=0.0005)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing evaluations; use a new output path")
    if args.trials < 1 or "healthy" not in args.cases.split(","):
        raise ValueError("Use positive trials and include the healthy control")
    started = time.perf_counter()
    report = assess(
        args.checkpoint,
        args.output,
        trials=args.trials,
        seed=args.seed,
        cases=args.cases,
        timestep=args.timestep,
    )
    gaits = [report["healthy_gait"]]
    for case in args.cases.split(","):
        if case != "healthy":
            gaits.append(
                inspect_stride(
                    args.checkpoint,
                    trials=args.trials,
                    seed=args.seed,
                    timestep=args.timestep,
                    case=case,
                )
            )
    feet = [
        leg
        for gait in gaits
        for leg in gait["leg_summary"]
        if gait["case"] == "healthy" or leg["leg"].lower() != gait["case"][-2:]
    ]
    trial_feet = [
        leg
        for gait in gaits
        for row in gait["rows"]
        for leg in row["legs"]
        if gait["case"] == "healthy" or leg["leg"].lower() != gait["case"][-2:]
    ]
    checks = {
        "all_tasks_with_allowed_support": all(
            case["completed_with_allowed_support"] == args.trials
            for case in report["cases"]
        ),
        "torque_caps": all(
            row["peak_torque_limit_ratio"] <= 1.0001
            for case in report["cases"]
            for row in case["rows"]
        ),
        "healthy_gait_matches_existing_targets": report["healthy_retained"],
        "each_intact_foot_mean_swing_peak_4cm": all(
            leg["mean_swing_peak_m"] >= 0.04 for leg in feet
        ),
        "each_intact_foot_visible_swing_fraction_80percent": all(
            leg["visible_swing_fraction"] >= 0.8 for leg in feet
        ),
        "every_trial_intact_foot_eight_visible_steps": all(
            leg["visible_swings"] >= 8 for leg in trial_feet
        ),
    }
    report.update(
        evaluation_physics="CPU MuJoCo/mjbatch",
        gait_cases=gaits,
        acquisition_checks=checks,
        acquisition_checks_passed=all(checks.values()),
        evaluation_wall_seconds=time.perf_counter() - started,
        scope="New walking acquisition; basic task passes and gait-quality gates are separate. No weight updates during evaluation.",
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"checks": checks, "seconds": report["evaluation_wall_seconds"]}),
        flush=True,
    )
    return 0 if report["acquisition_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
