"""Frozen selection for the intact-foot clearance follow-up; no final holdout."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.limb_eval import assess
from adaptive_locomotion.limb_loss import LOSS_CASES
from adaptive_locomotion.stride_eval import inspect_stride

PARENT = ROOT / "assets/locomotion/checkpoints/limb_healthy_sequence_90s_seed2.pt"
REAR = ("lower_rl", "lower_rr", "whole_rl", "whole_rr")


def evaluate(path, output, trials=8, seed=9153, timestep=0.002):
    report = assess(path, output, trials=trials, seed=seed, timestep=timestep)
    gaits = [report["healthy_gait"]]
    for case in LOSS_CASES[1:]:
        gait = inspect_stride(
            path, trials=trials, seed=seed, case=case, timestep=timestep
        )
        gait["checkpoint"] = report["checkpoint"]
        gaits.append(gait)
    report["gait_cases"] = gaits
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def intact_mean(gait, key):
    legs = [
        leg
        for leg in gait["leg_summary"]
        if gait["case"] == "healthy" or leg["leg"].lower() != gait["case"][-2:]
    ]
    values = [leg[key] for leg in legs]
    return float(np.mean(values)) if all(v is not None for v in values) else None


def compare(parent, candidate):
    bg = {g["case"]: g for g in parent["gait_cases"]}
    cg = {g["case"]: g for g in candidate["gait_cases"]}
    bm = {c["case"]: c["motion_quality"]["summary"] for c in parent["cases"]}
    cm = {c["case"]: c["motion_quality"]["summary"] for c in candidate["cases"]}
    rear = []
    for case in REAR:
        leg = "RR" if case.endswith("rl") else "RL"
        old = next(l for l in bg[case]["leg_summary"] if l["leg"] == leg)
        new = next(l for l in cg[case]["leg_summary"] if l["leg"] == leg)
        rear.append(
            {
                "case": case,
                "leg": leg,
                "before": old,
                "after": new,
                "clearance_gain_m": new["moving_clearance_m"]
                - old["moving_clearance_m"],
                "drag_reduction_fraction": 1
                - new["drag_travel_m"] / max(old["drag_travel_m"], 1e-8),
            }
        )
    complete = all(
        intact_mean(g, k) is not None
        for g in cg.values()
        for k in ("mean_stride_m", "mean_completed_stance_s", "strides_per_second")
    )
    checks = {
        "all_tasks_support_valid": all(
            c["completed_with_allowed_support"] == candidate["trials_per_case"]
            for c in candidate["cases"]
        ),
        "healthy_retained": candidate["healthy_retained"],
        "cycles_all_intact_legs": complete,
        "rear_clearance_mean_plus_5mm": bool(
            np.mean([r["clearance_gain_m"] for r in rear]) >= 0.005
        ),
        "rear_clearance_each_plus_3mm": all(
            r["clearance_gain_m"] >= 0.003 for r in rear
        ),
        "rear_drag_mean_reduced_25percent": bool(
            np.mean([r["drag_reduction_fraction"] for r in rear]) >= 0.25
        ),
        "rear_drag_no_case_worse": all(r["drag_reduction_fraction"] >= 0 for r in rear),
        "speed_retained": all(
            cm[c]["forward_speed_mps"] >= 0.9 * min(0.55, bm[c]["forward_speed_mps"])
            for c in cg
        ),
        "vertical_motion_retained": all(
            cm[c]["vertical_velocity_rms_mps"]
            <= 1.15 * bm[c]["vertical_velocity_rms_mps"] + 0.01
            for c in cg
        ),
        "airborne_mean_within_1pp": bool(
            np.mean(
                [
                    cm[c]["airborne_above_1mm_fraction"]
                    - bm[c]["airborne_above_1mm_fraction"]
                    for c in LOSS_CASES[1:]
                ]
            )
            <= 0.01
        ),
        "mean_slip_retained": bool(
            np.mean(
                [
                    cg[c]["summary"]["planted_foot_horizontal_speed_rms_mps"]
                    for c in LOSS_CASES[1:]
                ]
            )
            <= 1.15
            * np.mean(
                [
                    bg[c]["summary"]["planted_foot_horizontal_speed_rms_mps"]
                    for c in LOSS_CASES[1:]
                ]
            )
        ),
    }
    checks["stride_and_stance_retained"] = complete and all(
        intact_mean(cg[c], k) >= 0.9 * intact_mean(bg[c], k)
        for c in cg
        for k in ("mean_stride_m", "mean_completed_stance_s")
    )
    return {
        "checkpoint": candidate["checkpoint"],
        "checks": checks,
        "eligible": all(checks.values()),
        "rear_feet": rear,
        "score": float(np.mean([r["clearance_gain_m"] for r in rear])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    cache = run / "parent_dev.json"
    if cache.exists():
        parent = json.loads(cache.read_text())
        assert (
            parent["sha256"] == hashlib.sha256(PARENT.read_bytes()).hexdigest()
            and parent["seed"] == 9153
            and parent["trials_per_case"] == 8
        )
    else:
        parent = evaluate(PARENT, cache)
    if args.baseline_only:
        return
    rows = []
    for name in ("iteration_0050.pt", "iteration_0100.pt", "policy.pt"):
        path = run / name
        if not path.exists():
            continue
        row = compare(parent, evaluate(path, run / f"{path.stem}_dev.json"))
        rows.append(row)
        print("CANDIDATE", json.dumps(row), flush=True)
    eligible = [r for r in rows if r["eligible"]]
    report = {
        "development_seed": 9153,
        "final_seed": 20260921,
        "demo_seed": 9143,
        "parent": str(PARENT.relative_to(ROOT)),
        "candidates": rows,
        "selected": max(eligible, key=lambda r: r["score"]) if eligible else None,
    }
    (run / "selection.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SELECTED", json.dumps(report["selected"]), flush=True)


if __name__ == "__main__":
    main()
