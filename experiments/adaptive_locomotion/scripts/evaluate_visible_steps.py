"""Predeclared all-feet swing gates. No candidate accepted on pooled lift alone."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from adaptive_locomotion.bodies import ROOT
from evaluate_clearance import evaluate, intact_mean

PARENT = ROOT / "assets/locomotion/checkpoints/limb_clearance_candidate_seed2.pt"


def compare(parent, candidate):
    bg = {g["case"]: g for g in parent["gait_cases"]}
    cg = {g["case"]: g for g in candidate["gait_cases"]}
    bm = {c["case"]: c["motion_quality"]["summary"] for c in parent["cases"]}
    cm = {c["case"]: c["motion_quality"]["summary"] for c in candidate["cases"]}
    feet = [
        {"case": c, **leg}
        for c, g in cg.items()
        for leg in g["leg_summary"]
        if c == "healthy" or leg["leg"].lower() != c[-2:]
    ]
    all_rows = [
        leg
        for c, g in cg.items()
        for row in g["rows"]
        for leg in row["legs"]
        if c == "healthy" or leg["leg"].lower() != c[-2:]
    ]
    checks = {
        "all_tasks_support_valid": all(
            c["completed_with_allowed_support"] == candidate["trials_per_case"]
            for c in candidate["cases"]
        ),
        "healthy_retained": candidate["healthy_retained"],
        "each_foot_mean_swing_peak_4cm": all(
            l["mean_swing_peak_m"] >= 0.04 for l in feet
        ),
        "each_foot_visible_swing_fraction_80percent": all(
            l["visible_swing_fraction"] >= 0.8 for l in feet
        ),
        "every_trial_foot_eight_visible_steps": all(
            l["visible_swings"] >= 8 for l in all_rows
        ),
        "stride_stance_retained": all(
            intact_mean(cg[c], k) is not None
            and intact_mean(cg[c], k) >= 0.9 * intact_mean(bg[c], k)
            for c in cg
            for k in ("mean_stride_m", "mean_completed_stance_s")
        ),
        "speed_retained": all(
            cm[c]["forward_speed_mps"] >= 0.9 * min(0.55, bm[c]["forward_speed_mps"])
            for c in cg
        ),
        "vertical_motion_retained": all(
            cm[c]["vertical_velocity_rms_mps"]
            <= 1.2 * bm[c]["vertical_velocity_rms_mps"] + 0.01
            for c in cg
        ),
        "airborne_retained": bool(
            np.mean(
                [
                    cm[c]["airborne_above_1mm_fraction"]
                    - bm[c]["airborne_above_1mm_fraction"]
                    for c in cg
                    if c != "healthy"
                ]
            )
            <= 0.015
        ),
    }
    return {
        "checkpoint": candidate["checkpoint"],
        "sha256": candidate["sha256"],
        "checks": checks,
        "eligible": all(checks.values()),
        "feet": feet,
        "score": min(l["visible_swing_fraction"] for l in feet),
        "minimum_mean_swing_peak_m": min(l["mean_swing_peak_m"] for l in feet),
        "mean_visible_swing_fraction": float(
            np.mean([l["visible_swing_fraction"] for l in feet])
        ),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run", type=Path)
    p.add_argument("--names", default="iteration_0100.pt,iteration_0200.pt,policy.pt")
    args = p.parse_args()
    run = args.run.resolve()
    cache = run / "parent_dev.json"
    if cache.exists():
        parent = json.loads(cache.read_text())
        assert (
            parent["sha256"] == hashlib.sha256(PARENT.read_bytes()).hexdigest()
            and parent["seed"] == 9157
        )
    else:
        parent = evaluate(PARENT, cache, trials=8, seed=9157)
    rows = []
    for name in args.names.split(","):
        path = run / name
        if not path.exists():
            continue
        report = evaluate(path, run / f"{path.stem}_dev.json", trials=8, seed=9157)
        row = compare(parent, report)
        rows.append(row)
        print(
            "CANDIDATE",
            json.dumps({k: v for k, v in row.items() if k != "feet"}),
            flush=True,
        )
    eligible = [r for r in rows if r["eligible"]]
    selected = max(eligible, key=lambda r: r["score"]) if eligible else None
    (run / "visible_selection.json").write_text(
        json.dumps({"candidates": rows, "selected": selected}, indent=2) + "\n"
    )
    print("SELECTED", selected["checkpoint"] if selected else None, flush=True)
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
