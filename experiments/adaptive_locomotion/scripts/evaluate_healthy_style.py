"""Frozen development selection for the healthy-style experiments.

Run with the isolated locomotion uv environment. Final holdout is not evaluated.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.limb_eval import assess
from adaptive_locomotion.limb_loss import LOSS_CASES
from adaptive_locomotion.stride_eval import inspect_stride

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "run", type=Path, help="Training output directory containing saved candidates"
)
run = parser.parse_args().run.resolve()
parent = ROOT / "assets/locomotion/checkpoints/limb_ground_support_selected_seed2.pt"
teacher = ROOT / "assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt"


def summarize(r):
    legs = [
        v
        for v in r["leg_summary"]
        if r["case"] == "healthy" or v["leg"].lower() != r["case"][-2:]
    ]
    keys = [
        "mean_completed_stance_s",
        "mean_completed_swing_s",
        "mean_stride_m",
        "strides_per_second",
    ]
    return {
        k: float(np.mean([v[k] for v in legs]))
        if all(v[k] is not None for v in legs)
        else None
        for k in keys
    }


def run_eval(path, out):
    r = assess(path, out, trials=8, seed=9149)
    gaits = [r["healthy_gait"]]
    for case in LOSS_CASES[1:]:
        g = inspect_stride(path, trials=8, seed=9149, case=case)
        g["checkpoint"] = r["checkpoint"]
        gaits.append(g)
    r["gait_cases"] = gaits
    r["intact_leg_gait_means"] = {g["case"]: summarize(g) for g in gaits}
    out.write_text(json.dumps(r, indent=2) + "\n")
    print("GAITS", path.name, json.dumps(r["intact_leg_gait_means"]), flush=True)
    return r


reference = inspect_stride(teacher, trials=8, seed=9149)
reference["checkpoint"] = str(teacher.relative_to(ROOT))
(run / "healthy_reference_gait.json").write_text(json.dumps(reference, indent=2) + "\n")
healthy = summarize(reference)
cached = ROOT / "docs/locomotion/runs/limb_healthy_sequence_90s_seed2_parent_dev.json"
if cached.exists() and cached.parent != run:
    import hashlib

    baseline = json.loads(cached.read_text())
    assert (
        baseline["sha256"] == hashlib.sha256(parent.read_bytes()).hexdigest()
        and baseline["seed"] == 9149
        and baseline["trials_per_case"] == 8
    )
    (run / "parent_dev.json").write_text(json.dumps(baseline, indent=2) + "\n")
else:
    baseline = run_eval(parent, run / "parent_dev.json")
base = {c["case"]: c["motion_quality"]["summary"] for c in baseline["cases"]}
bg = baseline["intact_leg_gait_means"]
bs = {g["case"]: g["summary"] for g in baseline["gait_cases"]}
rows = []
for name in ("iteration_0050.pt", "iteration_0100.pt", "policy.pt"):
    path = run / name
    if not path.exists():
        continue
    r = run_eval(path, run / (path.stem + "_dev.json"))
    data = {
        c["case"]: c["motion_quality"]["summary"]
        for c in r["cases"]
        if c["case"] != "healthy"
    }
    gait = r["intact_leg_gait_means"]
    gs = {g["case"]: g["summary"] for g in r["gait_cases"]}
    allcycles = all(gait[c][key] is not None for c in data for key in healthy)
    row = {
        "checkpoint": r["checkpoint"],
        "checks": {
            "all_72_valid": all(
                c["completed_with_allowed_support"] == 8 for c in r["cases"]
            ),
            "healthy_retained": r["healthy_retained"],
            "completed_cycles_all_intact_legs": allcycles,
        },
        "gait_means": {
            k: float(np.mean([gait[c][k] for c in data])) if allcycles else None
            for k in healthy
        },
    }
    if allcycles:
        changes = {}
        for key in ("mean_completed_stance_s", "mean_stride_m"):
            old = float(np.mean([abs(bg[c][key] - healthy[key]) for c in data]))
            new = float(np.mean([abs(gait[c][key] - healthy[key]) for c in data]))
            changes[key] = {
                "old_distance": old,
                "new_distance": new,
                "fraction_closer": 1 - new / old,
            }
        row["distance_to_healthy"] = changes
        row["checks"].update(
            stance_at_least_20_percent_closer=changes["mean_completed_stance_s"][
                "fraction_closer"
            ]
            >= 0.2,
            stride_at_least_20_percent_closer=changes["mean_stride_m"][
                "fraction_closer"
            ]
            >= 0.2,
            no_body_stride_loss_over_10_percent=all(
                gait[c]["mean_stride_m"] >= 0.9 * bg[c]["mean_stride_m"] for c in data
            ),
            slip_retained=bool(
                np.mean([gs[c]["planted_foot_horizontal_speed_rms_mps"] for c in data])
                <= 1.15
                * np.mean(
                    [bs[c]["planted_foot_horizontal_speed_rms_mps"] for c in data]
                )
            ),
            per_body_speed_retained=all(
                m["forward_speed_mps"] >= 0.9 * min(0.55, base[c]["forward_speed_mps"])
                for c, m in data.items()
            ),
            vertical_velocity_not_worse=bool(
                np.mean([m["vertical_velocity_rms_mps"] for m in data.values()])
                <= np.mean([base[c]["vertical_velocity_rms_mps"] for c in data])
            ),
            airborne_not_worse=bool(
                np.mean([m["airborne_above_1mm_fraction"] for m in data.values()])
                <= np.mean([base[c]["airborne_above_1mm_fraction"] for c in data])
            ),
        )
        row["score"] = np.mean([v["fraction_closer"] for v in changes.values()])
    row["eligible"] = all(row["checks"].values())
    rows.append(row)
    print("CANDIDATE", json.dumps(row), flush=True)
eligible = [r for r in rows if r["eligible"]]
selection = {
    "development_seed": 9149,
    "final_seed": 20260920,
    "demo_seed": 9143,
    "healthy_target": healthy,
    "candidates": rows,
    "selected": max(eligible, key=lambda r: r["score"]) if eligible else None,
}
(run / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
print("SELECTION", json.dumps(selection), flush=True)
