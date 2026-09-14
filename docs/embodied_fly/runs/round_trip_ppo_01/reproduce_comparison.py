"""Compare frozen parent/child reports; run from repository root after capture."""

import argparse
import json
from pathlib import Path
from statistics import mean


def compare(parent, child):
    assert parent["model_sha256"] == child["model_sha256"]
    assert parent["physical_contract"] == child["physical_contract"]
    assert parent["origins_cm"] == child["origins_cm"]
    assert parent["amplitude_cm"] == child["amplitude_cm"]
    assert (
        parent["simulated_seconds_per_world"]
        == child["simulated_seconds_per_world"]
        == 12
    )
    a, b = parent["cases"], child["cases"]
    assert [c["case"] for c in a] == [c["case"] for c in b]
    old_path = mean(c["whole_path_position_rms_mm"] for c in a[:6])
    new_path = mean(c["whole_path_position_rms_mm"] for c in b[:6])
    old_hover = a[6]["after_first_second_position_rms_mm"]
    new_hover = b[6]["after_first_second_position_rms_mm"]
    path_gain = (old_path - new_path) / old_path
    hover_change = (new_hover - old_hover) / old_hover
    new_failures = [
        y["case"]
        for x, y in zip(a, b)
        if x["first_failure_seconds"] is None and y["first_failure_seconds"] is not None
    ]
    return {
        "parent_checkpoint_sha256": parent["checkpoint_sha256"],
        "child_checkpoint_sha256": child["checkpoint_sha256"],
        "six_route_mean_position_rms_mm": {"parent": old_path, "child": new_path},
        "route_improvement_fraction": path_gain,
        "stationary_hover_position_rms_mm": {"parent": old_hover, "child": new_hover},
        "stationary_hover_regression_fraction": hover_change,
        "new_physical_failures": new_failures,
        "parent_passed_cases": sum(c["passed"] for c in a),
        "child_passed_cases": sum(c["passed"] for c in b),
        "progress_gate": path_gain >= 0.1 and hover_change <= 0.05 and not new_failures,
        "full_acceptance": all(c["passed"] for c in b),
        "gate_definition": "At least 10% route RMS improvement; no more than 5% stationary-hover regression; no new failure. Full acceptance additionally requires all seven waypoint/hold gates.",
        "scope": "Fixed development cases, not independent-seed or generalization evidence",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--parent",
        type=Path,
        default=Path("outputs/embodied_fly/round_trip_parent_review_01/report.json"),
    )
    p.add_argument(
        "--child",
        type=Path,
        default=Path("outputs/embodied_fly/round_trip_ppo_review_01/report.json"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs/embodied_fly/runs/round_trip_ppo_01/comparison.json"),
    )
    args = p.parse_args()
    result = compare(
        json.loads(args.parent.read_text()), json.loads(args.child.read_text())
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
