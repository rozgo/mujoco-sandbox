"""Separate cold startup from later hover using saved physical captures only."""

import argparse
import json
from pathlib import Path

import numpy as np

from embodied_fly.provenance import sha256, utc_now
from embodied_fly.velocity_exercise import rolling_velocity


def flight_windows(arrays, failure_seconds):
    stop = len(arrays["time"])
    if failure_seconds is not None:
        stop = min(stop, max(1, round(failure_seconds * 500)))
    velocity = rolling_velocity(arrays["measured_velocity"][:stop] * 10)
    heights = arrays["post_position"][:stop, 2] * 10
    initial = float(arrays["qpos"][0, 2] * 10)
    result = {"initial_height_mm": initial, "failure_seconds": failure_seconds}
    for label, a, b in (("startup_0_2", 0, 1000), ("settled_2_10", 1000, 5000)):
        complete = stop >= b
        if not complete:
            result[label] = {"complete": False}
            continue
        sample = velocity[a:b]
        start_height = initial if a == 0 else float(heights[a - 1])
        result[label] = {
            "complete": True,
            "minimum_height_mm": float(heights[a:b].min()),
            "height_change_mm": float(heights[b - 1] - start_height),
            "mean_height_rate_mm_s": float(
                (heights[b - 1] - start_height) / ((b - a) * 0.002)
            ),
            "mean_velocity_mm_s": sample.mean(0).tolist(),
            "velocity_rms_mm_s": float(np.sqrt(np.mean(np.sum(sample**2, axis=1)))),
            "horizontal_rms_mm_s": float(np.sqrt(np.mean(np.sum(sample[:, :2] ** 2, axis=1)))),
            "vertical_rms_mm_s": float(np.sqrt(np.mean(sample[:, 2] ** 2))),
        }
    return result


def analyze(run, labels):
    report = {
        "completed_utc": utc_now(),
        "scope": "Read-only physical captures; no new reward, control, or filtering in the force law",
        "labels": {},
    }
    for label in labels:
        source = run / label / "report.json"
        evaluation = json.loads(source.read_text())
        cases = []
        for case in evaluation["cases"]:
            file = source.parent / case["file"]
            if sha256(file) != case["sha256"]:
                raise ValueError("Physical capture checksum differs")
            with np.load(file) as arrays:
                cases.append(
                    {
                        "episode": case["episode"],
                        **flight_windows(arrays, case["first_failure_seconds"]),
                    }
                )
        means = {}
        for phase in ("startup_0_2", "settled_2_10"):
            complete = [c[phase] for c in cases if c[phase]["complete"]]
            means[phase] = {"complete_cases": len(complete), "attempted_cases": len(cases)}
            # Only publish a matched all-case mean when every case completes the window.
            if len(complete) == len(cases):
                means[phase].update(
                    {
                        key: float(np.mean([c[key] for c in complete]))
                        for key in (
                            "minimum_height_mm",
                            "height_change_mm",
                            "mean_height_rate_mm_s",
                            "velocity_rms_mm_s",
                            "horizontal_rms_mm_s",
                            "vertical_rms_mm_s",
                        )
                    }
                )
        report["labels"][label] = {
            "evaluation_report_sha256": sha256(source),
            "cases": cases,
            "all_case_means": means,
        }
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=["parent", "quarter", "midpoint", "final"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous diagnostics")
    report = analyze(args.run, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["all_case_means"] for k, v in report["labels"].items()}))
