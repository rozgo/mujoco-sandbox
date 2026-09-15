"""Judge shared-decoder model guidance by complete live MaleCNS/MuJoCo flights."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_hover import evaluate_hover


def summarize(report):
    cases = report["cases"]
    return {
        "complete": sum(c["survived_ten_seconds"] for c in cases),
        "velocity_rms_mm_s": float(np.mean([c["velocity_rms_mm_s"] for c in cases])),
        "horizontal_rms_mm_s": float(
            np.mean([c["horizontal_velocity_rms_mm_s"] for c in cases])
        ),
        "vertical_rms_mm_s": float(np.mean([c["vertical_velocity_rms_mm_s"] for c in cases])),
        "net_climb_mm": float(np.mean([c["final_displacement_mm"][2] for c in cases])),
        "absolute_net_climb_mm": float(
            np.mean([abs(c["final_displacement_mm"][2]) for c in cases])
        ),
        "early_velocity_rms_mm_s": float(
            np.mean([c["first_two_seconds_velocity_rms_mm_s"] for c in cases])
        ),
        "airborne_seconds": [c["airborne_seconds"] for c in cases],
    }


def promotion(candidate, parent):
    return {
        "four_complete_flights": candidate["complete"] == 4,
        "velocity_at_least_five_percent_better": candidate["velocity_rms_mm_s"]
        <= 0.95 * parent["velocity_rms_mm_s"],
        "climb_at_least_five_percent_better": candidate["absolute_net_climb_mm"]
        <= 0.95 * parent["absolute_net_climb_mm"],
        "horizontal_regression_at_most_five_percent": candidate["horizontal_rms_mm_s"]
        <= 1.05 * parent["horizontal_rms_mm_s"],
    }


def run(args):
    start = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    parent = json.loads((args.parent_capture / "report.json").read_text())
    if parent["observation_timing"] != "refresh before actor":
        raise ValueError("Evaluation must match the parent's observation timing")
    parent_metrics = summarize(parent)
    report = {
        "provenance": evidence(),
        "physical_contract": parent["physical_contract"],
        "parent": {
            "capture": args.parent_capture.as_posix(),
            "report_sha256": sha256(args.parent_capture / "report.json"),
            "summary": parent_metrics,
        },
        "candidates": [],
    }
    actor = None
    for method, folder in (
        ("learned_residual", args.learned),
        ("analytical", args.analytical),
    ):
        training = json.loads((folder / "report.json").read_text())
        if training["physical_contract"] != parent["physical_contract"]:
            raise ValueError("Candidate physics changed")
        for snapshot in training["snapshots"]:
            checkpoint = folder / snapshot["file"]
            if sha256(checkpoint) != snapshot["sha256"]:
                raise ValueError("Candidate checkpoint checksum mismatch")
            if actor is None:
                actor, saved = load_actor(checkpoint, args.graph, device)
            else:
                saved = torch.load(checkpoint, map_location=device, weights_only=False)
                actor.load_state_dict(saved["state_dict"], strict=True)
            if actor.wing_residual is not None:
                raise ValueError("No separate wing decoder in current candidates")
            target = args.output / f"{method}_{snapshot['step']:04d}"
            result = evaluate_hover(
                actor, saved, args.dataset, target, device, refresh_observations=True
            )
            metrics = summarize(result)
            gates = promotion(metrics, parent_metrics)
            row = {
                "method": method,
                "step": snapshot["step"],
                "checkpoint": checkpoint.as_posix(),
                "checkpoint_sha256": snapshot["sha256"],
                "training_seconds": snapshot["training_seconds"],
                "capture": target.as_posix(),
                "capture_report_sha256": sha256(target / "report.json"),
                "summary": metrics,
                "gates": gates,
                "passed": all(gates.values()),
            }
            report["candidates"].append(row)
            (args.output / "partial_report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)
    eligible = [c for c in report["candidates"] if c["passed"]]
    report["selected"] = (
        min(eligible, key=lambda c: c["summary"]["velocity_rms_mm_s"]) if eligible else None
    )
    report["completed_utc"] = utc_now()
    report["wall_seconds"] = time.perf_counter() - start
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--learned", type=Path, required=True)
    p.add_argument("--analytical", type=Path, required=True)
    p.add_argument("--parent-capture", type=Path, required=True)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
