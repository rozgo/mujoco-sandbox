"""Physical line search along learned full-body decoder weight updates."""

import argparse
import json
import time
from pathlib import Path

import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.full_body_guidance_evaluate import promotion, summarize
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_hover import evaluate_hover


def run(args):
    begin = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.parent, args.graph, device)
    base_state = {k: v.cpu().clone() for k, v in parent["state_dict"].items()}
    metrics = summarize(json.loads((args.parent_capture / "report.json").read_text()))
    report = {
        "provenance": evidence(),
        "parent_checkpoint_sha256": sha256(args.parent),
        "physical_contract": parent["physical_contract"],
        "fractions": args.fractions,
        "additional_optimizer_updates": 0,
        "candidates": [],
    }
    for method, source in (
        ("learned_residual", args.learned),
        ("analytical", args.analytical),
    ):
        if method not in args.methods:
            continue
        source_checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        for fraction in args.fractions:
            state = {
                k: v + fraction * (source_checkpoint["state_dict"][k] - v)
                if k.startswith("motor_decoder.") and v.is_floating_point()
                else v.clone()
                for k, v in base_state.items()
            }
            actor.load_state_dict(state, strict=True)
            if actor.wing_residual is not None:
                raise ValueError("Shared decoder required")
            name = f"{method}_{fraction:.3f}"
            candidate = dict(parent)
            candidate.update(
                state_dict=state,
                parent_checkpoint_sha256=sha256(args.parent),
                method="Full-body decoder weight backtracking after model-guided learning",
                model_guidance=source_checkpoint["model_guidance"]
                | {
                    "weight_fraction": fraction,
                    "source_checkpoint_sha256": sha256(source),
                    "additional_optimizer_updates": 0,
                },
            )
            file = args.output / f"{name}.pt"
            torch.save(candidate, file)
            capture = args.output / name
            result = evaluate_hover(
                actor, candidate, args.dataset, capture, device, refresh_observations=True
            )
            summary = summarize(result)
            gates = promotion(summary, metrics)
            row = {
                "method": method,
                "fraction": fraction,
                "checkpoint": file.as_posix(),
                "checkpoint_sha256": sha256(file),
                "capture": capture.as_posix(),
                "capture_report_sha256": sha256(capture / "report.json"),
                "summary": summary,
                "gates": gates,
                "passed": all(gates.values()),
            }
            report["candidates"].append(row)
            (args.output / "partial_report.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            print(json.dumps(row), flush=True)
    eligible = [c for c in report["candidates"] if c["passed"]]
    report.update(
        selected=min(eligible, key=lambda c: c["summary"]["velocity_rms_mm_s"])
        if eligible
        else None,
        completed_utc=utc_now(),
        wall_seconds=time.perf_counter() - begin,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent",
        "learned",
        "analytical",
        "parent-capture",
        "graph",
        "dataset",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--fractions", type=float, nargs="+", default=[0.1, 0.03, 0.01])
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--methods",
        nargs="+",
        choices=("learned_residual", "analytical"),
        default=["learned_residual", "analytical"],
    )
    run(p.parse_args())
