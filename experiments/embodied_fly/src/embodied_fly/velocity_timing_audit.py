"""Frozen-weight comparison of observer refresh and PPO collection sensor timing."""

import argparse
import json
import time
from pathlib import Path

import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_hover import evaluate_hover


def diagnose(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    trained = json.loads((args.run / "report.json").read_text())
    if trained["checkpoint_sha256"] != sha256(args.checkpoint):
        raise ValueError("Reference must be the evaluation of these exact weights")
    reference = trained["evaluations"]["final"]
    if reference["physical_contract"] != parent["physical_contract"]:
        raise ValueError("Reference and candidate physical contract differ")
    for case in reference["cases"]:
        if sha256(args.run / "final" / case["file"]) != case["sha256"]:
            raise ValueError("Reference capture checksum mismatch")
    # This path uses the sensor state seen by PPO collection after env.step.
    result = evaluate_hover(
        actor,
        parent,
        args.dataset,
        args.output / "step_return",
        device,
        refresh_observations=False,
    )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha256(args.checkpoint),
        "reference_training_report_sha256": sha256(args.run / "report.json"),
        "scope": "Frozen weights, same starts and physical contract; compare extra pre-actor forward refresh against the sensor timing used by PPO collection",
        "physics_documentation": "https://mujoco.readthedocs.io/en/stable/programming/simulation.html#simulation-loop",
        "refreshed_observations": reference,
        "ppo_collection_observations": result,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"cases": result["cases"], "wall_seconds": report["wall_seconds"]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    diagnose(parser.parse_args())
