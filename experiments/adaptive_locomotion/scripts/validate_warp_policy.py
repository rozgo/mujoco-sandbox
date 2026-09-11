"""Complete GPU-physics trials of frozen v1; no learning or trajectory resets."""

import hashlib
import json
import time

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.evaluate import rollout
from adaptive_locomotion.official import CHECKPOINT
from adaptive_locomotion.train import load_checkpoint


def main():
    output = ROOT / "outputs/locomotion/warp_backend/frozen_policy_trials.json"
    if output.exists():
        raise ValueError("Preserve the existing policy trials")
    net, _ = load_checkpoint(CHECKPOINT)
    cases = []
    start = time.perf_counter()
    for case in ("healthy", "whole_fr"):
        result, _, _ = rollout(
            net,
            case,
            trials=4,
            seed=9225,
            seconds=12,
            timestep=0.0005,
            support_substeps=True,
            physics_backend="warp",
        )
        cases.append(result)
        print(case, result["completed_with_allowed_support"], flush=True)
    report = {
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "seed": 9225,
        "cases": cases,
        "new_training_seconds": 0,
        "wall_seconds_including_setup": time.perf_counter() - start,
        "scope": "Frozen official v1 in GPU physics at 0.5 ms; four complete trials each in two predetermined bodies. CPU actor inference; all support substeps checked. No robustness or training-speed claim.",
    }
    output.write_text(json.dumps(report, indent=2) + "\n")
    if any(c["completed_with_allowed_support"] != 4 for c in cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
