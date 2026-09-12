"""Evaluate only the new final Warp-trained policy against archived matched v1 trials."""

import json

from evaluate_clearance import evaluate
from evaluate_visible_steps import compare

from adaptive_locomotion.bodies import ROOT


def main():
    docs = ROOT / "docs/locomotion/gpu_comparison"
    output = docs / "warp_bvh_4096_evaluation.json"
    if output.exists():
        raise ValueError("Preserve the existing evaluation")
    reference = json.loads((docs / "v1_reference.json").read_text())
    path = ROOT / "assets/locomotion/checkpoints/learner_warp_bvh_4096_90s_seed2.pt"
    result = evaluate(path, output, trials=8, seed=9217, timestep=0.0005)
    comparison = compare(reference, result)
    report = {
        "label": "warp_bvh_4096",
        "completed_with_allowed_support": sum(
            c["completed_with_allowed_support"] for c in result["cases"]
        ),
        "trials": 72,
        "checks_relative_to_v1": comparison["checks"],
        "gait_comparison": comparison,
        "scope": "One 90-second 4096-world GPU-physics continuation; CPU task evaluation, not candidate promotion.",
    }
    (docs / "warp_bvh_4096_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "gait_comparison"}))
    if not all(comparison["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
