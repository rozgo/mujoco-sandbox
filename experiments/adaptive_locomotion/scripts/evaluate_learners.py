"""Equal evaluation conditions for every final learner benchmark policy."""

import json

from evaluate_clearance import evaluate
from evaluate_visible_steps import compare

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.official import CHECKPOINT


def main():
    docs = ROOT / "docs/locomotion/gpu_comparison"
    reference = evaluate(
        CHECKPOINT, docs / "v1_reference.json", trials=8, seed=9217, timestep=0.0005
    )
    rows = []
    for label in ("mac_mps", "desktop_cpu", "desktop_cuda"):
        path = ROOT / f"assets/locomotion/checkpoints/learner_{label}_90s_seed2.pt"
        result = evaluate(
            path,
            docs / f"{label}_evaluation.json",
            trials=8,
            seed=9217,
            timestep=0.0005,
        )
        comparison = compare(reference, result)
        report = {
            "label": label,
            "completed_with_allowed_support": sum(
                case["completed_with_allowed_support"] for case in result["cases"]
            ),
            "trials": 72,
            "checks_relative_to_v1": comparison["checks"],
            "gait_comparison": comparison,
        }
        (docs / f"{label}_comparison.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        rows.append({k: v for k, v in report.items() if k != "gait_comparison"})
        print(json.dumps(rows[-1]), flush=True)
    (docs / "evaluation_summary.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
