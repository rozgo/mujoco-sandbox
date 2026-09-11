"""Audit the user-selected frozen v1; report original gates without reselecting."""

import hashlib
import json

from evaluate_clearance import evaluate
from evaluate_rear_overlap import PARENT, compare

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.evaluate import rollout
from adaptive_locomotion.limb_loss import LOSS_CASES
from adaptive_locomotion.official import CHECKPOINT, CHECKPOINT_SHA256, TIMESTEP
from adaptive_locomotion.train import load_checkpoint


def main():
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == CHECKPOINT_SHA256
    docs = ROOT / "docs/locomotion"
    seed = 20260924  # Reserved before selection; first used after user acceptance.
    parent = evaluate(
        PARENT, docs / "V1_REFERENCE.json", trials=32, seed=seed, timestep=TIMESTEP
    )
    candidate = evaluate(
        CHECKPOINT, docs / "V1_VALIDATION.json", trials=32, seed=seed, timestep=TIMESTEP
    )
    comparison = compare(parent, candidate)
    (docs / "V1_COMPARISON.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print("ORIGINAL_GATES", json.dumps(comparison["checks"]), flush=True)
    net, _ = load_checkpoint(CHECKPOINT)
    physics = []
    for case in LOSS_CASES:
        row, _, _ = rollout(net, case, trials=8, seed=seed, timestep=TIMESTEP / 2)
        physics.append(row)
        print("HALF", case, row["completed_with_allowed_support"], flush=True)
    (docs / "V1_PHYSICS.json").write_text(json.dumps(physics, indent=2) + "\n")
    summary = {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "seed": seed,
        "selection": "User accepted the existing video and checkpoint before this audit",
        "training_seconds": 0,
        "completed_with_allowed_support": sum(
            c["completed_with_allowed_support"] for c in candidate["cases"]
        ),
        "task_trials": 32 * len(LOSS_CASES),
        "half_timestep_completed": sum(
            c["completed_with_allowed_support"] for c in physics
        ),
        "half_timestep_trials": 8 * len(LOSS_CASES),
        "original_automatic_gates": comparison["checks"],
        "original_automatic_selection_passed": comparison["eligible"],
        "known_recorded_penetration_m": 0.008889369128127676,
        "original_penetration_limit_m": 0.008,
    }
    (docs / "V1_AUDIT_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    # Preserve the failed original gates; user acceptance is a separate decision.
    return 0 if comparison["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
