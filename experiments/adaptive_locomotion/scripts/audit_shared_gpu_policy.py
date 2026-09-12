"""Audit walking, static holds and moving holds using exactly one checkpoint."""

import argparse
import hashlib
import json
import time
from pathlib import Path

from adaptive_locomotion.limb_eval import assess
from adaptive_locomotion.limb_loss import LOSS_CASES
from adaptive_locomotion.moving_evaluate import evaluate as moving
from adaptive_locomotion.standing_evaluate import evaluate as standing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9507)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument(
        "--physics-backend", choices=("mjbatch", "warp"), default="mjbatch"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a new directory; preserve all prior evaluations")
    if args.trials < 1:
        raise ValueError("Trial count must be positive")
    args.output.mkdir(parents=True)
    sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    started = time.perf_counter()
    walk = assess(
        args.checkpoint,
        args.output / "walking.json",
        trials=args.trials,
        seed=args.seed,
        timestep=0.0005,
    )
    shared = {
        "checkpoint": args.checkpoint,
        "trials": args.trials,
        "seed": args.seed,
        "physics_backend": args.physics_backend,
    }
    healthy = standing(
        output=args.output / "standing_healthy.json", surfaces="all", **shared
    )
    damaged = standing(
        output=args.output / "standing_damaged.json",
        surfaces="flat",
        bodies=",".join(LOSS_CASES[1:]),
        **shared,
    )
    deck = moving(output=args.output / "moving.json", motions="all", **shared)
    assert sha == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    assert all(r["checkpoint_sha256"] == sha for r in (healthy, damaged, deck))
    assert walk["sha256"] == sha
    summary = {
        "checkpoint_sha256": sha,
        "same_checkpoint_for_all_commands": True,
        "seed": args.seed,
        "walking_physics": "CPU MuJoCo/mjbatch, 0.5 ms",
        "standing_moving_physics": args.physics_backend,
        "walking_completed": sum(
            c["completed_with_allowed_support"] for c in walk["cases"]
        ),
        "walking_trials": len(walk["cases"]) * args.trials,
        "healthy_gait_checks": walk["healthy_gates"],
        "standing_healthy": {k: healthy[k] for k in ("passed", "trials")},
        "standing_damaged": {k: damaged[k] for k in ("passed", "trials")},
        "moving": {k: deck[k] for k in ("passed", "survived", "trials")},
        "evaluation_wall_seconds": time.perf_counter() - started,
        "new_training_seconds": 0,
        "scope": "All raw failures retained. Every command family uses the same frozen actor; no policy switching, refinement or checkpoint selection during this audit.",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
