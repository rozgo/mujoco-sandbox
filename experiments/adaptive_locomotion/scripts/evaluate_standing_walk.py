"""Use the existing nine-body walking gates, unchanged, for standing candidates."""

import argparse
import hashlib
import json
from pathlib import Path

from adaptive_locomotion.bodies import ROOT
from evaluate_clearance import evaluate
from evaluate_visible_steps import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=9301)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--timestep", type=float, default=0.0005)
    args = parser.parse_args()
    baseline = ROOT / "assets/locomotion/checkpoints/adaptive_dog_v1.pt"
    args.output.mkdir(parents=True, exist_ok=True)
    cache = (
        ROOT
        / f"outputs/locomotion/standing/v1_walk_{args.seed}_{args.trials}_{args.timestep}.json"
    )
    if cache.exists():
        reference = json.loads(cache.read_text())
        assert reference["sha256"] == hashlib.sha256(baseline.read_bytes()).hexdigest()
        assert reference["seed"] == args.seed
        assert reference["trials_per_case"] == args.trials
        assert reference["physics_timestep_s"] == args.timestep
    else:
        reference = evaluate(
            baseline, cache, trials=args.trials, seed=args.seed, timestep=args.timestep
        )
    candidate = evaluate(
        args.checkpoint,
        args.output / "walking.json",
        trials=args.trials,
        seed=args.seed,
        timestep=args.timestep,
    )
    result = compare(reference, candidate)
    result.update(
        reference_sha256=reference["sha256"],
        seed=args.seed,
        trials_per_case=args.trials,
        physics_timestep_s=args.timestep,
    )
    (args.output / "walking_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result["checks"]))
    return 0 if result["eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
