"""Reuse the unchanged nine-body gait gates for a new shared policy."""

import argparse
import json
from pathlib import Path

from evaluate_clearance import evaluate
from evaluate_visible_steps import compare

from adaptive_locomotion.bodies import ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parent = json.loads(
        (
            ROOT / "docs/locomotion/standing/consolidate_120s_seed12/walking.json"
        ).read_text()
    )
    candidate = evaluate(
        args.checkpoint,
        args.output / "walking.json",
        trials=4,
        seed=9301,
        timestep=0.0005,
    )
    result = compare(parent, candidate)
    (args.output / "walking_comparison.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "feet"}), flush=True)
    return 0 if result["eligible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
