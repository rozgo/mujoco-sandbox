"""Predetermined moving-support holdout and explicit damaged-body transfer probes."""

import argparse
import json
import time
from pathlib import Path

from adaptive_locomotion.moving_cli import STANDING
from adaptive_locomotion.moving_evaluate import evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--physics-backend", choices=("mjbatch", "warp"), default="mjbatch")
    p.add_argument("--seed", type=int, default=9407)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    rows = []
    for name, path, relative in (
        ("frozen_world", STANDING, False),
        ("frozen_relative", STANDING, True),
        ("trained", args.checkpoint, True),
    ):
        r = evaluate(
            path,
            args.output / f"{name}.json",
            seed=args.seed,
            physics_backend=args.physics_backend,
            relative=relative,
        )
        rows.append(
            {
                "variant": name,
                "passed": r["passed"],
                "survived": r["survived"],
                "trials": r["trials"],
            }
        )
    for body in ("whole_fr", "whole_rr"):
        r = evaluate(
            args.checkpoint,
            args.output / f"transfer_{body}.json",
            motions="combined",
            seed=args.seed,
            physics_backend=args.physics_backend,
            body=body,
        )
        rows.append(
            {
                "transfer_body": body,
                "passed": r["passed"],
                "survived": r["survived"],
                "trials": r["trials"],
            }
        )
    report = {
        "seed": args.seed,
        "physics_backend": args.physics_backend,
        "rows": rows,
        "wall_seconds": time.perf_counter() - start,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
