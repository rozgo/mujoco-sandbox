"""Predeclared same-network/reward/update comparison; fresh process per run."""

import argparse
import json
import subprocess
import sys

from adaptive_locomotion.bodies import ROOT


def main(prefix="matched", num_envs=512, iterations=48):
    out = ROOT / "outputs/locomotion/warp_training"
    out.mkdir(parents=True, exist_ok=True)
    for seed in (2, 3, 4):
        order = ("mjbatch", "warp") if seed % 2 == 0 else ("warp", "mjbatch")
        for backend in order:
            label = f"{prefix}_{backend}_seed{seed}"
            logfile = out / f"{label}.log"
            if logfile.exists():
                raise ValueError(f"Preserve the existing run: {label}")
            print(json.dumps({"starting": label}), flush=True)
            with logfile.open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "scripts/benchmark_learner.py",
                        "--label",
                        label,
                        "--device",
                        "cuda",
                        "--physics-backend",
                        backend,
                        "--num-envs",
                        str(num_envs),
                        "--seed",
                        str(seed),
                        "--seconds",
                        "180",
                        "--allowance",
                        "2400",
                        "--minibatch-size",
                        "3072",
                        "--max-iterations",
                        str(iterations),
                        "--output-family",
                        "warp_training",
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            report = json.loads((out / label / "training.json").read_text())
            print(
                json.dumps(
                    {
                        "finished": label,
                        "seconds": report["training_seconds"],
                        "iterations": report["iterations"],
                        "optimizer_steps": report["optimizer_steps"],
                    }
                ),
                flush=True,
            )
            expected_steps = iterations * 4 * ((num_envs * 24 + 3071) // 3072)
            if (
                report["iterations"] != iterations
                or report["optimizer_steps"] != expected_steps
            ):
                raise RuntimeError(
                    "Matched update count did not finish; retain and diagnose"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", choices=("matched", "scaled"), default="matched")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=48)
    main(**vars(parser.parse_args()))
