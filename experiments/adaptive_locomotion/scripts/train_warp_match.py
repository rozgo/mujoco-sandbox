"""Predeclared same-network/reward/update comparison; fresh process per run."""

import json
import subprocess
import sys

from adaptive_locomotion.bodies import ROOT


def main():
    out = ROOT / "outputs/locomotion/warp_training"
    out.mkdir(parents=True, exist_ok=True)
    for seed in (2, 3, 4):
        order = ("mjbatch", "warp") if seed % 2 == 0 else ("warp", "mjbatch")
        for backend in order:
            label = f"matched_{backend}_seed{seed}"
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
                        "512",
                        "--seed",
                        str(seed),
                        "--seconds",
                        "180",
                        "--allowance",
                        "2400",
                        "--minibatch-size",
                        "3072",
                        "--max-iterations",
                        "48",
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
            if report["iterations"] != 48 or report["optimizer_steps"] != 768:
                raise RuntimeError(
                    "Matched update count did not finish; retain and diagnose"
                )


if __name__ == "__main__":
    main()
