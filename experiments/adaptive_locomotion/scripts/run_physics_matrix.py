"""Run the predeclared backend/family/batch matrix sequentially on one desktop."""

import subprocess
import sys
from pathlib import Path

from adaptive_locomotion.bodies import ROOT


def main():
    output = ROOT / "outputs/locomotion/warp_backend"
    output.mkdir(parents=True, exist_ok=True)
    for n in (512, 4096):
        for family in ("healthy", "nine"):
            for backend in ("mjbatch", "warp"):
                label = f"{backend}_{family}_{n}"
                if backend == "warp":
                    label = "bvh_" + label
                path = output / f"{label}.json"
                if path.exists():
                    print("PRESERVED", label, flush=True)
                    continue
                log = output / f"{label}.log"
                if log.exists():
                    raise ValueError(
                        f"Prior incomplete attempt {label}; preserve it before rerunning"
                    )
                print("START", label, flush=True)
                with log.open("w") as handle:
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("benchmark_physics.py")),
                            "--backend",
                            backend,
                            "--family",
                            family,
                            "--num-envs",
                            str(n),
                            "--label",
                            label,
                        ],
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        timeout=600,
                        check=False,
                    )
                if result.returncode:
                    raise RuntimeError(
                        f"{label} failed with status {result.returncode}; see retained log"
                    )
                print("DONE", label, flush=True)


if __name__ == "__main__":
    main()
