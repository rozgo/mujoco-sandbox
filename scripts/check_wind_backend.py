"""Compare CPU and GPU autoregressive forecasts after the upstream FFT fix."""

import argparse
import json
from pathlib import Path

import torch

from sixlegs.wind.flow import initial_fields
from sixlegs.wind.operator import device, load


def check(models, backend):
    torch.set_num_threads(4)
    dev = device(backend)
    rows = []
    for kind in ("fno", "pino"):
        cpu, _ = load(models / f"{kind}.pt")
        gpu, _ = load(models / f"{kind}.pt", dev)
        for n in (64, 128, 256):
            w, f, mean = [torch.tensor(a) for a in initial_fields([200, 201], n)]
            a, b = w, w.to(dev)
            with torch.no_grad():
                for _ in range(8):
                    a = cpu(a, f, mean)
                    b = gpu(b, f.to(dev), mean.to(dev))
            error = (a - b.cpu()).abs().max().item()
            row = {"model": kind, "grid": n, "max_abs_omega_error": error}
            print(row, flush=True)
            assert error < 2e-4, row
            rows.append(row)
    return {"backend": str(dev), "torch": torch.__version__, "lead_s": 2, "rows": rows}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=Path("assets/wind"))
    p.add_argument("--device", default="auto")
    p.add_argument("--output", type=Path, default=Path("outputs/wind/backend.json"))
    args = p.parse_args()
    result = check(args.models, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
