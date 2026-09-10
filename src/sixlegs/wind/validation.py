"""Held-out trajectories, independent fine grids, and measured forecast errors."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .flow import FORECAST_DT, SpectralFlow
from .operator import device, load, upsample, velocity


def relative(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-9))


def validate(model_dir, output, backend="auto", seeds=(200, 201, 202, 203)):
    dev = device(backend)
    torch.set_num_threads(4)
    models = {
        name: load(Path(model_dir) / f"{name}.pt", dev)[0] for name in ("fno", "pino")
    }
    rows = []
    examples = {}
    begin = time.monotonic()
    for n in (64, 128, 256):
        solver = SpectralFlow(n, seeds)
        # 256 is unseen by both the data loss and the physics loss.
        for start in (12.0,) if n == 256 else (2.0, 12.0, 28.0):
            solver.advance(start - solver.time)
            initial = solver.omega.copy()
            mean = solver.mean
            f = torch.tensor(solver.forcing, device=dev)
            mm = torch.tensor(mean, device=dev)
            predictions = {name: torch.tensor(initial, device=dev) for name in models}
            initial_v = solver.velocity(initial)
            with torch.no_grad():
                for step in range(8):
                    truth = solver.advance()
                    truth_v = solver.velocity()
                    for name, model in models.items():
                        predictions[name] = model(predictions[name], f, mm)
                    if step not in (0, 3, 7):
                        continue
                    lead = (step + 1) * FORECAST_DT
                    for name, pred in predictions.items():
                        pv = velocity(pred, mm).cpu().numpy()
                        pw = pred.cpu().numpy()
                        rows.append(
                            {
                                "model": name,
                                "resolution": n,
                                "initial_time_s": start,
                                "lead_s": lead,
                                "vorticity_relative_l2": relative(pw, truth),
                                "velocity_fluctuation_relative_l2": relative(
                                    pv - mean[:, :, None, None],
                                    truth_v - mean[:, :, None, None],
                                ),
                                "velocity_rmse_m_s": float(
                                    np.sqrt(np.mean((pv - truth_v) ** 2))
                                ),
                                "velocity_error_vs_persistence": float(
                                    np.linalg.norm(pv - truth_v)
                                    / max(np.linalg.norm(initial_v - truth_v), 1e-9)
                                ),
                            }
                        )
                        if n == 128 and start == 12 and step == 7:
                            examples[name] = pw[0]
                    if n == 128 and start == 12 and step == 7:
                        examples.update(initial=initial[0], truth=truth[0])
            print(f"Validated grid {n} at initial time {start:g}s", flush=True)
    # Explicit interpolation baseline on the SAME fine-grid input and future.
    solver = SpectralFlow(128, seeds)
    solver.advance(12.0)
    w = solver.omega.copy()
    f = solver.forcing.copy()
    mean = solver.mean.copy()
    predictions = {}
    fine = torch.tensor(w, device=dev)
    coarse = fine[:, ::2, ::2]
    with torch.no_grad():
        for _ in range(8):
            fine = models["pino"](
                fine, torch.tensor(f, device=dev), torch.tensor(mean, device=dev)
            )
            coarse = models["pino"](
                coarse,
                torch.tensor(f[:, ::2, ::2], device=dev),
                torch.tensor(mean, device=dev),
            )
        truth = solver.advance(2.0)
        interpolation = {
            "direct_128_relative_l2": relative(fine.cpu().numpy(), truth),
            "fourier_upsampled_64_relative_l2": relative(
                upsample(coarse, 128).cpu().numpy(), truth
            ),
        }
    # Batch-one wall timing includes model transfers to/from its device.
    timing = {}
    s = SpectralFlow(128, [200])
    model = models["pino"]
    for _ in range(3):
        with torch.no_grad():
            model(
                torch.tensor(s.omega, device=dev),
                torch.tensor(s.forcing, device=dev),
                torch.tensor(s.mean, device=dev),
            ).cpu().numpy()
    t = time.perf_counter()
    for _ in range(12):
        with torch.no_grad():
            model(
                torch.tensor(s.omega, device=dev),
                torch.tensor(s.forcing, device=dev),
                torch.tensor(s.mean, device=dev),
            ).cpu().numpy()
    timing["operator_seconds_per_quarter_second"] = (time.perf_counter() - t) / 12
    t = time.perf_counter()
    for _ in range(12):
        s.advance()
    timing["reference_seconds_per_quarter_second"] = (time.perf_counter() - t) / 12
    timing["ratio_reference_to_operator"] = (
        timing["reference_seconds_per_quarter_second"]
        / timing["operator_seconds_per_quarter_second"]
    )
    timing["operator_backend"] = str(dev)
    timing["reference_backend"] = "NumPy CPU"
    timing["batch_size"] = 1
    timing["resolution"] = 128
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "held_out_seeds": list(seeds),
        "rows": rows,
        "resolution_transfer": interpolation,
        "timing": timing,
        "wall_seconds": time.monotonic() - begin,
    }
    (output / "operator_validation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    np.savez_compressed(output / "field_example.npz", **examples)
    return result
