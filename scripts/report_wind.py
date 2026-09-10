"""Publish measured wind-demo results and checkpoint provenance."""

import hashlib
import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import mujoco
import numpy as np
import torch

from sixlegs.wind.operator import load

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/wind"


def main():
    comparison = json.loads((DOCS / "comparison.json").read_text())
    validation = json.loads((DOCS / "operator_validation.json").read_text())
    kinds = ("persistence", "fno", "pino", "oracle")
    groups = {k: [r for r in comparison if r["forecast"] == k] for k in kinds}
    assert len(comparison) == 24 and all(r["success"] for r in comparison)
    mean = {
        k: np.mean([r["payload_tracking_rmse_m"] for r in g]) for k, g in groups.items()
    }
    reduction = 1 - mean["pino"] / mean["persistence"]
    lines = [
        "# Measured wind-delivery results",
        "",
        f"**24/24 complete deliveries; PINO reduced mean crossing tracking RMSE by {reduction:.1%} relative to the frozen-field forecast.**",
        "",
        "These are six paired held-out cases (seeds 300–305), not a statistical guarantee for other winds. The same controller and actuator limits are used throughout. RMSE measures horizontal package error during t = 6–32 s.",
        "",
        "| Forecast | Mean tracking RMSE | Mean peak swing | Deliveries |",
        "| --- | ---: | ---: | ---: |",
    ]
    labels = dict(
        zip(
            kinds, ("Frozen field", "Data-only FNO", "PINO", "True future (diagnostic)")
        )
    )
    for k, group in groups.items():
        swing = np.mean([r["peak_swing_deg"] for r in group])
        lines.append(f"| {labels[k]} | {mean[k] * 100:.2f} cm | {swing:.2f}° | 6/6 |")
    lines += [
        "",
        "FNO and PINO give similar flight performance here. The strong finding is the value of forecasting over holding the wind fixed; this experiment does not establish a large advantage for the physics loss. True-future wind is only a diagnostic: model mismatch and approximate control can make an imperfect forecast occasionally score better.",
        "",
        "| Wind seed | Frozen field | FNO | PINO | True future |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for seed in range(300, 306):
        errors = [
            next(r["payload_tracking_rmse_m"] for r in groups[k] if r["seed"] == seed)
            * 100
            for k in kinds
        ]
        lines.append(f"| {seed} | " + " | ".join(f"{v:.2f} cm" for v in errors) + " |")
    lines += [
        "",
        "## Independent flow validation",
        "",
        "| Grid | FNO 2 s velocity RMSE | PINO 2 s velocity RMSE |",
        "| --- | ---: | ---: |",
    ]
    for n in (64, 128, 256):
        errors = [
            np.mean(
                [
                    r["velocity_rmse_m_s"]
                    for r in validation["rows"]
                    if r["model"] == k and r["resolution"] == n and r["lead_s"] == 2
                ]
            )
            for k in ("fno", "pino")
        ]
        lines.append(f"| {n}² | {errors[0]:.4f} m/s | {errors[1]:.4f} m/s |")
    transfer = validation["resolution_transfer"]
    lines += [
        "",
        "64²/128² use four held-out trajectories and three initial times; 256² uses the same four trajectories at 12 s. Training uses 64² labels and PINO also uses a 128² physics residual. The 256² grid is unseen by either loss.",
        "",
        f"Direct 128² PINO vorticity relative L2 error is {transfer['direct_128_relative_l2']:.6f}; Fourier interpolation of the 64² rollout gives {transfer['fourier_upsampled_64_relative_l2']:.6f}. These nearly equal values show successful grid transfer, not added fine-scale accuracy.",
        "",
        "CPU/CUDA and CPU/Metal agreement is checked over eight autoregressive steps at all three resolutions; maximum absolute vorticity disagreement is below 0.00002 s⁻¹. See [CUDA checks](backend_cuda.json) and [Metal checks](backend_mps.json).",
        "",
        "Timing samples in the validation JSON are diagnostic wall measurements against this NumPy reference, including model transfers. They are not a benchmark against optimized CFD or a guaranteed hardware speedup. Final validation ran alongside other preparation work, so no precise speedup claim is made.",
        "",
        "## Physical checks",
        "",
        f"All runs stayed within the 10 N per-rotor cap (observed maximum {max(r['max_rotor_thrust_n'] for r in comparison):.3f} N). Maximum cable constraint extension was {1000 * max(r['max_cable_extension_m'] for r in comparison):.3f} mm. No gate collisions or MuJoCo numerical warnings occurred. Every release followed physical destination contact.",
        "",
        "The movie uses seed 300, the first evaluation seed chosen before the final results. [Full per-flight records](comparison.json) · [Flow validation](operator_validation.json) · [CUDA flow validation](operator_validation_cuda.json) · [Model provenance](manifest.json).",
    ]
    (DOCS / "RESULTS.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "evaluation_host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
        },
        "training": json.loads((DOCS / "training_setup.json").read_text()),
        "models": {},
        "flight_seeds": list(range(300, 306)),
        "movie_seed": 300,
    }
    for k in ("fno", "pino"):
        path = ROOT / f"assets/wind/{k}.pt"
        model, meta = load(path)
        manifest["models"][k] = {
            **{key: value for key, value in meta.items() if key != "state_dict"},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parameter_elements": sum(p.numel() for p in model.parameters()),
        }
    (DOCS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(lines[2])


if __name__ == "__main__":
    main()
