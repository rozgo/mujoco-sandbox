"""Causal forecast cache and identical numerical weather for paired flights."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .flow import DOMAIN, FORECAST_DT, SpectralFlow, sample
from .operator import device, load, velocity

REFERENCE_DT = 0.05


def prepare(seed, model_dir, output, duration=46.0, n=128, backend="auto", horizon=2.5):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    solver = SpectralFlow(n, [seed])
    velocities = [solver.velocity()[0]]
    observations = [solver.omega[0].copy()]
    begin = time.monotonic()
    for i in range(round(duration / REFERENCE_DT)):
        solver.advance(REFERENCE_DT)
        velocities.append(solver.velocity()[0])
        if (i + 1) % 5 == 0:
            observations.append(solver.omega[0].copy())
    observations = np.array(observations)
    np.savez_compressed(
        output / "reference.npz",
        velocity=velocities,
        omega=observations,
        forcing=solver.forcing[0],
        mean=solver.mean[0],
        seed=seed,
        dt=REFERENCE_DT,
        observation_dt=FORECAST_DT,
    )
    print(f"Prepared independent reference wind for seed {seed}", flush=True)
    dev = device(backend)
    forecast_steps = int(np.ceil(horizon / FORECAST_DT)) + 1
    torch.set_num_threads(4)
    meta = {
        "seed": seed,
        "resolution": n,
        "duration": duration,
        "forecast_observation_dt": FORECAST_DT,
        "reference_sample_dt": REFERENCE_DT,
        "backend": str(dev),
        "forecast_horizon_s": (forecast_steps - 1) * FORECAST_DT,
        "models": {},
    }
    for kind in ("fno", "pino"):
        model, checkpoint = load(Path(model_dir) / f"{kind}.pt", dev)
        result = np.lib.format.open_memmap(
            output / f"{kind}_forecast.npy",
            mode="w+",
            dtype=np.float32,
            shape=(len(observations), forecast_steps, 2, n, n),
        )
        inference_begin = time.monotonic()
        with torch.no_grad():
            for start in range(0, len(observations), 16):
                w = torch.tensor(observations[start : start + 16], device=dev)
                f = torch.tensor(solver.forcing, device=dev).expand(len(w), -1, -1)
                mean = torch.tensor(solver.mean, device=dev).expand(len(w), -1)
                for j in range(forecast_steps):
                    result[start : start + len(w), j] = velocity(w, mean).cpu().numpy()
                    if j < forecast_steps - 1:
                        w = model(w, f, mean)
        result.flush()
        del result
        meta["models"][kind] = {
            "epoch": checkpoint["epoch"],
            "training_resolution": checkpoint["training_resolution"],
            "inference_seconds": time.monotonic() - inference_begin,
            "sha256": hashlib.sha256(
                (Path(model_dir) / f"{kind}.pt").read_bytes()
            ).hexdigest(),
        }
        print(f"Prepared {kind.upper()} causal forecasts for seed {seed}", flush=True)
    meta["preparation_seconds"] = time.monotonic() - begin
    (output / "weather.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


class Weather:
    def __init__(self, folder, kind="pino", scale=1.0):
        folder = Path(folder)
        r = np.load(folder / "reference.npz")
        self.velocity = r["velocity"]
        self.omega = r["omega"]
        self.mean = r["mean"]
        self.seed = int(r["seed"])
        self.kind = kind
        if scale <= 0:
            raise ValueError("Wind similarity scale must be positive")
        self.scale = scale
        meta_path = folder / "weather.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.provenance = {
            "seed": self.seed,
            "resolution": self.velocity.shape[-1],
            "model_sha256": {k: v["sha256"] for k, v in meta.get("models", {}).items()},
        }
        if scale != 1:
            self.provenance["navier_stokes_similarity_scale"] = scale
        self.predictions = (
            np.load(folder / f"{kind}_forecast.npy", mmap_mode="r")
            if kind in ("fno", "pino")
            else None
        )

    def at(self, t, points):
        q = np.clip(t * self.scale / REFERENCE_DT, 0, len(self.velocity) - 1)
        i = min(int(q), len(self.velocity) - 2)
        f = q - i
        return self.scale * (
            (1 - f) * sample(self.velocity[i], points)
            + f * sample(self.velocity[i + 1], points)
        )

    def field(self, t):
        q = np.clip(t * self.scale / REFERENCE_DT, 0, len(self.velocity) - 1)
        i = min(int(q), len(self.velocity) - 2)
        f = q - i
        return self.scale * ((1 - f) * self.velocity[i] + f * self.velocity[i + 1])

    def forecast_field(self, t, lead):
        """Full learned field for visualization, with the same causal clock as control."""
        native = t * self.scale
        obs = min(int((native + 1e-8) / FORECAST_DT), len(self.predictions) - 1)
        q = (native - obs * FORECAST_DT + lead * self.scale) / FORECAST_DT
        array = self.predictions[obs]
        if q > len(array) - 1 + 1e-6:
            raise ValueError("Forecast cache is too short for this wind scale")
        i = min(int(q), len(array) - 2)
        fraction = q - i
        return self.scale * ((1 - fraction) * array[i] + fraction * array[i + 1])

    def forecast(self, t, leads, points):
        if self.kind == "oracle":
            return np.array(
                [self.at(t + lead, [p])[0] for lead, p in zip(leads, points)]
            )
        t *= self.scale
        leads = np.asarray(leads) * self.scale
        obs = min(int((t + 1e-8) / FORECAST_DT), len(self.omega) - 1)
        if self.kind == "persistence":
            return self.scale * sample(
                self.velocity[min(obs * 5, len(self.velocity) - 1)], points
            )
        array = self.predictions[obs]
        if (
            np.max(t - obs * FORECAST_DT + leads)
            > (len(array) - 1) * FORECAST_DT + 1e-6
        ):
            raise ValueError("Forecast cache is too short for this wind scale")
        q = np.clip((t - obs * FORECAST_DT + leads) / FORECAST_DT, 0, len(array) - 1)
        i = np.minimum(q.astype(int), len(array) - 2)
        a = (q - i)[:, None]

        def spatial(index):
            n = array.shape[-1]
            grid = (points + DOMAIN / 2) / DOMAIN * n
            j = np.floor(grid).astype(int)
            f = grid - j
            x, y = j[:, 0] % n, j[:, 1] % n
            dx, dy = f[:, 0, None], f[:, 1, None]
            return (
                (1 - dx) * (1 - dy) * array[index, :, y, x]
                + dx * (1 - dy) * array[index, :, y, (x + 1) % n]
                + (1 - dx) * dy * array[index, :, (y + 1) % n, x]
                + dx * dy * array[index, :, (y + 1) % n, (x + 1) % n]
            )

        return self.scale * ((1 - a) * spatial(i) + a * spatial(i + 1))
