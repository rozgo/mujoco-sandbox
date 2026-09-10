"""Dealiased 2-D periodic Navier–Stokes reference, in vorticity form.

The robot is a one-way probe of an externally generated horizontal flow.
No rotor wash, gate wake, or body boundary condition is implied.
"""

import json
import time
from pathlib import Path

import numpy as np

DOMAIN = 12.0
VISCOSITY = 0.04
FORECAST_DT = 0.25


def initial_fields(seeds, n=64):
    """Same continuous initial functions on every grid; independent seed splits."""
    coord = np.arange(n) * DOMAIN / n
    x, y = np.meshgrid(coord, coord)
    modes = [
        (a, b)
        for a in range(5)
        for b in range(-4, 5)
        if (a > 0 or b > 0) and 1 <= a * a + b * b <= 16
    ]
    fields = []
    forcings = []
    means = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        chosen = rng.choice(len(modes), 12, replace=False)
        amplitudes = rng.normal(size=12) / np.array(
            [sum(v * v for v in modes[i]) ** 0.65 for i in chosen]
        )
        amplitudes *= 2.0 / np.sqrt(np.sum(amplitudes**2) / 2)
        phases = rng.uniform(0, 2 * np.pi, 12)
        omega = sum(
            a * np.sin(2 * np.pi * (modes[i][0] * x + modes[i][1] * y) / DOMAIN + p)
            for i, a, p in zip(chosen, amplitudes, phases)
        )
        force = 0.3 * np.sin(
            2 * np.pi * 3 * y / DOMAIN + rng.uniform(0, 2 * np.pi)
        ) + 0.15 * np.cos(2 * np.pi * 2 * x / DOMAIN + rng.uniform(0, 2 * np.pi))
        fields.append(omega)
        forcings.append(force)
        means.append([rng.uniform(-0.4, 0.4), rng.uniform(1.4, 2.2)])
    return (
        np.array(fields, dtype=np.float32),
        np.array(forcings, dtype=np.float32),
        np.array(means, dtype=np.float32),
    )


class SpectralFlow:
    def __init__(self, n=64, seeds=(0,), viscosity=VISCOSITY):
        self.n = n
        self.viscosity = viscosity
        self.time = 0.0
        self.omega, self.forcing, self.mean = initial_fields(seeds, n)
        self.kx = (2 * np.pi * np.fft.rfftfreq(n, d=DOMAIN / n))[None, :]
        self.ky = (2 * np.pi * np.fft.fftfreq(n, d=DOMAIN / n))[:, None]
        self.k2 = self.kx**2 + self.ky**2
        self.inverse = np.divide(
            1.0, self.k2, out=np.zeros_like(self.k2), where=self.k2 > 0
        )
        self.mask = (abs(self.kx) < 2 * np.pi / DOMAIN * (n / 3)) & (
            abs(self.ky) < 2 * np.pi / DOMAIN * (n / 3)
        )
        self.w_hat = np.fft.rfft2(self.omega) * self.mask
        self.f_hat = np.fft.rfft2(self.forcing) * self.mask

    def velocity(self, omega=None):
        wh = self.w_hat if omega is None else np.fft.rfft2(omega)
        u = (
            np.fft.irfft2(1j * self.ky * self.inverse * wh, s=(self.n, self.n))
            + self.mean[..., 0, None, None]
        )
        v = (
            np.fft.irfft2(-1j * self.kx * self.inverse * wh, s=(self.n, self.n))
            + self.mean[..., 1, None, None]
        )
        return np.stack((u, v), axis=-3).astype(np.float32)

    def rhs(self, wh):
        u = (
            np.fft.irfft2(1j * self.ky * self.inverse * wh, s=(self.n, self.n))
            + self.mean[..., 0, None, None]
        )
        v = (
            np.fft.irfft2(-1j * self.kx * self.inverse * wh, s=(self.n, self.n))
            + self.mean[..., 1, None, None]
        )
        dx = np.fft.irfft2(1j * self.kx * wh, s=(self.n, self.n))
        dy = np.fft.irfft2(1j * self.ky * wh, s=(self.n, self.n))
        return (
            -np.fft.rfft2(u * dx + v * dy) - self.viscosity * self.k2 * wh + self.f_hat
        ) * self.mask

    def advance(self, seconds=FORECAST_DT):
        steps = int(np.ceil(seconds / (0.01 * 64 / self.n)))
        dt = seconds / steps
        for _ in range(steps):
            w = self.w_hat
            a = w + dt * self.rhs(w)
            b = 0.75 * w + 0.25 * (a + dt * self.rhs(a))
            self.w_hat = (w / 3 + 2 / 3 * (b + dt * self.rhs(b))) * self.mask
        self.time += seconds
        self.omega = np.fft.irfft2(self.w_hat, s=(self.n, self.n)).astype(np.float32)
        if not np.isfinite(self.omega).all():
            raise RuntimeError("Reference flow became unstable")
        return self.omega.copy()


def sample(field, points):
    """Bilinear periodic sample: field [...,channels,y,x], points [p,2]."""
    n = field.shape[-1]
    q = (np.asarray(points) + DOMAIN / 2) / DOMAIN * n
    i = np.floor(q).astype(int)
    f = q - i
    x, y = i[:, 0] % n, i[:, 1] % n
    dx, dy = f[:, 0], f[:, 1]
    return (
        (1 - dx) * (1 - dy) * field[..., y, x]
        + dx * (1 - dy) * field[..., y, (x + 1) % n]
        + (1 - dx) * dy * field[..., (y + 1) % n, x]
        + dx * dy * field[..., (y + 1) % n, (x + 1) % n]
    ).swapaxes(-1, -2)


def dataset(output, train_seeds=48, validation_seeds=8, snapshots=64, n=64):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    seeds = list(range(train_seeds)) + list(range(100, 100 + validation_seeds))
    chunks = []
    forcing = []
    means = []
    for start in range(0, len(seeds), 8):
        batch = seeds[start : start + 8]
        solver = SpectralFlow(n, batch)
        states = [solver.omega.copy()]
        for _ in range(snapshots):
            states.append(solver.advance())
        chunks.append(np.stack(states, axis=1))
        forcing.append(solver.forcing)
        means.append(solver.mean)
        print(
            f"Generated {min(start + 8, len(seeds))}/{len(seeds)} independent flow trajectories",
            flush=True,
        )
    np.savez_compressed(
        output,
        omega=np.concatenate(chunks),
        forcing=np.concatenate(forcing),
        mean=np.concatenate(means),
        seeds=seeds,
        dt=FORECAST_DT,
    )
    report = {
        "train_seeds": list(range(train_seeds)),
        "validation_seeds": list(range(100, 100 + validation_seeds)),
        "resolution": n,
        "snapshots_per_seed": snapshots + 1,
        "forecast_dt": FORECAST_DT,
        "domain_m": DOMAIN,
        "viscosity_m2_s": VISCOSITY,
        "generation_seconds": time.monotonic() - begin,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
