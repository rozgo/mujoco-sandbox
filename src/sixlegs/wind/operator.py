"""A genuinely trained FNO time-step map, with optional fine-grid PDE loss."""

import json
import time
from pathlib import Path

import numpy as np
import torch
from neuralop.models import FNO
from torch import nn

from .flow import DOMAIN, FORECAST_DT, VISCOSITY

# Keep CUDA validation comparable to CPU/Metal instead of using TF32 convolutions.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def device(name="auto"):
    if name != "auto":
        return torch.device(name)
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class FlowOperator(nn.Module):
    def __init__(self, width=16, modes=12, layers=4):
        super().__init__()
        self.config = {"width": width, "modes": modes, "layers": layers}
        self.fno = FNO(
            n_modes=(modes, modes),
            hidden_channels=width,
            in_channels=4,
            out_channels=1,
            n_layers=layers,
            positional_embedding=None,
            enforce_hermitian_symmetry=True,
        )

    def forward(self, w, forcing, mean):
        n = w.shape[-1]
        channels = torch.cat(
            (
                w[:, None] / 2.0,
                forcing[:, None] / 0.35,
                mean[:, :, None, None].expand(-1, -1, n, n) / 2.0,
            ),
            dim=1,
        )
        delta = self.fno(channels)[:, 0] * 2.0 * FORECAST_DT
        return w + delta - delta.mean(dim=(-2, -1), keepdim=True)


def frequencies(n, dev):
    kx = (2 * torch.pi * torch.fft.rfftfreq(n, d=DOMAIN / n, device=dev))[None, :]
    ky = (2 * torch.pi * torch.fft.fftfreq(n, d=DOMAIN / n, device=dev))[:, None]
    k2 = kx * kx + ky * ky
    inv = torch.where(k2 > 0, 1.0 / torch.clamp(k2, min=1e-12), 0.0)
    return kx, ky, k2, inv


def velocity(w, mean):
    n = w.shape[-1]
    kx, ky, _, inv = frequencies(n, w.device)
    wh = torch.fft.rfft2(w)
    u = torch.fft.irfft2(1j * ky * inv * wh, s=(n, n)) + mean[:, 0, None, None]
    v = torch.fft.irfft2(-1j * kx * inv * wh, s=(n, n)) + mean[:, 1, None, None]
    return torch.stack((u, v), dim=1)


def pde_rhs(w, f, mean):
    n = w.shape[-1]
    kx, ky, k2, inv = frequencies(n, w.device)
    wh = torch.fft.rfft2(w)
    u = torch.fft.irfft2(1j * ky * inv * wh, s=(n, n)) + mean[:, 0, None, None]
    v = torch.fft.irfft2(-1j * kx * inv * wh, s=(n, n)) + mean[:, 1, None, None]
    dx = torch.fft.irfft2(1j * kx * wh, s=(n, n))
    dy = torch.fft.irfft2(1j * ky * wh, s=(n, n))
    rhs = -torch.fft.rfft2(u * dx + v * dy) - VISCOSITY * k2 * wh + torch.fft.rfft2(f)
    mask = (abs(kx) < 2 * torch.pi / DOMAIN * (n / 3)) & (
        abs(ky) < 2 * torch.pi / DOMAIN * (n / 3)
    )
    return torch.fft.irfft2(rhs * mask, s=(n, n))


def upsample(w, n):
    """Periodic Fourier interpolation for collocation inputs, never fine labels."""
    old = w.shape[-1]
    wh = torch.fft.fftshift(torch.fft.fft2(w), dim=(-2, -1))
    p = (n - old) // 2
    padded = torch.nn.functional.pad(wh, (p, p, p, p))
    return (
        torch.fft.ifft2(torch.fft.ifftshift(padded, dim=(-2, -1))).real * (n / old) ** 2
    )


def load(path, dev="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FlowOperator(**checkpoint["config"])
    model.load_state_dict(checkpoint["state_dict"])
    model.to(dev).eval()
    return model, checkpoint


def train(
    data_path,
    output,
    epochs=40,
    batch_size=32,
    backend="auto",
    width=16,
    modes=12,
    physics_training=True,
):
    torch.manual_seed(7)
    torch.set_num_threads(4)
    dev = device(backend)
    source = np.load(data_path)
    seeds = source["seeds"]
    train_ids = np.flatnonzero(seeds < 100)
    val_ids = np.flatnonzero((seeds >= 100) & (seeds < 200))
    w = torch.from_numpy(source["omega"])
    f = torch.from_numpy(source["forcing"])
    mean = torch.from_numpy(source["mean"])
    steps = w.shape[1] - 1
    x = w[train_ids, :-1].reshape(-1, w.shape[-2], w.shape[-1]).to(dev)
    y = w[train_ids, 1:].reshape(-1, w.shape[-2], w.shape[-1]).to(dev)
    forces = f[train_ids].repeat_interleave(steps, dim=0).to(dev)
    means = mean[train_ids].repeat_interleave(steps, dim=0).to(dev)
    vx = w[val_ids, :-1:4].flatten(0, 1).to(dev)
    vy = w[val_ids, 1::4].flatten(0, 1).to(dev)
    vf = f[val_ids].repeat_interleave(w[:, :-1:4].shape[1], dim=0).to(dev)
    vm = mean[val_ids].repeat_interleave(w[:, :-1:4].shape[1], dim=0).to(dev)
    model = FlowOperator(width=width, modes=modes).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0015)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    begin = time.monotonic()
    best = float("inf")
    pino_start = epochs // 2
    print(
        f"Training {sum(p.numel() for p in model.parameters())} parameters on {dev}; {len(x)} training pairs",
        flush=True,
    )
    for epoch in range(epochs):
        physics = physics_training and epoch >= pino_start
        if epoch == pino_start:
            best = float("inf")
            for group in optimizer.param_groups:
                group["lr"] = 0.0006
        model.train()
        order = torch.randperm(len(x), device=dev)
        losses = []
        residuals = []
        for bi, start in enumerate(range(0, len(x), batch_size)):
            ids = order[start : start + batch_size]
            a, b, ff, mm = x[ids], y[ids], forces[ids], means[ids]
            pred = model(a, ff, mm)
            denom = (b - a).square().mean().detach().clamp(min=0.01)
            loss = (pred - b).square().mean() / denom
            # A velocity loss emphasizes the field actually used by the robot.
            loss += (
                0.3
                * (velocity(pred, mm) - velocity(b, mm)).square().mean()
                / (
                    (velocity(b, mm) - velocity(a, mm))
                    .square()
                    .mean()
                    .detach()
                    .clamp(min=0.005)
                )
            )
            if physics and bi % 4 == 0:
                fine = upsample(a[:4], 128)
                fine_f = upsample(ff[:4], 128)
                fm = mm[:4]
                fp = model(fine, fine_f, fm)
                rhs = 0.5 * (pde_rhs(fine, fine_f, fm) + pde_rhs(fp, fine_f, fm))
                residual = (
                    (fp - fine) / FORECAST_DT - rhs
                ).square().mean() / rhs.detach().square().mean().clamp(min=0.01)
                loss += 0.03 * residual
                residuals.append(float(residual.detach().cpu()))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        errors = []
        persistence = []
        with torch.no_grad():
            for start in range(0, len(vx), batch_size):
                sl = slice(start, start + batch_size)
                pred = model(vx[sl], vf[sl], vm[sl])
                truth = vy[sl]
                errors.append(float((pred - truth).square().sum().cpu()))
                persistence.append(float((vx[sl] - truth).square().sum().cpu()))
        score = float(np.sqrt(sum(errors) / sum(persistence)))
        entry = {
            "epoch": epoch + 1,
            "kind": "PINO" if physics else "FNO",
            "loss": float(np.mean(losses)),
            "physics_loss": float(np.mean(residuals)) if residuals else None,
            "validation_error_vs_persistence": score,
            "elapsed_s": time.monotonic() - begin,
        }
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if score < best:
            best = score
            checkpoint = {
                "config": model.config,
                "state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                    if isinstance(v, torch.Tensor)
                },
                "epoch": epoch + 1,
                "kind": entry["kind"],
                "validation_error_vs_persistence": score,
                "train_seeds": seeds[train_ids].tolist(),
                "validation_seeds": seeds[val_ids].tolist(),
                "forecast_dt": FORECAST_DT,
                "training_resolution": 64,
                "physics_resolution": 128 if physics else None,
                "backend": str(dev),
            }
            torch.save(checkpoint, output / ("pino.pt" if physics else "fno.pt"))
        (output / "training.json").write_text(json.dumps(history, indent=2) + "\n")
    return history[-1]
