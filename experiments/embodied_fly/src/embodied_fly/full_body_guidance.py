"""Local model-based policy improvement of one shared 78-output decoder.

The predictor supplies gradients during training only. Actual parent neural
histories are fixed in these local windows; complete live-brain MuJoCo trials
must validate transfer. No action or parameter selection by body part.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.full_body_decoder import FullBodyDecoder
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.world_data import HISTORY, HORIZON
from embodied_fly.world_residual_model import ResidualWorldModel
from embodied_fly.world_residual_physics import DifferentiableFly


def motion_cost(path, lift):
    """Phase-independent physical objective; no reference wing angles."""
    mean_velocity = path[..., 3:6].reshape(-1, 2, 50, 3).mean(2)
    velocity = (mean_velocity / 0.5).square().sum(-1)
    velocity = (0.25 * velocity[:, 0] + 0.75 * velocity[:, 1]).mean()
    angular = (path[:, 50:, 15:18].mean(1) / 0.5).square().sum(-1).mean()
    tilt = (1 - path[:, 50:, 14]).square().mean()
    support = (0.8 - lift[:, 100:].mean(1)).clamp_min(0).square().mean()
    floor = (0.7 - path[..., 2]).clamp_min(0).square().mean()
    total = velocity + 0.05 * angular + 2 * tilt + 5 * support + 50 * floor
    return total, torch.stack((velocity, angular, tilt, support, floor))


class Guidance:
    def __init__(self, decoder, net, plant, analytical=False):
        self.decoder, self.net, self.plant = decoder, net, plant
        self.analytical = analytical

    def __call__(self, b):
        actions = self.decoder(b["motor"])
        if self.analytical:
            residual = torch.zeros_like(b["zero_residual"])
        else:
            latent, _ = self.net.predictor(
                self.net.action_encoder(actions), b["latent_start"].unsqueeze(0).contiguous()
            )
            residual = self.net.accelerations(latent)
        path, lift = self.plant(
            b["initial"],
            self.plant.wing_rollout(b["initial"], actions),
            b["inertia"],
            b["inverse"],
            residual,
        )
        physical, terms = motion_cost(path, lift)
        deviation = (actions - b["actions"]).square().mean()
        regularization = 0.1 * deviation / 0.005**2
        return physical + regularization, torch.cat(
            (terms, regularization[None], deviation.sqrt()[None])
        )


@torch.no_grad()
def project_to_parent(decoder, origin, calibration, base_actions, limit):
    """Training-only weight interpolation; the deployed actor has no limiter."""
    fraction = 1.0
    for _ in range(6):
        discrepancy = float((decoder(calibration) - base_actions).abs().max())
        if discrepancy <= limit * (1 + 1e-5):
            return discrepancy, fraction
        alpha = min(0.9 * limit / discrepancy, 0.9)
        for name, p in decoder.named_parameters():
            p.copy_(origin[name] + alpha * (p - origin[name]))
        fraction *= alpha
    raise RuntimeError("Could not satisfy calibration-bank action trust bound")


@torch.no_grad()
def prepare_bank(folder, report, split, net, plant, device):
    values = {k: [] for k in ("motor", "actions", "initial", "qpos", "history")}
    records = []
    for r in report["records"]:
        if r["split"] != split:
            continue
        if r["episode"] in (8, 9) or sha256(folder / r["file"]) != r["sha256"]:
            raise ValueError("Held-out or changed source episode")
        with np.load(folder / r["file"]) as z:
            starts = np.arange(0, len(z["actions"]) - HORIZON, 25)
            future = starts[:, None] + np.arange(HORIZON)
            raw = starts[:, None] + np.arange(-HISTORY + 1, 1)
            history = np.concatenate(
                (z["features"][np.maximum(raw, 0)], (raw >= 0)[..., None]), -1
            )
            values["motor"].append(z["motor"][future])
            values["actions"].append(z["actions"][future])
            values["initial"].append(z["metrics"][starts])
            values["qpos"].append(z["qpos"][starts])
            values["history"].append(history)
            records.append({"episode": r["episode"], "windows": len(starts)})
    all_values = {k: np.concatenate(v) for k, v in values.items()}
    bank = {
        k: torch.as_tensor(v, device=device, dtype=torch.float32)
        for k, v in all_values.items()
        if k not in ("qpos", "history")
    }
    bank["inertia"], bank["inverse"] = plant.inertia(all_values["qpos"], bank["initial"])
    bank["latent_start"] = torch.cat(
        [
            net.encode(
                torch.as_tensor(
                    all_values["history"][i : i + 128], device=device, dtype=torch.float32
                )
            )[:, -1]
            for i in range(0, len(bank["initial"]), 128)
        ]
    )
    bank["zero_residual"] = torch.zeros(len(bank["initial"]), HORIZON, 6, device=device)
    return bank, records


@torch.no_grad()
def validation(guidance, bank):
    ids = torch.linspace(0, len(bank["initial"]) - 1, 16, device=bank["initial"].device).long()
    loss, terms = guidance({k: v[ids] for k, v in bank.items()})
    return {
        "loss": float(loss),
        "velocity_cost": float(terms[0]),
        "angular_cost": float(terms[1]),
        "support_cost": float(terms[3]),
        "action_rms_change": float(terms[-1]),
    }


def run(args):
    begin = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    parent = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = json.loads((args.experience / "report.json").read_text())
    if (
        report["checkpoint_sha256"] != sha256(args.checkpoint)
        or report["physical_contract"] != parent["physical_contract"]
    ):
        raise ValueError("Experience came from another actor or physics")
    if parent.get("wing_residual_enabled") or parent.get("shared_decoder_hidden") != 384:
        raise ValueError("A shared full-body decoder is required")
    decoder = FullBodyDecoder(815, 384, 78).to(device)
    decoder.load_state_dict(
        {
            k.removeprefix("motor_decoder."): v
            for k, v in parent["state_dict"].items()
            if k.startswith("motor_decoder.")
        }
    )
    origin = {k: v.detach().clone() for k, v in decoder.state_dict().items()}
    saved = torch.load(args.world_model, map_location=device, weights_only=False)
    net = ResidualWorldModel(**saved["config"]).to(device)
    net.load_state_dict(saved["state_dict"])
    # CuDNN requires training-mode RNN execution for input gradients. The net
    # contains no dropout/BatchNorm and all its parameters remain frozen.
    net.train()
    net.requires_grad_(False)
    if saved["physical_contract"] != parent["physical_contract"]:
        raise ValueError("World model physics mismatch")
    model = mujoco.MjModel.from_binary_path(str(args.experience / "model.mjb"))
    if physical_contract(model) != parent["physical_contract"]:
        raise ValueError("Integrator physics mismatch")
    plant = DifferentiableFly(model).to(device=device, dtype=torch.float32)
    start = time.perf_counter()
    bank, train_rows = prepare_bank(args.experience, report, "train", net, plant, device)
    valbank, val_rows = prepare_bank(args.experience, report, "validation", net, plant, device)
    rng = np.random.default_rng(args.seed)
    flattened = bank["motor"].reshape(-1, 815)
    ids = torch.linspace(0, len(flattened) - 1, 1024, device=device).long()
    calibration = flattened[ids].clone()
    base_actions = decoder(calibration).detach()
    reconstruction = float((decoder(bank["motor"][:4]) - bank["actions"][:4]).abs().max())
    if reconstruction > 3e-6:
        raise ValueError(
            f"Raw neural history cannot reconstruct acting commands: {reconstruction}"
        )
    synchronize(device)
    bank_seconds = time.perf_counter() - start
    guidance = Guidance(decoder, net, plant, args.analytical)
    initial = validation(guidance, valbank)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, fused=device.type == "cuda")
    indices = torch.as_tensor(
        rng.integers(len(bank["initial"]), size=args.batch), device=device
    )
    static = {k: v[indices].clone() for k, v in bank.items()}
    start = time.perf_counter()
    graph = None
    if device.type == "cuda" and not args.no_cuda_graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                decoder.zero_grad(set_to_none=False)
                loss, _terms = guidance(static)
                loss.backward()
        torch.cuda.current_stream().wait_stream(stream)
        reference_loss = loss.detach().clone()
        reference_gradients = [p.grad.detach().clone() for p in decoder.parameters()]
        synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            decoder.zero_grad(set_to_none=False)
            loss, _terms = guidance(static)
            loss.backward()
        graph.replay()
        synchronize(device)
        torch.testing.assert_close(loss, reference_loss, atol=1e-6, rtol=1e-5)
        for p, expected in zip(decoder.parameters(), reference_gradients, strict=True):
            torch.testing.assert_close(p.grad, expected, atol=3e-5, rtol=5e-4)
    if not all(torch.equal(v, decoder.state_dict()[k]) for k, v in origin.items()):
        raise RuntimeError("Warmup changed decoder weights")
    setup_seconds = time.perf_counter() - start
    results = {
        "provenance": evidence(),
        "started_utc": utc_now(),
        "parent_checkpoint_sha256": sha256(args.checkpoint),
        "world_model_sha256": sha256(args.world_model),
        "experience_report_sha256": sha256(args.experience / "report.json"),
        "physical_contract": parent["physical_contract"],
        "method": "analytical" if args.analytical else "learned_residual",
        "recipe": {
            "batch": args.batch,
            "updates": args.updates,
            "lr": args.lr,
            "max_seconds": args.max_seconds,
            "max_action_change": args.max_action_change,
            "all_decoder_parameters": sum(p.numel() for p in decoder.parameters()),
            "all_outputs": 78,
            "seed": args.seed,
            "trust_weight": 0.1,
            "trust_scale": 0.005,
        },
        "training_windows": train_rows,
        "validation_windows": val_rows,
        "bank_seconds": bank_seconds,
        "graph_setup_seconds": setup_seconds,
        "cuda_graph_matches_eager": graph is not None,
        "cached_action_error": reconstruction,
        "initial_model_validation": initial,
        "snapshots": [],
        "records": [],
    }
    print(
        json.dumps(
            {k: v for k, v in results.items() if k not in ("provenance", "physical_contract")}
        ),
        flush=True,
    )
    training = 0.0
    projected_steps = 0

    def save(step):
        start_save = time.perf_counter()
        candidate = dict(parent)
        state = dict(parent["state_dict"])
        state.update(
            {
                "motor_decoder." + k: v.detach().cpu().clone()
                for k, v in decoder.state_dict().items()
            }
        )
        candidate.update(
            state_dict=state,
            parent_checkpoint_sha256=sha256(args.checkpoint),
            method="Shared full-body model-guided policy improvement",
            model_guidance={
                "method": results["method"],
                "step": step,
                "training_seconds": training,
                "world_model_sha256": results["world_model_sha256"],
                "live_world_model_control": False,
            },
        )
        file = args.output / f"actor_{step:04d}.pt"
        torch.save(candidate, file)
        value = validation(guidance, valbank)
        row = {
            "step": step,
            "training_seconds": training,
            "file": file.name,
            "sha256": sha256(file),
            "model_validation": value,
            "save_and_validation_seconds": time.perf_counter() - start_save,
        }
        results["snapshots"].append(row)
        print(json.dumps(row), flush=True)

    step = 0
    while step < args.updates and training < args.max_seconds:
        start = time.perf_counter()
        indices = torch.as_tensor(
            rng.integers(len(bank["initial"]), size=args.batch), device=device
        )
        for k, v in static.items():
            v.copy_(bank[k][indices])
        if graph is not None:
            graph.replay()
        else:
            decoder.zero_grad(set_to_none=False)
            loss, _terms = guidance(static)
            loss.backward()
        if not torch.isfinite(loss) or not all(
            torch.isfinite(p.grad).all() for p in decoder.parameters()
        ):
            raise RuntimeError("Nonfinite model-guided gradient")
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        discrepancy, fraction = project_to_parent(
            decoder, origin, calibration, base_actions, args.max_action_change
        )
        projected_steps += fraction < 1
        synchronize(device)
        training += time.perf_counter() - start
        step += 1
        if step % 10 == 0:
            row = {
                "step": step,
                "training_seconds": training,
                "loss": float(loss),
                "max_calibration_action_change": discrepancy,
                "projected_steps": projected_steps,
            }
            results["records"].append(row)
            print(json.dumps(row), flush=True)
        if step in (args.updates // 2, args.updates) or training >= args.max_seconds:
            save(step)
    assert all(torch.equal(v, net.state_dict()[k]) for k, v in saved["state_dict"].items())
    results.update(
        completed_utc=utc_now(),
        actual_updates=step,
        training_seconds=training,
        total_wall_seconds=time.perf_counter() - begin,
        world_model_unchanged=True,
        upstream_actor_unchanged=True,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
    )
    (args.output / "report.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--world-model", type=Path, required=True)
    p.add_argument("--experience", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--analytical", action="store_true")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--updates", type=int, default=100)
    p.add_argument("--max-seconds", type=float, default=180)
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--max-action-change", type=float, default=0.015)
    p.add_argument("--seed", type=int, default=391401)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-cuda-graph", action="store_true")
    run(p.parse_args())
