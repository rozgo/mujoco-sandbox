"""Time-budgeted JEPA and differentiable acceleration-residual training."""

import argparse
import copy
import json
import math
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.world_data import HISTORY
from embodied_fly.world_model import physical_errors
from embodied_fly.world_residual_model import ResidualWorldModel
from embodied_fly.world_residual_physics import DifferentiableFly
from embodied_fly.world_train import Corpus, validate


class InterventionCorpus:
    def __init__(self, folder, corpus):
        self.manifest = json.loads((folder / "manifest.json").read_text())
        if self.manifest["dataset_sha256"] != sha256(corpus.folder / "manifest.json"):
            raise ValueError("Interventions have different source data")
        self.device = corpus.device
        self.arrays, self.rows = {}, {}
        for split in ("train", "validation"):
            rows = [r for r in self.manifest["records"] if r["split"] == split]
            if not rows or any(
                corpus.records[i]["episode"] in (8, 9) for i in corpus.ranges[split]
            ):
                raise ValueError("Invalid intervention split")
            values = {k: [] for k in ("sequence", "actions", "initial", "target", "inertia")}
            for row in rows:
                file = folder / row["file"]
                if row["episode"] in (8, 9) or sha256(file) != row["sha256"]:
                    raise ValueError("Held-out episode or changed intervention capture")
                with np.load(file) as z:
                    for k, destination in values.items():
                        destination.append(
                            np.repeat(z[k][None], 16, 0) if k == "inertia" else z[k]
                        )
            self.arrays[split] = {
                k: torch.as_tensor(np.concatenate(v), device=self.device, dtype=torch.float32)
                for k, v in values.items()
            }
            self.rows[split] = rows

    def sample(self, split, count, rng):
        arrays = self.arrays[split]
        indices = torch.as_tensor(
            rng.integers(len(arrays["initial"]), size=count), device=self.device
        )
        return {k: v[indices] for k, v in arrays.items() if k != "inertia"}


def combined_batch(corpus, interventions, split, count, rng):
    a = corpus.select(split, count // 2, rng)
    b = interventions.sample(split, count - count // 2, rng)
    return {k: torch.cat((a[k], b[k]), 0) for k in a}


@torch.no_grad()
def make_bank(model, plant, corpus, interventions, split, count, rng):
    original = corpus.select(split, count, rng)
    qpos = torch.cat(
        (
            original["initial"][:, :3],
            original["sequence"][:, HISTORY - 1, : plant.analytic.model.nq - 3],
        ),
        -1,
    )
    inertia, _ = plant.inertia(qpos.cpu().numpy(), original["initial"])
    original["inertia"] = inertia
    source = interventions.arrays[split]
    all_data = {k: torch.cat((v, source[k]), 0) for k, v in original.items()}
    cached = {k: all_data[k] for k in ("initial", "target", "inertia")}
    cached["inverse"] = torch.linalg.inv(cached["inertia"])
    latent, wings = [], []
    for start in range(0, len(cached["initial"]), 128):
        sl = slice(start, start + 128)
        latent.append(
            model.latent_rollout(all_data["sequence"][sl, :HISTORY], all_data["actions"][sl])
        )
        wings.append(plant.wing_rollout(all_data["initial"][sl], all_data["actions"][sl]))
    cached["latent"], cached["wings"] = torch.cat(latent), torch.cat(wings)
    return cached, {
        "original_windows": count,
        "intervention_groups": len(interventions.rows[split]),
        "total_windows": len(cached["initial"]),
    }


def bank_indices(info, batch, rng, device):
    # Half ordinary flight, one quarter unchanged branches, one quarter matched
    # changed branches; pair positions are fixed for the effect objective.
    half, quarter = batch // 2, batch // 4
    original = rng.integers(info["original_windows"], size=half)
    group = rng.integers(info["intervention_groups"], size=quarter)
    base = info["original_windows"] + group * 16
    variant = base + rng.integers(1, 15, size=quarter)
    return torch.as_tensor(np.r_[original, base, variant], device=device)


def physical_loss(model, plant, batch):
    residual = model.accelerations(batch["latent"])
    prediction, _ = plant(
        batch["initial"], batch["wings"], batch["inertia"], batch["inverse"], residual
    )
    error = (prediction[..., :18] - batch["target"][..., :18]) / model.metric_scale[:18]
    ordinary = (
        sum(
            error[..., sl].square().mean()
            for sl in (slice(0, 3), slice(3, 6), slice(6, 15), slice(15, 18))
        )
        / 4
    )
    half, quarter = len(prediction) // 2, len(prediction) // 4
    delta = (prediction[half + quarter :] - prediction[half : half + quarter]) - (
        batch["target"][half + quarter :] - batch["target"][half : half + quarter]
    )
    paired = (
        (delta[..., 3:6] / 0.1).square().mean() + (delta[..., 15:18] / 0.1).square().mean()
    ) / 2
    loss = ordinary + 0.1 * paired + 1e-4 * (residual / model.residual_scale).square().mean()
    return loss


@torch.no_grad()
def physical_validation(model, plant, bank, original_count):
    predicted, actual = [], []
    for start in range(0, len(bank["initial"]), 128):
        b = {k: v[start : start + 128] for k, v in bank.items()}
        path, _ = plant(
            b["initial"],
            b["wings"],
            b["inertia"],
            b["inverse"],
            model.accelerations(b["latent"]),
        )
        predicted.append(path[:, [24, 49, 74, 99]].cpu().numpy())
        actual.append(b["target"][:, [24, 49, 74, 99]].cpu().numpy())
    pred, truth = np.concatenate(predicted), np.concatenate(actual)
    ordinary = physical_errors(pred[:original_count], truth[:original_count])
    variants = physical_errors(pred[original_count:], truth[original_count:])
    return {
        "selection": (ordinary["velocity_mm_s"] + variants["velocity_mm_s"]) / 2,
        "recorded_flight": ordinary,
        "interventions": variants,
    }


def run(args):
    if args.batch % 4 or min(args.latent_seconds, args.prober_seconds) <= 0:
        raise ValueError("Positive budgets and a batch divisible by four required")
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    corpus = Corpus(args.dataset, device)
    interventions = InterventionCorpus(args.interventions, corpus)
    model = ResidualWorldModel().to(device)
    model.mean[:], model.scale[:] = corpus.normalization()
    mjmodel = mujoco.MjModel.from_binary_path(
        str(args.root / "velocity_teacher_dataset_01/model.mjb")
    )
    plant = DifferentiableFly(mjmodel).to(device=device, dtype=torch.float32)
    if corpus.manifest["physical_contract"] != interventions.manifest["physical_contract"]:
        raise ValueError("Physical contracts differ")
    from embodied_fly.physical_contract import physical_contract

    if physical_contract(mjmodel) != corpus.manifest["physical_contract"]:
        raise ValueError("Integrator physical contract differs")
    rng, vrng = np.random.default_rng(args.seed), np.random.default_rng(391202)
    validation = [
        combined_batch(corpus, interventions, "validation", 64, vrng) for _ in range(4)
    ]
    directions = torch.randn(
        64, 32, generator=torch.Generator(device=device).manual_seed(391203), device=device
    )
    synchronize(device)
    report = {
        "schema": "fly-residual-training-v1",
        "provenance": evidence(),
        "started_utc": utc_now(),
        "config": {k: v.name if isinstance(v, Path) else v for k, v in vars(args).items()},
        "physical_contract": corpus.manifest["physical_contract"],
        "dataset_sha256": sha256(args.dataset / "manifest.json"),
        "interventions_sha256": sha256(args.interventions / "manifest.json"),
        "setup_seconds": time.perf_counter() - begin,
        "stages": {},
    }

    def save(stage, step):
        torch.save(
            {
                "format": "integrated-acceleration-residual-v1",
                "config": model.config,
                "state_dict": model.state_dict(),
                "physical_contract": report["physical_contract"],
                "dataset_sha256": report["dataset_sha256"],
                "interventions_sha256": report["interventions_sha256"],
                "stage": stage,
                "step": step,
                "seed": args.seed,
            },
            args.output / f"{stage}.pt",
        )

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("prober."))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=1e-5
    )
    training, evaluation, step, last_eval = 0.0, 0.0, 0, -15.0
    best, best_state, best_step, records = float("inf"), None, 0, []
    while training < args.latent_seconds:
        start = time.perf_counter()
        progress = min(training / args.latent_seconds, 1)
        optimizer.param_groups[0]["lr"] = (
            0.0001 + 0.0009 * (1 + math.cos(math.pi * progress)) / 2
        )
        batch = combined_batch(corpus, interventions, "train", args.batch, rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.latent_loss(batch["sequence"], batch["actions"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        synchronize(device)
        training += time.perf_counter() - start
        step += 1
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite latent loss")
        if training - last_eval >= 15 or training >= args.latent_seconds:
            start = time.perf_counter()
            result = validate(model, validation, "latent", directions)
            if result["selection"] < best:
                best, best_state, best_step = (
                    result["selection"],
                    copy.deepcopy(model.state_dict()),
                    step,
                )
            row = {
                "stage": "latent",
                "step": step,
                "training_seconds": training,
                "loss": loss.item(),
                "validation": result,
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            synchronize(device)
            evaluation += time.perf_counter() - start
            last_eval = training
    model.load_state_dict(best_state)
    save("latent", best_step)
    report["stages"]["latent"] = {
        "steps": step,
        "best_step": best_step,
        "training_seconds": training,
        "validation_seconds": evaluation,
        "presentations": step * args.batch,
        "records": records,
    }
    del optimizer, best_state

    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("prober."))
    frozen = {
        k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad
    }
    start = time.perf_counter()
    bank, info = make_bank(
        model, plant, corpus, interventions, "train", args.bank_windows, rng
    )
    valbank, valinfo = make_bank(model, plant, corpus, interventions, "validation", 128, vrng)
    synchronize(device)
    report["bank_preparation_seconds"] = time.perf_counter() - start
    report["banks"] = {"train": info, "validation": valinfo}
    print(
        json.dumps(
            {
                "bank_preparation_seconds": report["bank_preparation_seconds"],
                "banks": report["banks"],
            }
        ),
        flush=True,
    )
    start = time.perf_counter()
    initial_validation = physical_validation(model, plant, valbank, 128)
    synchronize(device)
    evaluation = time.perf_counter() - start
    best, best_state, best_step = (
        initial_validation["selection"],
        copy.deepcopy(model.state_dict()),
        0,
    )
    records = [
        {"stage": "prober", "step": 0, "training_seconds": 0, "validation": initial_validation}
    ]
    save("zero_residual", 0)
    print(json.dumps(records[-1]), flush=True)

    indices = bank_indices(info, args.batch, rng, device)
    static = {k: v[indices].clone() for k, v in bank.items()}
    optimizer = torch.optim.AdamW(
        model.prober.parameters(), lr=0.001, weight_decay=1e-5, fused=device.type == "cuda"
    )
    start = time.perf_counter()
    graph = None
    if device.type == "cuda" and not args.no_cuda_graph:
        # Warm up forward/backward only; no optimizer updates or hidden training.
        for _ in range(2):
            model.zero_grad(set_to_none=False)
            loss = physical_loss(model, plant, static)
            loss.backward()
        eager_loss = loss.detach().clone()
        eager_gradients = [p.grad.detach().clone() for p in model.prober.parameters()]
        synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            model.zero_grad(set_to_none=False)
            loss = physical_loss(model, plant, static)
            loss.backward()
        graph.replay()
        synchronize(device)
        torch.testing.assert_close(loss, eager_loss, atol=1e-7, rtol=1e-5)
        for parameter, expected in zip(
            model.prober.parameters(), eager_gradients, strict=True
        ):
            torch.testing.assert_close(parameter.grad, expected, atol=1e-6, rtol=1e-4)
        report["cuda_graph_matches_eager_loss_and_gradients"] = True
    if not all(torch.equal(v, model.state_dict()[k]) for k, v in best_state.items()):
        raise RuntimeError("Graph setup unexpectedly changed weights")
    report["graph_setup_seconds"] = time.perf_counter() - start
    report["cuda_graph"] = graph is not None
    print(
        json.dumps(
            {
                "graph_setup_seconds": report["graph_setup_seconds"],
                "cuda_graph": graph is not None,
            }
        ),
        flush=True,
    )

    training, step, last_eval = 0.0, 0, 0.0
    while training < args.prober_seconds:
        start = time.perf_counter()
        optimizer.param_groups[0]["lr"] = (
            0.0001
            + 0.0009 * (1 + math.cos(math.pi * min(training / args.prober_seconds, 1))) / 2
        )
        indices = bank_indices(info, args.batch, rng, device)
        for k, v in static.items():
            v.copy_(bank[k][indices])
        if graph is not None:
            graph.replay()
        else:
            model.zero_grad(set_to_none=False)
            loss = physical_loss(model, plant, static)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.prober.parameters(), 1.0)
        optimizer.step()
        synchronize(device)
        training += time.perf_counter() - start
        step += 1
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite integrated residual loss")
        if training - last_eval >= 15 or training >= args.prober_seconds:
            start = time.perf_counter()
            result = physical_validation(model, plant, valbank, 128)
            if result["selection"] < best:
                best, best_state, best_step = (
                    result["selection"],
                    copy.deepcopy(model.state_dict()),
                    step,
                )
            row = {
                "stage": "prober",
                "step": step,
                "training_seconds": training,
                "loss": loss.item(),
                "validation": result,
            }
            records.append(row)
            print(json.dumps(row), flush=True)
            synchronize(device)
            evaluation += time.perf_counter() - start
            last_eval = training
    if not all(torch.equal(p, frozen[k]) for k, p in model.named_parameters() if k in frozen):
        raise RuntimeError("Frozen latent model changed during physical-probe training")
    save("final_prober", step)
    model.load_state_dict(best_state)
    save("prober", best_step)
    report["stages"]["prober"] = {
        "steps": step,
        "best_step": best_step,
        "training_seconds": training,
        "validation_seconds": evaluation,
        "presentations": step * args.batch,
        "frozen_parameters_unchanged": True,
        "records": records,
    }
    report.update(
        {
            "completed_utc": utc_now(),
            "total_wall_seconds": time.perf_counter() - begin,
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_sha256": sha256(args.output / "prober.pt"),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0,
        }
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--interventions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("outputs/embodied_fly"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=391201)
    p.add_argument("--latent-seconds", type=float, default=60)
    p.add_argument("--prober-seconds", type=float, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--bank-windows", type=int, default=1024)
    p.add_argument("--no-cuda-graph", action="store_true")
    run(p.parse_args())
