"""Bounded offline wing-readout learning from an audited motor-cell feature cache.

Only the new feedforward decoder is fitted. The recorded recurrent histories are
fixed, so physical closed-loop evaluation remains mandatory after this fit.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.brain import NonlinearWingReadout
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def read_cache(root, resume, parent):
    report = json.loads((root / "report.json").read_text())
    if report["parent_checkpoint_sha256"] != sha256(resume):
        raise ValueError("Feature cache came from a different frozen parent")
    if report["physical_contract"] != parent["physical_contract"]:
        raise ValueError("Feature cache physical contract differs")
    if report["feature_layer"] != "motor":
        raise ValueError("Normalized motor-cell features required")
    if report["features_sha256"] != sha256(root / "features.npz"):
        raise ValueError("Feature cache hash differs")
    with np.load(root / "features.npz") as data:
        arrays = {
            k: data[k]
            for k in (
                "hidden",
                "logits",
                "target",
                "train_mask",
                "validation_mask",
                "frame_weights",
            )
        }
    x, y, b = arrays["hidden"], arrays["target"], arrays["logits"]
    shape = x.shape[:2]
    if (
        x.ndim != 3
        or x.shape[-1] != len(parent["state_dict"]["motor_ids"])
        or y.shape != (*shape, 6)
        or b.shape != y.shape
    ):
        raise ValueError("Invalid motor feature or wing label dimensions")
    for key in ("train_mask", "validation_mask", "frame_weights"):
        if arrays[key].shape != shape:
            raise ValueError("Invalid cache masks/weights")
    train, valid = arrays["train_mask"], arrays["validation_mask"]
    if train.dtype != bool or valid.dtype != bool or (train & valid).any():
        raise ValueError("Training and validation masks must be disjoint booleans")
    if not train.any(axis=0).all() or not valid.any(axis=0).all():
        raise ValueError("Every history needs train and validation frames")
    if (
        not all(np.isfinite(v).all() for v in arrays.values())
        or (arrays["frame_weights"] <= 0).any()
        or (np.abs(y) > 1).any()
    ):
        raise ValueError("Nonfinite or invalid cached data")
    if (
        int(train.sum()) != report["training_frames"]
        or int(valid.sum()) != report["validation_frames"]
    ):
        raise ValueError("Cache frame counts differ from its report")
    return arrays, report


def train(args):
    if not (
        args.seconds > 0
        and args.max_updates > 0
        and args.hidden > 0
        and args.batch_size > 0
        and args.learning_rate > 0
    ):
        raise ValueError("Positive fit limits and dimensions required")
    started, started_utc = time.perf_counter(), utc_now()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    parent = torch.load(args.resume, map_location="cpu", weights_only=True)
    if (
        not parent["motor_only"]
        or parent["observation_size"] != 397
        or parent["action_size"] != 78
        or parent.get("wing_residual_enabled", False)
    ):
        raise ValueError("Unextended canonical motor-only parent required")
    data, cache = read_cache(args.cache, args.resume, parent)
    args.output.mkdir(parents=True, exist_ok=False)
    x, base, target, weights = [
        torch.as_tensor(data[k], dtype=torch.float32, device=device)
        for k in ("hidden", "logits", "target", "frame_weights")
    ]
    fit = torch.as_tensor(data["train_mask"], device=device)
    valid = torch.as_tensor(data["validation_mask"], device=device)
    history_weights = torch.as_tensor(cache["history_weights"], device=device)
    module = NonlinearWingReadout(x.shape[-1], args.hidden).to(device)
    with torch.no_grad():
        module.feature_mean.copy_(x[fit].mean(dim=0))
        module.feature_scale.copy_(x[fit].std(dim=0, unbiased=False).clamp_min(0.05))
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    train_x, train_base, train_y, train_w = x[fit], base[fit], target[fit], weights[fit]
    mean_weight = train_w.mean()

    @torch.no_grad()
    def metrics():
        error = ((base + module(x)).tanh() - target).square().mean(dim=-1)
        a = torch.stack([error[fit[:, j], j].mean() for j in range(x.shape[1])])
        b = torch.stack([error[valid[:, j], j].mean() for j in range(x.shape[1])])
        return {
            "train_action_mse_by_history": a.cpu().tolist(),
            "validation_action_mse_by_history": b.cpu().tolist(),
            "selection_score": float((b * history_weights).sum() / history_weights.sum()),
        }

    initial = metrics()
    best, best_update = initial, 0
    best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    fit_start = time.perf_counter()
    trace = []
    update = 0
    while update < args.max_updates and time.perf_counter() - fit_start < args.seconds:
        ids = torch.randint(len(train_x), (args.batch_size,), device=device)
        prediction = (train_base[ids] + module(train_x[ids])).tanh()
        loss = (
            (prediction - train_y[ids]).square().mean(dim=-1) * train_w[ids]
        ).mean() / mean_weight
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite decoder fitting loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
        optimizer.step()
        update += 1
        if update % 100 == 0 or update == args.max_updates:
            entry = metrics()
            entry.update(update=update, elapsed_seconds=time.perf_counter() - fit_start)
            trace.append(entry)
            if entry["selection_score"] < best["selection_score"]:
                best, best_update = entry, update
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            if update % 2000 == 0:
                print(json.dumps(entry), flush=True)
    # Score the terminal partial interval too, without extending the optimizer budget.
    terminal = metrics()
    if terminal["selection_score"] < best["selection_score"]:
        best, best_update = terminal, update
        best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
    synchronize(device)
    fitting_seconds = time.perf_counter() - fit_start
    module.load_state_dict(best_state)
    with torch.no_grad():
        predictions = (base + module(x)).tanh().cpu().numpy()
    state = {k: v.clone() for k, v in parent["state_dict"].items()}
    state.update({"wing_residual." + k: v.cpu() for k, v in best_state.items()})
    for k, value in parent["state_dict"].items():
        assert torch.equal(value, state[k]), k
    child = {
        k: parent[k]
        for k in (
            "observation_size",
            "sensor_extension_size",
            "action_size",
            "config",
            "graph_sha256",
            "graph_metadata_sha256",
            "physical_contract",
            "motor_only",
        )
    }
    child.update(
        state_dict=state,
        wing_residual_enabled=True,
        wing_residual_hidden=args.hidden,
        source_commit=provenance["source_commit"],
        parent_checkpoint_sha256=sha256(args.resume),
        method="nonlinear wing readout from frozen motor-cell histories; original actor unchanged",
    )
    torch.save(child, args.output / "actor.pt")
    np.savez_compressed(args.output / "predictions.npz", prediction=predictions)
    (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
    report = {
        "provenance": provenance,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "method": child["method"],
        "parent_checkpoint_sha256": sha256(args.resume),
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "physical_contract": parent["physical_contract"],
        "cache_report_sha256": sha256(args.cache / "report.json"),
        "features_sha256": cache["features_sha256"],
        "sources": cache["sources"],
        "predictions_sha256": sha256(args.output / "predictions.npz"),
        "trace_sha256": sha256(args.output / "trace.json"),
        "seed": args.seed,
        "hidden": args.hidden,
        "features": x.shape[-1],
        "trainable_parameters": sum(p.numel() for p in module.parameters()),
        "normalization": "training frames only, per-feature mean/std, scale floor0.05, clip10",
        "loss": "weighted bounded wing action MSE; AdamW; grad norm1; weight decay1e-4",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "updates": update,
        "selected_update": best_update,
        "max_updates": args.max_updates,
        "fit_budget_seconds": args.seconds,
        "initial": initial,
        "selected": best,
        "terminal": terminal,
        "history_weights": cache["history_weights"],
        "validation_split": cache["validation_split"],
        "training_frames": int(fit.sum()),
        "validation_frames": int(valid.sum()),
        "physical_transitions_collected": 0,
        "live_worlds": 0,
        "new_graph_replay_frames": 0,
        "runtime_teacher": False,
        "all_original_state_unchanged": True,
        "setup_seconds": setup_seconds,
        "fitting_seconds": fitting_seconds,
        "total_wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "physical_success": "Unproven until unassisted three-task evaluation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k not in ("provenance", "sources")}),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("resume", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--max-updates", type=int, default=18000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=72008)
    train(parser.parse_args())
