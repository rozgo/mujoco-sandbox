"""Calibrate only six existing wing-output rows using frozen graph features.

No additional deployed module or observation-to-action bypass is introduced.
All non-wing outputs and every preceding actor parameter remain exactly intact.
Physical evaluation is required; expert-history loss does not prove flight.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def replace_wing_rows(parent, weights, bias, wings):
    state = {k: v.clone() for k, v in parent.items()}
    for name, value in (("motor_decoder.3.weight", weights), ("motor_decoder.3.bias", bias)):
        state[name][wings] = value.detach().cpu()
    return state


def train(args):
    if min(args.seconds, args.lr, args.batch_size) <= 0:
        raise ValueError("Positive learning duration, rate and batch size required")
    args.output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    parent = torch.load(args.resume, map_location="cpu", weights_only=True)
    diagnostic = json.loads(args.cache_report.read_text())
    if diagnostic["checkpoint_sha256"] != sha256(args.resume) or diagnostic[
        "frozen_feature_cache"
    ]["sha256"] != sha256(args.cache):
        raise ValueError("Frozen features do not belong to this checkpoint/cache")
    with np.load(args.cache, allow_pickle=False) as data:
        wings = data["wing_channels"]
        train_ids, test_ids = data["training_indices"], data["validation_indices"]
        if set(train_ids) & set(test_ids):
            raise ValueError("Training and validation episodes overlap")
        features, target = data["hidden"], data["target"]
        x = torch.as_tensor(features[:, train_ids].reshape(-1, 256), device=device)
        y = torch.as_tensor(target[:, train_ids].reshape(-1, 6), device=device)
        test_x = torch.as_tensor(features[:, test_ids].reshape(-1, 256), device=device)
        test_y = torch.as_tensor(target[:, test_ids].reshape(-1, 6), device=device)
    # These temporary optimization tensors are the six existing output rows.
    # Their learned values are copied back into the same full 78-output layer.
    weights = nn.Parameter(
        parent["state_dict"]["motor_decoder.3.weight"][wings].to(device).clone()
    )
    bias = nn.Parameter(parent["state_dict"]["motor_decoder.3.bias"][wings].to(device).clone())
    optimizer = torch.optim.Adam([weights, bias], lr=args.lr)

    def loss(inputs, labels):
        return F.mse_loss(F.linear(inputs, weights, bias).tanh(), labels)

    with torch.no_grad():
        initial = float(loss(test_x, test_y))
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    started_utc = utc_now()
    started = time.perf_counter()
    updates = 0
    while time.perf_counter() - started < args.seconds:
        indices = torch.randint(len(x), (args.batch_size,), device=device)
        value = loss(x[indices], y[indices])
        if not torch.isfinite(value):
            raise RuntimeError("Nonfinite wing calibration loss")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
        updates += 1
        synchronize(device)
    training_seconds = time.perf_counter() - started
    evaluation_start = time.perf_counter()
    with torch.no_grad():
        final = float(loss(test_x, test_y))
    state = replace_wing_rows(parent["state_dict"], weights, bias, wings)
    allowed = {"motor_decoder.3.weight", "motor_decoder.3.bias"}
    for name, value in state.items():
        if name not in allowed and not torch.equal(value, parent["state_dict"][name]):
            raise RuntimeError("Unexpected change outside the existing wing output rows")
    other = np.array([i for i in range(parent["action_size"]) if i not in wings])
    for name in allowed:
        if not torch.equal(state[name][other], parent["state_dict"][name][other]):
            raise RuntimeError("Non-wing motor outputs changed")
    checkpoint = {
        key: parent[key]
        for key in (
            "observation_size",
            "sensor_extension_size",
            "action_size",
            "config",
            "graph_sha256",
            "graph_metadata_sha256",
        )
    }
    checkpoint.update(
        state_dict=state,
        source_commit=provenance["source_commit"],
        method="frozen-core supervised wing output calibration",
        parent_checkpoint_sha256=sha256(args.resume),
    )
    torch.save(checkpoint, args.output / "actor.pt")
    report = {
        "provenance": provenance,
        "training_started_utc": started_utc,
        "completed_utc": utc_now(),
        "method": checkpoint["method"],
        "parent_checkpoint_sha256": sha256(args.resume),
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "cache_report_sha256": sha256(args.cache_report),
        "cache_sha256": sha256(args.cache),
        "seed": args.seed,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "training_episode_indices": train_ids.tolist(),
        "validation_episode_indices": test_ids.tolist(),
        "unique_training_frames": len(x),
        "updates": updates,
        "sampled_training_examples": updates * args.batch_size,
        "trained_existing_parameters": weights.numel() + bias.numel(),
        "new_deployed_parameters": 0,
        "wing_output_indices": wings.tolist(),
        "all_other_actor_parameters_unchanged": True,
        "non_wing_output_rows_unchanged": True,
        "live_physics_worlds": 0,
        "initial_validation_wing_mse": initial,
        "final_validation_wing_mse": final,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "evaluation_and_save_seconds": time.perf_counter() - evaluation_start,
        "neural_device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "physical_success": "Not established; live evaluation required",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cache-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())
