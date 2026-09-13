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


def load_feature_corpus(cache, cache_report, checkpoint_hash, device):
    diagnostic = json.loads(cache_report.read_text())
    if diagnostic["checkpoint_sha256"] != checkpoint_hash or diagnostic[
        "frozen_feature_cache"
    ]["sha256"] != sha256(cache):
        raise ValueError("Frozen features do not belong to this checkpoint/cache")
    with np.load(cache, allow_pickle=False) as data:
        wings = data["wing_channels"].copy()
        train_ids, test_ids = data["training_indices"], data["validation_indices"]
        features, target = data["hidden"], data["target"]
        if (
            features.ndim != 3
            or features.shape[-1] != 256
            or target.shape != features.shape[:2] + (6,)
        ):
            raise ValueError("Unexpected frozen motor feature or target shape")
        if not np.isfinite(features).all() or not np.isfinite(target).all():
            raise ValueError("Frozen feature corpus must be finite")
        for ids in (train_ids, test_ids):
            if (
                ids.ndim != 1
                or not len(ids)
                or len(np.unique(ids)) != len(ids)
                or np.any((ids < 0) | (ids >= features.shape[1]))
            ):
                raise ValueError("Invalid whole-episode split")
        if set(train_ids) & set(test_ids):
            raise ValueError("Training and validation episodes overlap")
        arrays = {
            "x": features[:, train_ids].reshape(-1, 256),
            "y": target[:, train_ids].reshape(-1, 6),
            "test_x": features[:, test_ids].reshape(-1, 256),
            "test_y": target[:, test_ids].reshape(-1, 6),
        }
        corpus = {key: torch.as_tensor(value, device=device) for key, value in arrays.items()}
        corpus["wings"] = wings
        corpus["report"] = {
            "cache_report_sha256": sha256(cache_report),
            "cache_sha256": sha256(cache),
            "training_episode_indices": train_ids.tolist(),
            "validation_episode_indices": test_ids.tolist(),
            "unique_training_frames": len(arrays["x"]),
            "validation_frames": len(arrays["test_x"]),
        }
    return corpus


def train(args):
    if min(args.seconds, args.lr, args.batch_size) <= 0:
        raise ValueError("Positive learning duration, rate and batch size required")
    if len(args.additional_cache) != len(args.additional_cache_report):
        raise ValueError("Every additional cache requires its matching report")
    args.output.mkdir(parents=True, exist_ok=False)
    setup_start = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    parent = torch.load(args.resume, map_location="cpu", weights_only=True)
    corpora = [
        load_feature_corpus(cache, report, sha256(args.resume), device)
        for cache, report in zip(
            [args.cache, *args.additional_cache],
            [args.cache_report, *args.additional_cache_report],
            strict=True,
        )
    ]
    wings = corpora[0]["wings"]
    if (
        len(wings) != 6
        or len(np.unique(wings)) != 6
        or np.any((wings < 0) | (wings >= parent["action_size"]))
    ):
        raise ValueError("Invalid six-wing output routing")
    if any(not np.array_equal(corpus["wings"], wings) for corpus in corpora):
        raise ValueError("Motor output routing differs between corpora")
    # These temporary optimization tensors are the six existing output rows.
    # Their learned values are copied back into the same full 78-output layer.
    weights = nn.Parameter(
        parent["state_dict"]["motor_decoder.3.weight"][wings].to(device).clone()
    )
    bias = nn.Parameter(parent["state_dict"]["motor_decoder.3.bias"][wings].to(device).clone())
    optimizer = torch.optim.Adam([weights, bias], lr=args.lr)

    def loss(inputs, labels):
        return F.mse_loss(F.linear(inputs, weights, bias).tanh(), labels)

    def validation_losses():
        with torch.no_grad():
            return [float(loss(corpus["test_x"], corpus["test_y"])) for corpus in corpora]

    initial_by_corpus = validation_losses()
    synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    started_utc = utc_now()
    started = time.perf_counter()
    updates = 0
    corpus_updates = [0] * len(corpora)
    while time.perf_counter() - started < args.seconds:
        choice = (
            0 if len(corpora) == 1 else int(torch.randint(len(corpora), (), device=device))
        )
        corpus = corpora[choice]
        x, y = corpus["x"], corpus["y"]
        indices = torch.randint(len(x), (args.batch_size,), device=device)
        value = loss(x[indices], y[indices])
        if not torch.isfinite(value):
            raise RuntimeError("Nonfinite wing calibration loss")
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        optimizer.step()
        updates += 1
        corpus_updates[choice] += 1
        synchronize(device)
    training_seconds = time.perf_counter() - started
    evaluation_start = time.perf_counter()
    final_by_corpus = validation_losses()
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
        "training_episode_indices": corpora[0]["report"]["training_episode_indices"],
        "validation_episode_indices": corpora[0]["report"]["validation_episode_indices"],
        "unique_training_frames": sum(len(corpus["x"]) for corpus in corpora),
        "corpora": [corpus["report"] for corpus in corpora],
        "corpus_sampling": "Equal probability per corpus per update; uniform training-frame sampling within corpus",
        "corpus_updates": corpus_updates,
        "updates": updates,
        "sampled_training_examples": updates * args.batch_size,
        "trained_existing_parameters": weights.numel() + bias.numel(),
        "new_deployed_parameters": 0,
        "wing_output_indices": wings.tolist(),
        "all_other_actor_parameters_unchanged": True,
        "non_wing_output_rows_unchanged": True,
        "live_physics_worlds": 0,
        "initial_validation_wing_mse": float(np.mean(initial_by_corpus)),
        "final_validation_wing_mse": float(np.mean(final_by_corpus)),
        "initial_validation_by_corpus": initial_by_corpus,
        "final_validation_by_corpus": final_by_corpus,
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
    parser.add_argument("--additional-cache", type=Path, action="append", default=[])
    parser.add_argument("--additional-cache-report", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=51001)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())
