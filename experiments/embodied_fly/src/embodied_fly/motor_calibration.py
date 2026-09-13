"""Fit the existing nonlinear motor decoder with explicit ground retention.

The graph, cell dynamics, sensory and utility interfaces stay exactly unchanged.
Cached motor-cell states are the only inputs to this existing deployed module.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize


def decoder_from_state(state):
    weights = state["motor_decoder.1.weight"]
    output = state["motor_decoder.3.weight"].shape[0]
    hidden, inputs = weights.shape
    decoder = nn.Sequential(
        nn.LayerNorm(inputs),
        nn.Linear(inputs, hidden),
        nn.Tanh(),
        nn.Linear(hidden, output),
        nn.Tanh(),
    )
    decoder.load_state_dict(
        {
            k.removeprefix("motor_decoder."): v
            for k, v in state.items()
            if k.startswith("motor_decoder.")
        },
        strict=True,
    )
    return decoder


def verify_feature_parent(resume, reference):
    """A changed decoder can reuse features only when its full upstream map matches."""
    for key in (
        "observation_size",
        "sensor_extension_size",
        "action_size",
        "graph_sha256",
        "graph_metadata_sha256",
    ):
        if resume[key] != reference[key]:
            raise ValueError("Feature parent architecture or graph differs")
    if resume["config"]["internal_steps"] != reference["config"]["internal_steps"]:
        raise ValueError("Feature parent neural clock differs")
    a, b = resume["state_dict"], reference["state_dict"]
    if a.keys() != b.keys():
        raise ValueError("Feature parent state schema differs")
    for key, value in a.items():
        if value.shape != b[key].shape or (
            not key.startswith("motor_decoder.") and not torch.equal(value, b[key])
        ):
            raise ValueError(f"Feature parent upstream state differs: {key}")


def load_corpus(path, parent_hash, state, device):
    report = json.loads((path / "report.json").read_text())
    if (
        report["schema"] != "frozen-motor-cell-features-v1"
        or report["checkpoint_sha256"] != parent_hash
        or report["cache_sha256"] != sha256(path / "features.npz")
    ):
        raise ValueError("Feature corpus schema, checkpoint or cache mismatch")
    if report["role"] not in ("flight", "ground"):
        raise ValueError("Unknown corpus role")
    with np.load(path / "features.npz", allow_pickle=False) as c:
        count = len(c["episode"])
        inputs = state["motor_decoder.1.weight"].shape[1]
        outputs = state["motor_decoder.3.weight"].shape[0]
        if (
            c["motor"].shape != (count, inputs)
            or c["parent_action"].shape != (count, outputs)
            or c["teacher_action"].shape != (count, outputs)
            or c["frame"].shape != (count,)
        ):
            raise ValueError("Invalid feature shapes")
        if not all(np.isfinite(c[k]).all() for k in c.files):
            raise ValueError("Nonfinite cached features")
        training, validation = c["training_episodes"], c["validation_episodes"]
        for ids in (training, validation):
            if ids.ndim != 1 or not len(ids) or len(ids) != len(np.unique(ids)):
                raise ValueError("Invalid episode split")
        if set(training) & set(validation):
            raise ValueError("Training/validation episode overlap")
        if set(c["episode"]) != set(training) | set(validation):
            raise ValueError("Episode split does not cover the cache")
        wings = c["wing_channels"].copy()
        if (
            wings.shape != (6,)
            or len(set(wings)) != 6
            or np.any((wings < 0) | (wings >= outputs))
        ):
            raise ValueError("Invalid wing routing")
        packed = {}
        for split, ids in (("train", training), ("valid", validation)):
            mask = np.isin(c["episode"], ids)
            if not mask.any():
                raise ValueError("Empty corpus split")
            packed[split] = {
                k: torch.as_tensor(c[key][mask], device=device)
                for k, key in (
                    ("x", "motor"),
                    ("parent", "parent_action"),
                    ("teacher", "teacher_action"),
                    ("frame", "frame"),
                )
            }
        packed.update(
            role=report["role"],
            wings=wings,
            identity={
                "role": report["role"],
                "report_sha256": sha256(path / "report.json"),
                "cache_sha256": report["cache_sha256"],
                "training_episodes": training.tolist(),
                "validation_episodes": validation.tolist(),
                "training_frames": len(packed["train"]["x"]),
                "validation_frames": len(packed["valid"]["x"]),
            },
        )
    return packed


def train(args):
    if (
        not np.isfinite(
            [
                args.seconds,
                args.lr,
                args.retention_weight,
                args.startup_fraction,
                args.ground_fraction,
            ]
        ).all()
        or min(args.seconds, args.lr, args.retention_weight, args.batch_size) <= 0
        or not 0 <= args.startup_fraction <= 1
        or not 0 < args.ground_fraction < 1
        or args.startup_frames < 0
        or (args.startup_fraction and not args.startup_frames)
    ):
        raise ValueError("Invalid bounded motor calibration configuration")
    args.output.mkdir(parents=True, exist_ok=False)
    setup = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    parent_hash = sha256(args.resume)
    parent = torch.load(args.resume, map_location="cpu", weights_only=True)
    original = parent["state_dict"]
    feature_hash = parent_hash
    if args.feature_parent is not None:
        reference = torch.load(args.feature_parent, map_location="cpu", weights_only=True)
        verify_feature_parent(parent, reference)
        feature_hash = sha256(args.feature_parent)
    corpora = [load_corpus(path, feature_hash, original, device) for path in args.cache]
    ground = [i for i, c in enumerate(corpora) if c["role"] == "ground"]
    flight = [i for i, c in enumerate(corpora) if c["role"] == "flight"]
    if not ground or not flight:
        raise ValueError("Both flight supervision and ground retention are required")
    wings = corpora[0]["wings"]
    if any(not np.array_equal(c["wings"], wings) for c in corpora):
        raise ValueError("Different wing routing between corpora")
    other = np.array([i for i in range(parent["action_size"]) if i not in wings])
    axis_scale = (
        torch.cat([corpora[i]["train"]["teacher"][:, wings] for i in flight])
        .std(0, correction=0)
        .clamp_min(0.1)
    )
    for c in corpora:
        c["startup"] = torch.nonzero(c["train"]["frame"] < args.startup_frames).flatten()
        if args.startup_fraction and c["role"] == "flight" and not len(c["startup"]):
            raise ValueError("No training startup frames")
    decoder = decoder_from_state(original).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    @torch.no_grad()
    def validate():
        result = []
        for c in corpora:
            d = c["valid"]
            prediction = decoder(d["x"])
            startup = d["frame"] < args.startup_frames
            result.append(
                {
                    "role": c["role"],
                    "parent_all_output_mse": float((prediction - d["parent"]).square().mean()),
                    "non_wing_parent_mse": float(
                        (prediction[:, other] - d["parent"][:, other]).square().mean()
                    ),
                    "wing_label_mse": float(
                        (prediction[:, wings] - d["teacher"][:, wings]).square().mean()
                    ),
                    "startup_wing_label_mse": float(
                        (prediction[startup][:, wings] - d["teacher"][startup][:, wings])
                        .square()
                        .mean()
                    )
                    if startup.any()
                    else None,
                }
            )
        return result

    initial = validate()
    synchronize(device)
    setup_seconds = time.perf_counter() - setup
    started_utc, started = utc_now(), time.perf_counter()
    updates, counts = 0, [0] * len(corpora)
    while time.perf_counter() - started < args.seconds:
        choices = (
            ground if float(torch.rand((), device=device)) < args.ground_fraction else flight
        )
        index = choices[int(torch.randint(len(choices), (), device=device))]
        c = corpora[index]
        d = c["train"]
        ids = torch.randint(len(d["x"]), (args.batch_size,), device=device)
        if c["role"] == "flight" and args.startup_fraction:
            n = round(args.batch_size * args.startup_fraction)
            ids[:n] = c["startup"][torch.randint(len(c["startup"]), (n,), device=device)]
        prediction = decoder(d["x"][ids])
        if c["role"] == "ground":
            loss = args.retention_weight * (prediction - d["parent"][ids]).square().mean()
        else:
            loss = (
                ((prediction[:, wings] - d["teacher"][ids][:, wings]) / axis_scale)
                .square()
                .mean()
            )
            loss = (
                loss
                + args.retention_weight
                * (prediction[:, other] - d["parent"][ids][:, other]).square().mean()
            )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite motor calibration loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
        optimizer.step()
        synchronize(device)
        updates += 1
        counts[index] += 1
    training_seconds = time.perf_counter() - started
    saving = time.perf_counter()
    final = validate()
    state = {k: v.clone() for k, v in original.items()}
    for k, v in decoder.state_dict().items():
        state["motor_decoder." + k] = v.detach().cpu()
    for k, v in original.items():
        if not k.startswith("motor_decoder.") and not torch.equal(v, state[k]):
            raise RuntimeError("Change outside existing motor decoder")
    checkpoint = {
        k: parent[k]
        for k in (
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
        method="frozen-core nonlinear motor calibration",
        parent_checkpoint_sha256=parent_hash,
    )
    torch.save(checkpoint, args.output / "actor.pt")
    report = {
        "provenance": provenance,
        "training_started_utc": started_utc,
        "completed_utc": utc_now(),
        "method": checkpoint["method"],
        "parent_checkpoint_sha256": parent_hash,
        "feature_parent_checkpoint_sha256": feature_hash,
        "feature_parent_upstream_map_verified_identical": True,
        "retention_target": "Outputs of the feature-parent actor on recorded histories",
        "optimizer_initialization": "Fresh Adam for this declared motor calibration",
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "corpora": [c["identity"] for c in corpora],
        "seed": args.seed,
        "learning_rate": args.lr,
        "retention_weight": args.retention_weight,
        "ground_sample_probability": args.ground_fraction,
        "startup_frames": args.startup_frames,
        "startup_sample_fraction_within_flight": args.startup_fraction,
        "axis_loss_scale_from_training_only": axis_scale.tolist(),
        "batch_size": args.batch_size,
        "updates": updates,
        "updates_by_corpus": counts,
        "unique_training_frames": sum(len(c["train"]["x"]) for c in corpora),
        "sampled_training_examples": updates * args.batch_size,
        "trained_existing_parameters": sum(p.numel() for p in decoder.parameters()),
        "new_deployed_parameters": 0,
        "all_upstream_actor_parameters_unchanged": True,
        "non_wing_outputs_can_change": True,
        "live_physics_worlds": 0,
        "initial_validation": initial,
        "final_validation": final,
        "setup_seconds": setup_seconds,
        "training_seconds": training_seconds,
        "evaluation_and_save_seconds": time.perf_counter() - saving,
        "neural_device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "physical_success": "Not established; all ground and flight gates require live evaluation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("resume", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cache", type=Path, action="append", required=True)
    parser.add_argument("--feature-parent", type=Path)
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--retention-weight", type=float, default=4)
    parser.add_argument("--ground-fraction", type=float, default=0.5)
    parser.add_argument("--startup-frames", type=int, default=25)
    parser.add_argument("--startup-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=58001)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())
