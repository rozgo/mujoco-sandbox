"""Fit current-state wing targets through the existing frozen graph features.

This is offline readout learning from canonical recorded histories. Physical
evaluation is mandatory: a good fit on teacher histories cannot prove hover.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.wing_readout import replace_wing_rows


def fit_delta(features, residual, weights, regularization):
    """Regularized least squares in original hidden coordinates, with intercept."""
    x = np.column_stack((np.asarray(features, np.float64), np.ones(len(features))))
    y = np.asarray(residual, np.float64)
    w = np.asarray(weights, np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or len(y) != len(x)
        or w.shape != (len(x),)
        or not all(np.isfinite(v).all() for v in (x, y, w))
        or np.any(w <= 0)
        or not np.isfinite(regularization)
        or regularization <= 0
    ):
        raise ValueError("Invalid finite weighted regression data")
    gram = x.T @ (w[:, None] * x) / w.sum()
    rhs = x.T @ (w[:, None] * y) / w.sum()
    return np.linalg.solve(gram + regularization * np.eye(x.shape[1]), rhs)


def read_capture(root, case, expected_contract):
    report = json.loads((root / "report.json").read_text())
    if report["physical_contract"] != expected_contract or report["control_hz"] != 500:
        raise ValueError("Capture physical body or action clock differs")
    if sha256(root / "model.mjb") != report["model_sha256"]:
        raise ValueError("Capture compiled-model hash differs")
    path = root / (case + ".npz")
    entry = next(c for c in report["results"] if c["case"] == case)
    if sha256(path) != entry["state_sha256"]:
        raise ValueError("Capture hash differs")
    with np.load(path) as data:
        observation, action = data["observation"].copy(), data["action"].copy()
        if observation.shape != (2500, 397) or action.shape != (2500, 78):
            raise ValueError("Five-second canonical full-motor capture required")
        if not np.isfinite(observation).all() or not np.isfinite(action).all():
            raise ValueError("Nonfinite captured history")
        np.testing.assert_array_equal(observation[1:, 297:375], action[:-1])
    return (
        observation,
        action,
        {"case": case, "sha256": sha256(path), "report_sha256": sha256(root / "report.json")},
    )


@torch.no_grad()
def train(args):
    args.output.mkdir(parents=True, exist_ok=False)
    started, started_utc = time.perf_counter(), utc_now()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.resume, args.graph, device)
    if not actor.motor_only or actor.observation_size != 397:
        raise ValueError("Canonical motor-only actor required")
    actor.requires_grad_(False)
    ground_report = json.loads((args.ground / "report.json").read_text())
    hover_report = json.loads((args.hover / "report.json").read_text())
    if ground_report["checkpoint_sha256"] != sha256(args.resume):
        raise ValueError("Ground retention data must be from the exact parent")
    if not hover_report["teacher_present"] or hover_report["hover_reference"] != "state":
        raise ValueError("Explicit measured-state hover reference required")
    episodes = [
        read_capture(root, case, parent["physical_contract"])
        for root, case in (
            (args.ground, "stand"),
            (args.ground, "walk"),
            (args.hover, "hover"),
        )
    ]
    observations = np.stack([e[0] for e in episodes], axis=1)
    targets = np.stack([e[1] for e in episodes], axis=1)
    wings = np.arange(14, 20)
    frozen_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    memory = actor.initial_state(3)
    captured = {}

    def capture(module, inputs, output):
        captured["hidden"] = inputs[0].cpu().numpy().copy()
        captured["logits"] = output.cpu().numpy().copy()

    hook = actor.motor_decoder[3].register_forward_hook(capture)
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    replay_start = time.perf_counter()
    hidden, logits, labels = [], [], []
    try:
        for frame, obs in enumerate(observations):
            prediction = actor(torch.as_tensor(obs, device=device), memory)
            memory = prediction.state
            label = prediction.action.cpu().numpy().copy()
            label[2, wings] = targets[frame, 2, wings]
            hidden.append(captured["hidden"])
            logits.append(captured["logits"][:, wings])
            labels.append(label[:, wings])
    finally:
        hook.remove()
    synchronize(device)
    replay_seconds = time.perf_counter() - replay_start
    fit_start = time.perf_counter()
    hidden, logits, labels = map(np.asarray, (hidden, logits, labels))
    # Whole contiguous time windows: validation is descriptive and temporally
    # related, not an independent episode or a physical acceptance experiment.
    cutoff = 2000
    residual = np.arctanh(np.clip(labels, -0.999, 0.999)) - logits
    weight = np.tile([8.0, 8.0, 1.0], cutoff)
    x, y = hidden[:cutoff].reshape(-1, 256), residual[:cutoff].reshape(-1, 6)
    candidates = []
    deltas = []
    for regularization in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        delta = fit_delta(x, y, weight, regularization)
        prediction = np.tanh(logits + hidden @ delta[:-1] + delta[-1])
        mse = np.square(prediction - labels).mean(axis=-1)
        train_mse, valid_mse = mse[:cutoff].mean(0), mse[cutoff:].mean(0)
        candidates.append(
            {
                "regularization": regularization,
                "train_action_mse_by_task": train_mse.tolist(),
                "validation_action_mse_by_task": valid_mse.tolist(),
                "validation_selection_score": float(np.dot(valid_mse, [8, 8, 1]) / 17),
                "delta_l2": float(np.linalg.norm(delta)),
            }
        )
        deltas.append(delta)
    selected = min(
        range(len(candidates)), key=lambda i: candidates[i]["validation_selection_score"]
    )
    delta = deltas[selected]
    weight_new = frozen_state["motor_decoder.3.weight"][wings] + torch.as_tensor(
        delta[:-1].T, dtype=torch.float32
    )
    bias_new = frozen_state["motor_decoder.3.bias"][wings] + torch.as_tensor(
        delta[-1], dtype=torch.float32
    )
    state = replace_wing_rows(frozen_state, weight_new, bias_new, wings)
    other = np.r_[0:14, 20:78]
    for key, value in frozen_state.items():
        if key in ("motor_decoder.3.weight", "motor_decoder.3.bias"):
            assert torch.equal(value[other], state[key][other])
        else:
            assert torch.equal(value, state[key])
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
        source_commit=provenance["source_commit"],
        parent_checkpoint_sha256=sha256(args.resume),
        method="canonical state-hover ridge readout with frozen ground outputs",
    )
    torch.save(child, args.output / "actor.pt")
    np.savez_compressed(
        args.output / "features.npz", hidden=hidden, logits=logits, target=labels, delta=delta
    )
    report = {
        "provenance": provenance,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "method": child["method"],
        "parent_checkpoint_sha256": sha256(args.resume),
        "checkpoint_sha256": sha256(args.output / "actor.pt"),
        "physical_contract": parent["physical_contract"],
        "sources": [e[2] for e in episodes],
        "features_sha256": sha256(args.output / "features.npz"),
        "candidates": candidates,
        "selected": candidates[selected],
        "initial_action_mse_by_task": np.square(np.tanh(logits) - labels)
        .mean(axis=(0, 2))
        .tolist(),
        "training_frames": 6000,
        "validation_frames": 1500,
        "parallel_neural_histories": 3,
        "validation_split": "first four seconds fit, final second validation in each of three recorded histories; temporally related",
        "searched_ridge_values": 5,
        "eligible_existing_parameters": 1542,
        "all_other_state_unchanged": True,
        "physical_transitions_collected": 0,
        "live_worlds": 0,
        "runtime_teacher": False,
        "setup_seconds": setup_seconds,
        "frozen_replay_seconds": replay_seconds,
        "fit_and_save_seconds": time.perf_counter() - fit_start,
        "total_wall_seconds": time.perf_counter() - started,
        "device": str(device),
        "physical_success": "Unproven; requires unassisted three-task evaluation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("resume", "graph", "ground", "hover", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    train(parser.parse_args())
