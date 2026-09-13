"""Fit current-state wing targets through the existing frozen graph features.

This is offline readout learning from canonical recorded histories. Physical
evaluation is mandatory: a good fit on teacher histories cannot prove hover.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.state_hover import wing_commands
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


def correction_capture(root, expected_contract):
    """Relabel an actor's actual pre-fall history; never modify or integrate it."""
    observation, action, identity = read_capture(root, "hover", expected_contract)
    report = json.loads((root / "report.json").read_text())
    if report["teacher_present"] or not report["student_present"]:
        raise ValueError("Corrections require unassisted actor history")
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    if physical_contract(model) != expected_contract:
        raise ValueError("Correction compiled mechanics differ")
    data = mujoco.MjData(model)
    thorax = model.body("walker/thorax").id
    joints = [
        model.joint(f"walker/wing_{axis}_{side}").id
        for side in ("left", "right")
        for axis in ("yaw", "roll", "pitch")
    ]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    wings = [model.actuator(model.joint(j).name).id for j in joints]
    if wings != list(range(14, 20)):
        raise ValueError("Unexpected canonical wing actuator routing")
    labels = action.copy()
    retained = 0
    with np.load(root / "hover.npz") as captured:
        for i in range(min(1000, len(action))):
            data.qpos[:] = captured["qpos"][i]
            data.qvel[:] = captured["qvel"][i]
            data.act[:] = captured["activation"][i]
            data.ctrl[:] = captured["ctrl"][i]
            mujoco.mj_forward(model, data)
            rotation = data.xmat[thorax].reshape(3, 3)
            if data.qpos[2] < 0.5 or rotation[2, 2] < 0.5:
                break
            velocity = []
            for kind in ("angvel", "linvel"):
                adr = model.sensor(f"batch_xbody_{kind}").adr[0]
                velocity.extend(rotation.T @ data.sensordata[adr : adr + 3])
            labels[i, wings] = wing_commands(
                data.qpos[qa][None],
                data.qvel[va][None],
                np.asarray(velocity)[None],
                np.array([data.qpos[2]]),
                np.array([captured["requested_height_cm"][i]]),
                model.qpos_spring[qa],
            )[0]
            retained += 1
    if retained < 50:
        raise ValueError("Correction history needs at least 100ms before failure")
    identity.update(
        case="hover_correction",
        producing_checkpoint_sha256=report["checkpoint_sha256"],
        retained_pre_failure_frames=retained,
        excluded_frames=len(action) - retained,
        label_method="existing measured-state hover reference on actual actor states",
        physics_integration_steps=0,
    )
    return observation, labels, identity


def history_masks(length, counts):
    """Ground/reference final second; correction every fifth 20ms block held out."""
    train = np.zeros((length, len(counts)), bool)
    valid = np.zeros_like(train)
    for j, count in enumerate(counts):
        if not 50 <= count <= length:
            raise ValueError("Invalid retained history length")
        if j < 3:
            train[:2000, j] = True
            valid[2000:count, j] = True
        else:
            held = np.arange(count) // 10 % 5 == 4
            train[:count, j] = ~held
            valid[:count, j] = held
        if not train[:, j].any() or not valid[:, j].any():
            raise ValueError("Both fit and descriptive validation frames required")
    return train, valid


@torch.no_grad()
def train(args):
    startup_weight = getattr(args, "startup_weight", 1.0)
    if not np.isfinite(startup_weight) or startup_weight < 1:
        raise ValueError("Startup weight must be finite and at least one")
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
    corrections = getattr(args, "correction_capture", []) or []
    episodes.extend(
        correction_capture(path, parent["physical_contract"]) for path in corrections
    )
    retained = [2500] * 3 + [e[2]["retained_pre_failure_frames"] for e in episodes[3:]]
    train_mask, valid_mask = history_masks(2500, retained)
    history_weights = np.asarray([8.0, 8.0, 1.0] + [1.0] * len(corrections))
    frame_weights = np.tile(history_weights, (2500, 1))
    frame_weights[:50, 2:] *= startup_weight
    observations = np.stack([e[0] for e in episodes], axis=1)
    targets = np.stack([e[1] for e in episodes], axis=1)
    wings = np.arange(14, 20)
    frozen_state = {k: v.detach().cpu().clone() for k, v in actor.state_dict().items()}
    memory = actor.initial_state(len(episodes))
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
            label[2:, wings] = targets[frame, 2:][:, wings]
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
    residual = np.arctanh(np.clip(labels, -0.999, 0.999)) - logits
    weight = frame_weights[train_mask]
    x, y = hidden[train_mask], residual[train_mask]
    candidates = []
    deltas = []
    for regularization in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        delta = fit_delta(x, y, weight, regularization)
        prediction = np.tanh(logits + hidden @ delta[:-1] + delta[-1])
        mse = np.square(prediction - labels).mean(axis=-1)
        train_mse = np.array([mse[train_mask[:, j], j].mean() for j in range(len(episodes))])
        valid_mse = np.array([mse[valid_mask[:, j], j].mean() for j in range(len(episodes))])
        candidates.append(
            {
                "regularization": regularization,
                "train_action_mse_by_task": train_mse.tolist(),
                "validation_action_mse_by_task": valid_mse.tolist(),
                "validation_selection_score": float(
                    np.dot(valid_mse, history_weights) / history_weights.sum()
                ),
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
        method="canonical state-hover ridge readout with frozen ground outputs"
        + (" and actual actor-error corrections" if corrections else ""),
    )
    torch.save(child, args.output / "actor.pt")
    np.savez_compressed(
        args.output / "features.npz",
        hidden=hidden,
        logits=logits,
        target=labels,
        delta=delta,
        train_mask=train_mask,
        validation_mask=valid_mask,
        frame_weights=frame_weights,
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
        "initial_action_mse_by_task": [
            float(np.square(np.tanh(logits[:n, j]) - labels[:n, j]).mean())
            for j, n in enumerate(retained)
        ],
        "training_frames": int(train_mask.sum()),
        "validation_frames": int(valid_mask.sum()),
        "retained_frames_by_history": retained,
        "parallel_neural_histories": len(episodes),
        "neural_frames_including_excluded_history": len(episodes) * 2500,
        "startup_weight_first_100ms_air_histories": startup_weight,
        "history_weights": history_weights.tolist(),
        "validation_split": "ground/reference first4s fit, last1s validation; actor corrections every fifth20ms block held out; temporally related, not independent episodes",
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
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "physical_success": "Unproven; requires unassisted three-task evaluation",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("resume", "graph", "ground", "hover", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--correction-capture", type=Path, action="append", default=[])
    parser.add_argument("--startup-weight", type=float, default=1.0)
    train(parser.parse_args())
