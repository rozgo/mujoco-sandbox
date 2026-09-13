"""Offline readout diagnostic; fitted regressions never control the body.

Replay causal expert observations through a frozen checkpoint. Ask whether
current wing actions can be reconstructed from measured wing state or actual
motor-cell state on excluded whole episodes. This measures linear decodability
under expert histories, not autonomous flight or biological neural causality.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.body import CONTROL_DT
from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import load_episodes, synchronize


def ridge_readout(train_x, train_y, test_x, alpha=0.001):
    """All centering, scaling and fitting use training rows only."""
    train_x, train_y, test_x = [np.asarray(v, np.float64) for v in (train_x, train_y, test_x)]
    mean, scale = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    target_mean = train_y.mean(0)
    x = (train_x - mean) / scale
    covariance = x.T @ x / len(x) + alpha * np.eye(x.shape[1])
    coefficients = np.linalg.solve(covariance, x.T @ (train_y - target_mean) / len(x))
    return (test_x - mean) / scale @ coefficients + target_mean


def metrics(target, prediction, baseline):
    mse = np.mean((prediction - target) ** 2, axis=0)
    denominator = np.mean((target - baseline) ** 2, axis=0)
    return {
        "mse": float(mse.mean()),
        "per_wing_axis_mse": mse.tolist(),
        "r2_against_training_mean": (1 - mse / np.maximum(denominator, 1e-12)).tolist(),
    }


@torch.no_grad()
def diagnose(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    provenance = evidence()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, device)
    if actor.sensor_extension_size != 12:
        raise ValueError("This diagnostic requires the declared 395-input schema")
    manifest = json.loads((args.data / "manifest.json").read_text())
    if manifest["control_hz"] != 5000:
        raise ValueError("Expected the physical 5 kHz airborne capture")
    episodes = load_episodes(args.data, True, True)
    length = min(len(e["action"]) for e in episodes)
    validation = np.random.default_rng(1193).permutation(len(episodes))[
        : max(1, len(episodes) // 4)
    ]
    training = np.array([i for i in range(len(episodes)) if i not in validation])
    observations = np.stack([e["observation"][:length] for e in episodes], axis=1)
    model = mujoco.MjModel.from_binary_path(str(args.data / "model.mjb"))
    wings = np.array([i for i in range(model.nu) if "wing_" in model.actuator(i).name])
    target = np.stack([e["action"][:length, wings] for e in episodes], axis=1)
    state = actor.initial_state(len(episodes))
    motor, predicted = [], []
    synchronize(device)
    setup_seconds = time.perf_counter() - started
    replay_start = time.perf_counter()
    for observation in observations:
        result = actor(
            torch.as_tensor(observation, device=device),
            state,
            time_scale=1 / 5000 / CONTROL_DT,
        )
        state = result.state
        motor.append(state[actor.motor_ids].T.cpu().numpy().copy())
        predicted.append(result.action[:, wings].cpu().numpy())
    synchronize(device)
    replay_seconds = time.perf_counter() - replay_start
    features = {
        "measured_wing_angles_and_velocities": observations[:, :, 383:395],
        "motor_cell_state": np.asarray(motor),
    }
    test_target = target[:, validation].reshape(-1, 6)
    train_target = target[:, training].reshape(-1, 6)
    baseline = train_target.mean(0)
    readouts = {}
    fit_start = time.perf_counter()
    for name, values in features.items():
        train_x = values[:, training].reshape(-1, values.shape[-1])
        test_x = values[:, validation].reshape(-1, values.shape[-1])
        estimate = ridge_readout(train_x, train_target, test_x)
        readouts[name] = {
            "features": values.shape[-1],
            "training_feature_std_mean": float(train_x.std(0).mean()),
            "unbounded_regression": metrics(test_target, estimate, baseline),
            "bounded_action_prediction": metrics(test_target, estimate.clip(-1, 1), baseline),
        }
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "graph_sha256": checkpoint["graph_sha256"],
        "data_manifest_sha256": sha256(args.data / "manifest.json"),
        "scope": "Offline linear readout probes of frozen actor under expert histories; never deployed",
        "limitations": "Expert previous actions are causal inputs. Linear decodability does not prove autonomous oscillation, closed-loop stability or causal necessity.",
        "new_deployed_parameters": 0,
        "physical_steps": 0,
        "neural_device": str(device),
        "parallel_neural_sequences": len(episodes),
        "frames_per_sequence": length,
        "training_episode_indices": training.tolist(),
        "validation_episode_indices": sorted(validation.tolist()),
        "ridge_alpha": 0.001,
        "wing_channel_names": [model.actuator(i).name for i in wings],
        "mean_predictor": metrics(test_target, baseline, baseline),
        "current_motor_decoder": metrics(
            test_target, np.asarray(predicted)[:, validation].reshape(-1, 6), baseline
        ),
        "readouts": readouts,
        "setup_seconds": setup_seconds,
        "frozen_actor_replay_seconds": replay_seconds,
        "regression_and_metrics_seconds": time.perf_counter() - fit_start,
        "total_wall_seconds": time.perf_counter() - started,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    diagnose(parser.parse_args())
