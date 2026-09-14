"""Read-only audit: distinguish fitting mean wing poses from tracking wing motion."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from embodied_fly.evaluate import load_actor
from embodied_fly.provenance import sha256, utc_now
from embodied_fly.velocity_imitation import sample_windows
from embodied_fly.wing_position import wing_actuators


@torch.no_grad()
def main(args):
    started = time.perf_counter()
    torch.set_num_threads(4)
    actor, _ = load_actor(args.checkpoint, args.graph, torch.device("cuda"))
    model = mujoco.MjModel.from_binary_path(str(args.dataset / "model.mjb"))
    wings = wing_actuators(model)
    obs = np.load(args.dataset / "observations.npy", mmap_mode="r")
    target = np.load(args.dataset / "actions.npy", mmap_mode="r")
    worlds, indices, valid, _ = sample_windows(
        np.random.default_rng(121102), [8, 9], 64, 128, 64, obs.shape[1]
    )
    selected = torch.as_tensor(obs[worlds[None], indices], device="cuda")
    valid = torch.as_tensor(valid, device="cuda")
    state = actor.initial_state(64)
    predictions = []
    for step in range(len(selected)):
        result = actor(selected[step], state)
        state = torch.where(valid[step][None], result.state, state)
        if step >= 64:
            predictions.append(result.action[:, wings].cpu().numpy())
    predictions = np.stack(predictions).reshape(-1, 6)
    expected = target[worlds[None], indices[64:]][:, :, wings].reshape(-1, 6)
    # Numpy's advanced indexing can put the wing axis first. Explicitly retain it last.
    constant = np.asarray(target[:8])[:, :, wings].reshape(-1, 6).mean(0)
    channels = []
    for i, actuator in enumerate(wings):
        error = float(np.mean((predictions[:, i] - expected[:, i]) ** 2))
        baseline = float(np.mean((constant[i] - expected[:, i]) ** 2))
        channels.append(
            {
                "name": model.actuator(int(actuator)).name,
                "prediction_mse": error,
                "constant_training_mean_mse": baseline,
                "predicted_std": float(predictions[:, i].std()),
                "target_std": float(expected[:, i].std()),
                "correlation": float(
                    np.corrcoef(predictions[:, i], expected[:, i])[0, 1]
                ),
                "improvement_over_constant": 1 - error / baseline,
            }
        )
    motion = []
    qvel_ids = model.jnt_dofadr[model.actuator_trnid[wings, 0]]
    for name, folder in (("teacher", args.dataset), ("student", args.evaluation)):
        with np.load(folder / "episode_00.npz") as data:
            # Match the interval strictly before the student's first height failure.
            selected = data["time"] < 0.07
            speed = data["qvel"][selected][:, qvel_ids[[0, 3]]]
            motion.append(
                {
                    "controller": name,
                    "measured_window_seconds": 0.07,
                    "mean_absolute_sweep_speed_rad_s": np.abs(speed).mean(0).tolist(),
                    "peak_sweep_speed_rad_s": np.abs(speed).max(0).tolist(),
                }
            )
    report = {
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "audit_wall_seconds": time.perf_counter() - started,
        "channels": channels,
        "physical_motion": motion,
        "mean_wing_mse": float(np.mean((predictions - expected) ** 2)),
        "mean_constant_training_pose_mse": float(
            np.mean((constant[None] - expected) ** 2)
        ),
        "scope": "Frozen checkpoint; same held-out windows as training validation. Constant predictor is an analysis baseline, never deployed control. No training updates.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    main(parser.parse_args())
