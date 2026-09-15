"""Separate early model prediction error from frozen-neural-history transfer."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.full_body_decoder import FullBodyDecoder
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.world_data import HISTORY
from embodied_fly.world_residual_model import ResidualWorldModel
from embodied_fly.world_residual_physics import DifferentiableFly


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device)
    model = mujoco.MjModel.from_binary_path(str(args.model))
    plant = DifferentiableFly(model).to(device=device, dtype=torch.float32)
    checkpoint = torch.load(args.world_model, map_location=device, weights_only=False)
    net = ResidualWorldModel(**checkpoint["config"]).to(device)
    net.load_state_dict(checkpoint["state_dict"])
    net.eval()
    with np.load(args.experience / "episode_00.npz") as z:
        motor = torch.as_tensor(z["motor"][:100], device=device)[None]
        raw = np.repeat(z["features"][0][None, None], HISTORY, axis=1)
        present = np.zeros((1, HISTORY, 1), np.float32)
        present[:, -1] = 1
        history = torch.as_tensor(np.concatenate((raw, present), -1), device=device)
        initial = torch.as_tensor(z["metrics"][0], device=device)[None]
        qpos = z["qpos"][0:1]
    inertia, inverse = plant.inertia(qpos, initial)
    rows = []
    for label, folder in (("learned_residual", args.learned), ("analytical", args.analytical)):
        saved = torch.load(folder / "actor_0100.pt", map_location=device, weights_only=False)
        decoder = FullBodyDecoder(815, 384, 78).to(device)
        decoder.load_state_dict(
            {
                k.removeprefix("motor_decoder."): v
                for k, v in saved["state_dict"].items()
                if k.startswith("motor_decoder.")
            }
        )
        cached = decoder(motor)
        with np.load(args.evaluation / f"{label}_0100/episode_00.npz") as z:
            live = torch.as_tensor(z["action"][:100], device=device)[None]
            actual = z["qvel"][100, :3] * 10
        predictions = {}
        for name, action in (("cached_neural_history", cached), ("actual_live_actions", live)):
            residual = (
                net.accelerations(net.latent_rollout(history, action))
                if label == "learned_residual"
                else torch.zeros(1, 100, 6, device=device)
            )
            path, _ = plant(
                initial, plant.wing_rollout(initial, action), inertia, inverse, residual
            )
            predictions[name] = (path[0, -1, 3:6] * 10).cpu().tolist()
        rows.append(
            {
                "method": label,
                "episode": 0,
                "horizon_seconds": 0.2,
                "maximum_action_sequence_difference": float((cached - live).abs().max()),
                "action_sequence_rms_difference": float(
                    (cached - live).square().mean().sqrt()
                ),
                "actual_end_velocity_mm_s": actual.tolist(),
                "predicted_end_velocity_mm_s": predictions,
                "model_error_given_live_commands_mm_s": float(
                    np.linalg.norm(np.array(predictions["actual_live_actions"]) - actual)
                ),
            }
        )
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "rows": rows,
        "scope": "Cold episode 0; same initial physical state; measured live commands versus decoder commands on frozen parent motor histories",
        "world_model_sha256": sha256(args.world_model),
        "optimizer_updates": 0,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "world-model",
        "experience",
        "learned",
        "analytical",
        "evaluation",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--device", default="cpu")
    run(p.parse_args())
