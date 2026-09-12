"""Anatomical tracking diagnostics on saved states; no simulated step or policy.

Reports raw and 100 ms block-mean errors without changing any acceptance gate.
Replaying teachers and students identically distinguishes gait oscillation from
command drift. It does not reconstruct per-substep support-force acceptance.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from embodied_fly.body import CONTROL_DT
from embodied_fly.provenance import evidence, sha256, utc_now


def tracking(source, output, block_seconds=0.1):
    if output.exists():
        raise FileExistsError("Preserve previous diagnostics; choose a new output")
    started = time.perf_counter()
    provenance = evidence()
    model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
    data = mujoco.MjData(model)
    thorax = model.body("walker/thorax").id
    teacher = (source / "manifest.json").exists()
    manifest = json.loads(
        (source / ("manifest.json" if teacher else "report.json")).read_text()
    )
    cases = manifest["episodes"] if teacher else manifest["results"]
    block = max(1, round(block_seconds / CONTROL_DT))
    results = []
    for case in cases:
        name = f"episode_{case['episode']:03d}" if teacher else case["case"]
        command = (
            [case["speed_cm_s"], case["yaw_rad_s"]] if teacher else case["command_cm_s_rad_s"]
        )
        with np.load(source / f"{name}.npz", allow_pickle=False) as states:
            velocities = []
            for qpos, qvel in zip(states["qpos"], states["qvel"]):
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(model, data)
                velocity = np.empty(6)
                mujoco.mj_objectVelocity(
                    model, data, mujoco.mjtObj.mjOBJ_XBODY, thorax, velocity, 1
                )
                velocities.append(velocity[[3, 2]])
            velocity = np.asarray(velocities)
            error = velocity - command
            usable = len(error) // block * block
            averaged = error[:usable].reshape(-1, block, 2).mean(1)
            result = {
                "case": name,
                "command_cm_s_rad_s": command,
                "frames": len(velocity),
                "raw_rmse_cm_s_rad_s": np.sqrt(np.square(error).mean(0)).tolist(),
                "mean_velocity_cm_s_rad_s": velocity.mean(0).tolist(),
                "block_mean_rmse_cm_s_rad_s": np.sqrt(np.square(averaged).mean(0)).tolist()
                if usable
                else None,
                "frames_in_block_metric": usable,
                "state_sha256": sha256(source / f"{name}.npz"),
            }
        results.append(result)
        print(json.dumps(result), flush=True)
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "source_manifest_sha256": sha256(
            source / ("manifest.json" if teacher else "report.json")
        ),
        "model_sha256": sha256(source / "model.mjb"),
        "frame": "anatomical thorax; mjOBJ_XBODY",
        "block_seconds": block * CONTROL_DT,
        "metric_window": "whole available capture; only final incomplete block omitted from block metric",
        "acceptance_gates_changed": False,
        "wall_seconds": time.perf_counter() - started,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--block-seconds", type=float, default=0.1)
    args = parser.parse_args()
    tracking(args.source, args.output, args.block_seconds)
