"""Free-body aerodynamic diagnostic; upstream approximate wing pattern, no brain.

Not learned flight, takeoff or acceptance. Start airborne once, drive only six
bounded wing actuators, and compare air on/off and a halved physics timestep.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
from flybody.tasks.pattern_generators import WingBeatPatternGenerator

from embodied_fly.body import FlyEnvironment
from embodied_fly.provenance import evidence, sha256, utc_now


def probe(output, seconds=0.03, wing_pattern=None):
    if seconds <= 0 or not np.isfinite(seconds):
        raise ValueError("Positive finite probe duration required")
    output.mkdir(parents=True, exist_ok=False)
    provenance = evidence()
    results = []
    for name, physics_dt, air in (
        ("air_20khz", 0.00005, True),
        ("vacuum_20khz", 0.00005, False),
        ("air_40khz", 0.000025, True),
    ):
        setup_started = time.perf_counter()
        env = FlyEnvironment("flight")
        model, data = env.model, env.data
        model.opt.timestep = physics_dt
        env.substeps = round(env.control_dt / physics_dt)
        if not air:
            model.opt.density = model.opt.viscosity = 0
        wing_names = [
            f"walker/wing_{axis}_{side}"
            for side in ("left", "right")
            for axis in ("yaw", "roll", "pitch")
        ]
        joints = [model.joint(n).id for n in wing_names]
        actuators = [model.actuator(n).id for n in wing_names]
        addresses = model.jnt_qposadr[joints]
        pattern = WingBeatPatternGenerator(
            base_pattern_path=wing_pattern, dt_ctrl=env.control_dt
        )
        q, v = pattern.reset(initial_phase=0, return_qvel=True)
        # Initialization only; the complete body remains free afterward.
        data.qpos[2] = 1.0  # cm; enough for a short airborne diagnostic.
        hover_up = model.site_quat[model.site("walker/hover_up_dir").id].copy()
        hover_up[1:] *= -1
        data.qpos[3:7] = hover_up / np.linalg.norm(hover_up)
        data.qpos[addresses] = q
        data.qvel[model.jnt_dofadr[joints]] = v
        mujoco.mj_forward(model, data)
        origin = data.qpos[:3].copy()
        mujoco.mj_saveModel(model, str(output / f"{name}.mjb"))
        setup_seconds = time.perf_counter() - setup_started
        started = time.perf_counter()
        rows = {
            k: []
            for k in (
                "qpos",
                "qvel",
                "activation",
                "ctrl",
                "wing_force",
                "root_passive_force",
                "time",
            )
        }
        error = None
        try:
            for _ in range(round(seconds / env.control_dt)):
                raw = np.zeros(model.nu)
                # Same pattern-to-torque proportional error as upstream teacher,
                # with no learned residual. Its synthetic pattern is a probe only.
                raw[actuators] = pattern.step(pattern.base_beat_freq) - data.qpos[addresses]
                action = 2 * (raw - env.low) / (env.high - env.low) - 1
                env.step(action)
                for key, value in (
                    ("qpos", data.qpos),
                    ("qvel", data.qvel),
                    ("activation", data.act),
                    ("ctrl", data.ctrl),
                    ("wing_force", data.actuator_force[actuators]),
                    ("root_passive_force", data.qfrc_passive[:3]),
                    ("time", data.time),
                ):
                    rows[key].append(np.array(value, copy=True))
                assert not data.xfrc_applied.any() and not data.qfrc_applied.any()
        except RuntimeError as caught:
            error = str(caught)
        arrays = {k: np.asarray(v) for k, v in rows.items()}
        np.savez_compressed(output / f"{name}.npz", **arrays)
        weight = model.body_mass.sum() * abs(model.opt.gravity[2])
        force = arrays["wing_force"]
        limit = np.maximum(
            np.abs(model.actuator_forcerange[actuators, 0]),
            np.abs(model.actuator_forcerange[actuators, 1]),
        )
        report = {
            "name": name,
            "air_enabled": air,
            "physics_hz": 1 / physics_dt,
            "control_hz": 1 / env.control_dt,
            "body_mass_kg": model.body_mass.sum() * 0.001,
            "setup_seconds": setup_seconds,
            "stepping_capture_and_state_write_wall_seconds": time.perf_counter() - started,
            "simulated_seconds": data.time,
            "displacement_m": ((data.qpos[:3] - origin) * 0.01).tolist(),
            "final_root_velocity_cm_s": data.qvel[:3].tolist(),
            "mean_upward_passive_force_over_weight": float(
                arrays["root_passive_force"][:, 2].mean() / weight
            ),
            "max_wing_actuator_force_over_limit": float(np.max(np.abs(force) / limit)),
            "wing_saturation_fraction": float(np.mean(np.abs(force) >= 0.999 * limit)),
            "warning_count": int(data.warning.number.sum()),
            "numerical_failure": error,
            "final_ground_contacts": sum(
                int(model.geom_bodyid[c.geom1] == 0 or model.geom_bodyid[c.geom2] == 0)
                for c in data.contact
            ),
            "model_sha256": sha256(output / f"{name}.mjb"),
            "state_sha256": sha256(output / f"{name}.npz"),
        }
        results.append(report)
        print(json.dumps(report), flush=True)
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "controller": "upstream WingBeatPatternGenerator plus bounded proportional wing torque",
        "wing_pattern": "supplied dataset" if wing_pattern else "synthetic approximation",
        "wing_pattern_sha256": sha256(wing_pattern) if wing_pattern else None,
        "learned_controller_present": False,
        "physical_preset": "flight, complete body and floor contact",
        "acceptance_eligible": False,
        "pose_writes": "initialization only",
        "external_root_forces": False,
        "scope": "short airborne force and timestep diagnostic; not demonstrated stable flight or takeoff",
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=0.03)
    parser.add_argument("--wing-pattern", type=Path)
    args = parser.parse_args()
    probe(args.output, args.seconds, args.wing_pattern)
