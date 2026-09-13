"""Audit the existing hover teacher's clock dependence without stepping physics.

This probes instantaneous observation ambiguity. A recurrent actor also has
history, so this cannot establish that its teaching task is unlearnable.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.motor_focus import MotorTasks, MotorTeacher
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now


def audit(teacher_path, output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    provenance = evidence()
    env = FlyBatch(3, 3, 14, preset="wing_motion")
    tasks = MotorTasks(env, 82001)
    teacher = MotorTeacher(tasks, teacher_path, "cpu", ground_posture=True)
    contract = physical_contract(env.model)
    state = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    previous = env.previous_action.copy()
    ages = env.ages.copy()
    observation = env.observation()[2].copy()
    setup = time.perf_counter() - started
    started = time.perf_counter()
    samples = []
    try:
        for requested_phase in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            age = round(requested_phase / (2 * np.pi * 12 * env.control_dt))
            env.ages[2] = age
            command = teacher.act()[2, teacher.channels]
            np.testing.assert_array_equal(env.observation()[2], observation)
            for key, value in state.items():
                np.testing.assert_array_equal(env.fields[key], value)
            np.testing.assert_array_equal(env.previous_action, previous)
            samples.append(
                {
                    "teacher_age_actions": age,
                    "teacher_phase_rad": float(age * env.control_dt * 2 * np.pi * 12),
                    "wing_command": command.tolist(),
                }
            )
    finally:
        env.ages[:] = ages
    commands = np.array([row["wing_command"] for row in samples])
    assert physical_contract(env.model) == contract
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "teacher_sha256": sha256(teacher_path),
        "physical_contract": contract,
        "setup_seconds": setup,
        "audit_seconds": time.perf_counter() - started,
        "physics_transitions": 0,
        "training_updates": 0,
        "instantaneous_observations_identical": True,
        "physical_state_unchanged": True,
        "observation_sha256": hashlib.sha256(observation.tobytes()).hexdigest(),
        "samples": samples,
        "maximum_command_range": float(np.ptp(commands, axis=0).max()),
        "interpretation": "Existing teacher changes wing labels with its timer while current sensory observations are identical.",
        "limitation": "Recurrent history may encode phase; this is not proof of unlearnability or the sole cause of failed hover.",
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.teacher, args.output)
