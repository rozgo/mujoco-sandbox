"""Causal ground-motor diagnostics, explicitly ineligible for policy acceptance.

Matched copies of one fly use the same actor. Selected actuator groups receive
current-state reference commands to locate the source of instability. No weights,
body properties or live poses are changed; all executed feedback is retained.
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.braking import BrakingTeacher
from embodied_fly.evaluate import load_actor
from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now


def intervention_masks(env):
    names = env.template.action_names
    wings = np.array(["wing_" in name for name in names])
    legs = np.array(
        [any(part in name for part in ("coxa", "femur", "tibia", "tarsus")) for name in names]
    )
    adhesion = np.array(["adhere_claw" in name for name in names])
    return {
        "student": np.zeros(env.model.nu, bool),
        "reference_wings": wings,
        "reference_nonwalking": env.template.walking_inactive.copy(),
        "reference_foot_adhesion": adhesion,
        "reference_leg_joints": legs,
        "reference_legs_and_adhesion": legs | adhesion,
        "reference_all": np.ones(env.model.nu, bool),
    }


def run(args):
    if args.seconds < 0.002:
        raise ValueError("Capture must contain at least one control interval")
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = evidence()
    started = time.perf_counter()
    torch.set_num_threads(4)
    actor, checkpoint = load_actor(args.checkpoint, args.graph, args.device)
    if not actor.motor_only:
        raise ValueError("This diagnostic expects the explicit motor-only actor")
    env = FlyBatch(14, 14, 14, preset="wing_motion")
    if checkpoint["physical_contract"] != physical_contract(env.model):
        raise ValueError("Diagnostic must use the same physical fly as training")
    groups = intervention_masks(env)
    assert len(groups) * 2 == env.n
    task = np.repeat([0, 1], len(groups))
    env.reset(np.arange(env.n), yaw=np.repeat([-0.1, 0.1], len(groups)))
    env.command[:, 0] = task
    teacher = BrakingTeacher(env, args.teacher, args.device, track_command=True)
    masks = np.tile(np.stack(list(groups.values())), (2, 1))
    memory = actor.initial_state(env.n)
    mujoco.mj_saveModel(env.model, str(args.output / "model.mjb"))
    setup = time.perf_counter() - started
    history, measures = [], []
    started = time.perf_counter()
    with torch.no_grad():
        for step in range(round(args.seconds / env.control_dt)):
            obs = env.observation()
            result = actor(torch.as_tensor(obs, device=args.device), memory)
            memory = result.state
            student, reference = result.action.cpu().numpy(), teacher.act().cpu().numpy()
            executed = np.where(masks, reference, student)
            history.append(
                {
                    "time": np.full(env.n, step * env.control_dt),
                    "qpos": env.fields["qpos"].copy(),
                    "qvel": env.fields["qvel"].copy(),
                    "activation": env.fields["act"].copy(),
                    "ctrl": env.fields["ctrl"].copy(),
                    "observation": obs,
                    "action": executed,
                    "student_action": student,
                    "reference_action": reference,
                    "wing_activity": env.wing_forces.activity.copy(),
                    "wing_wrench": env.wing_forces.wrench.copy(),
                }
            )
            env.step(executed)
            measures.append(
                {
                    "up": env.fields["xmat"][:, env.template.thorax_id, 8].copy(),
                    "height_cm": env.fields["qpos"][:, 2].copy(),
                    "forbidden_load": env.forbidden_peak.copy() / env.body_weight,
                    "speed_error": env.velocity()[:, 3] - env.command[:, 0],
                }
            )
    elapsed = time.perf_counter() - started
    results = []
    for i in range(env.n):
        intervention = list(groups)[i % len(groups)]
        case = ("stand" if task[i] == 0 else "walk") + "_" + intervention
        states = {k: np.stack([h[k][i] for h in history]) for k in history[0]}
        values = {k: np.array([m[k][i] for m in measures]) for k in measures[0]}
        bad = np.flatnonzero((values["up"] < 0.5) | (values["height_cm"] < 0.06))
        np.savez_compressed(args.output / f"{case}.npz", **states)
        results.append(
            {
                "case": case,
                "intervention": intervention,
                "replaced_actuators": [
                    n for n, m in zip(env.template.action_names, masks[i]) if m
                ],
                "first_envelope_exit_s": float((bad[0] + 1) * env.control_dt)
                if len(bad)
                else None,
                "stable": not bool(len(bad)),
                "minimum_up": float(values["up"].min()),
                "max_forbidden_load": float(values["forbidden_load"].max()),
                "speed_rmse_cm_s": float(np.sqrt(np.mean(values["speed_error"] ** 2))),
                "state_sha256": sha256(args.output / f"{case}.npz"),
            }
        )
    report = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "teacher_sha256": sha256(args.teacher),
        "physical_contract": physical_contract(env.model),
        "model_sha256": sha256(args.output / "model.mjb"),
        "diagnostic_only": True,
        "policy_acceptance_eligible": False,
        "one_checkpoint_for_all_cases": True,
        "training_seconds": 0,
        "setup_seconds": setup,
        "stepping_seconds": elapsed,
        "seconds_per_case": len(history) * env.control_dt,
        "physics_worlds": env.n,
        "physics_hz": 5000,
        "control_hz": 500,
        "warning_count": int(env.fields["warning"].sum()),
        "results": results,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                **{k: v for k, v in report.items() if k not in ("provenance", "results")},
                "results": [
                    {k: v for k, v in r.items() if k != "replaced_actuators"} for r in results
                ],
            }
        ),
        flush=True,
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=1)
    parser.add_argument("--device", default="cuda")
    run(parser.parse_args())
