"""Sampled healthy-walk contact and posture diagnostics, separate from reward."""

import mujoco
import numpy as np

from .bodies import ACTION_SCALE, LEGS, STAND
from .evaluate import rollout
from .train import load_checkpoint


def inspect_walk(checkpoint, seconds=12):
    net, saved = load_checkpoint(checkpoint)
    outcome, frames, model = rollout(
        net, "healthy", trials=1, seconds=seconds, capture=True
    )
    data = mujoco.MjData(model)
    support, clearance, heights, tilts, penetrations = [], [], [], [], []
    trunk_frames = 0
    terminals = [model.geom(f"{leg}_terminal").id for leg in LEGS]
    base = model.body("base").id
    for state in frames:
        data.qpos[:] = state["qpos"]
        data.qvel[:] = state["qvel"]
        data.ctrl[:] = STAND + ACTION_SCALE * state["action"]
        mujoco.mj_forward(model, data)
        if state["time"] < 1:
            continue
        contact = np.zeros(4, bool)
        trunk = False
        for c in data.contact:
            if c.dist > 0:
                continue
            a, b = int(c.geom1), int(c.geom2)
            if model.geom_bodyid[a] == 0 or model.geom_bodyid[b] == 0:
                for i, terminal in enumerate(terminals):
                    contact[i] |= terminal in (a, b)
                trunk |= base in (model.geom_bodyid[a], model.geom_bodyid[b])
        support.append(contact)
        clearance.append(
            [data.geom_xpos[g, 2] - model.geom_size[g, 0] for g in terminals]
        )
        heights.append(data.qpos[2])
        tilts.append(np.degrees(np.arccos(np.clip(data.xmat[base, 8], -1, 1))))
        penetrations.append(
            max(0, -float(np.min(data.contact.dist))) if data.ncon else 0
        )
        trunk_frames += trunk
    return (
        {
            "checkpoint": str(checkpoint),
            "training_seconds": saved["cumulative_training_seconds"],
            "outcome": outcome,
            "diagnostic_window": "after first second, sampled every 20 ms",
            "leg_order": LEGS,
            "terminal_support_fraction": np.mean(support, axis=0).tolist(),
            "maximum_foot_clearance_m": np.max(clearance, axis=0).tolist(),
            "mean_base_height_m": float(np.mean(heights)),
            "base_height_min_max_m": [float(np.min(heights)), float(np.max(heights))],
            "mean_tilt_degrees": float(np.mean(tilts)),
            "trunk_contact_frames": int(trunk_frames),
            "maximum_sampled_penetration_m": float(max(penetrations)),
            "allowed_support_test": {
                "completed": outcome["completed_with_allowed_support"],
                "checked_every_physics_step": outcome[
                    "support_checked_every_physics_step"
                ],
                "force_threshold_n": outcome["support_force_threshold_n"],
                "unintended_force_by_geom_n": {
                    n: f
                    for n, f in outcome["maximum_force_by_geom_n"].items()
                    if n not in outcome["allowed_support_geom_names"] and f > 0
                },
            },
        },
        frames,
        model,
    )
