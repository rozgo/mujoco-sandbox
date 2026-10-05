"""Geometric inspection of reset/scratch poses, without advancing simulation."""

from itertools import product

import mujoco
import numpy as np

from .scene import CAMERAS, JOINTS, load_scene


def inspect_scene(path=None):
    model, data = load_scene() if path is None else load_scene(path)
    initial_contacts = int(data.ncon)
    initial_gap = mujoco.mj_geomDistance(model, data, model.geom("needle").id,
                                       model.geom("tissue_phantom").id, 1.0, None)
    weight = float(data.qfrc_bias[model.joint("stage_z").dofadr[0]])
    samples = list(product(np.linspace(-.175, .175, 5), (-.075, 0, .075),
                           (-.008, 0, .050), (0,), (0, .006)))
    samples += list(product(np.linspace(-.175, .175, 5), (-.075, 0, .075),
                            (.050,), (.018,), (0, .006)))
    failures, minimum_gap, finite = [], float("inf"), True
    for q in samples:
        # Scratch configurations only: neither viewer nor preview steps physics.
        for name, value in zip(JOINTS, q):
            data.joint(name).qpos[0] = value
        mujoco.mj_forward(model, data)
        finite = finite and bool(np.isfinite(data.qpos).all() and np.isfinite(data.qacc).all())
        minimum_gap = min(minimum_gap, mujoco.mj_geomDistance(
            model, data, model.geom("needle").id, model.geom("tissue_phantom").id, 1.0, None))
        for contact in data.contact:
            if contact.dist < -1e-8:
                failures.append({"qpos": list(q), "geoms": [model.geom(contact.geom1).name,
                                                           model.geom(contact.geom2).name],
                                 "penetration_m": float(-contact.dist)})
    limits_valid = bool(np.all(model.actuator_ctrllimited) and np.all(model.actuator_forcelimited)
                        and np.all(model.actuator_forcerange[:, 1] > 0))
    warnings = int(sum(w.number for w in data.warning))
    return {
        "success": not failures and initial_contacts == 0 and limits_valid and finite and warnings == 0,
        "preset": "static_v1", "dof": model.nv, "actuators": model.nu,
        "camera_names": list(CAMERAS), "initial_contacts": initial_contacts,
        "initial_needle_surface_gap_mm": float(initial_gap * 1000),
        "sampled_poses": len(samples), "sampled_minimum_needle_surface_gap_mm": float(minimum_gap * 1000),
        "penetrations": failures, "finite_state": finite, "mujoco_warnings": warnings,
        "force_and_command_limits_enabled": limits_valid,
        "z_supported_weight_n": weight, "z_actuator_limit_n": JOINTS["stage_z"][-1],
        "moving_mass_kg": sum(v[1] for v in JOINTS.values()),
        "simulated_seconds": float(data.time),
    }
