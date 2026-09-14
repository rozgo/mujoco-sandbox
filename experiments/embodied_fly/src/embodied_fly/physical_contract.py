"""One physical fly for every motor task; exclude observer sensors from identity."""

import hashlib
import json
from dataclasses import asdict

import numpy as np

from embodied_fly.wing_motion import config_for_model
from embodied_fly.wing_position import POSITION, is_position

ARRAYS = [
    "body_parentid",
    "body_mass",
    "body_inertia",
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "body_gravcomp",
    "jnt_type",
    "jnt_bodyid",
    "jnt_axis",
    "jnt_pos",
    "jnt_range",
    "jnt_limited",
    "jnt_stiffness",
    "jnt_solref",
    "jnt_solimp",
    "dof_armature",
    "dof_damping",
    "dof_frictionloss",
    "geom_type",
    "geom_bodyid",
    "geom_size",
    "geom_pos",
    "geom_quat",
    "geom_contype",
    "geom_conaffinity",
    "geom_condim",
    "geom_friction",
    "geom_solref",
    "geom_solimp",
    "geom_margin",
    "geom_gap",
    "geom_fluid",
    "actuator_trntype",
    "actuator_trnid",
    "actuator_dyntype",
    "actuator_gaintype",
    "actuator_biastype",
    "actuator_dynprm",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_gear",
    "actuator_ctrlrange",
    "actuator_forcerange",
    "actuator_forcelimited",
    "actuator_ctrllimited",
    "actuator_actlimited",
    "actuator_actrange",
    "tendon_stiffness",
    "tendon_damping",
    "tendon_frictionloss",
    "tendon_range",
    "eq_type",
    "eq_obj1id",
    "eq_obj2id",
    "eq_active0",
    "eq_data",
    "pair_geom1",
    "pair_geom2",
    "pair_friction",
    "pair_solref",
    "pair_solimp",
    "exclude_signature",
    "qpos0",
    "qpos_spring",
]
OPTIONS = [
    "timestep",
    "gravity",
    "wind",
    "density",
    "viscosity",
    "integrator",
    "solver",
    "cone",
    "jacobian",
    "iterations",
    "ls_iterations",
    "tolerance",
    "impratio",
    "enableflags",
    "disableflags",
    "noslip_iterations",
    "noslip_tolerance",
]


def physical_contract(model):
    bodies = [model.body(f"walker/wing_{s}").id for s in ("left", "right")]
    geoms = np.isin(model.geom_bodyid, bodies)
    if (
        np.any(model.body_mass[bodies])
        or np.any(model.body_inertia[bodies])
        or np.any(model.geom_contype[geoms])
        or np.any(model.geom_conaffinity[geoms])
        or np.any(model.geom_fluid)
        or model.opt.density
        or model.opt.viscosity
    ):
        raise ValueError(
            "Canonical fly requires massless, non-colliding, non-aerodynamic wings"
        )
    digest = hashlib.sha256()
    for name in ARRAYS:
        array = np.ascontiguousarray(getattr(model, name))
        digest.update(f"{name}:{array.shape}:{array.dtype}".encode())
        digest.update(array.tobytes())
    options = {n: np.asarray(getattr(model.opt, n)).tolist() for n in OPTIONS}
    digest.update(json.dumps(options, sort_keys=True).encode())
    force_config = config_for_model(model)
    digest.update(json.dumps(asdict(force_config), sort_keys=True).encode())
    position = is_position(model)
    if position:
        digest.update(json.dumps(asdict(POSITION), sort_keys=True).encode())
    contract = {
        "schema": "fly-motor-physical-contract-v1",
        "sha256": digest.hexdigest(),
        "preset": "wing_position" if position else "wing_motion",
        "body_and_force_law_shared_across_tasks": True,
        "wing_mass_inertia_collision_aerodynamics": "zero",
        "wing_body_coupling": "custom flight law reads measured wing motion",
        "physics_hz": 1 / model.opt.timestep,
        "scope": "body, joints, contacts, actuation, physics options and force parameters; observer sensors/cameras excluded",
    }
    if force_config.activity_filter_seconds == 0:
        contract["wing_response"] = "instant"
    if getattr(force_config, "yaw_pitch_acceleration", 0):
        contract["independent_heading_control"] = True
    return contract
