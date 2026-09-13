"""Complete FlyBody anatomy with direct, bounded native MuJoCo actuation.

Keep the upstream CGS convention to reproduce its physical parameters: cm, g, s.
Public metrics convert to SI. No ghost body, gait generator or reference exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from dm_control import mjcf
from dm_control.locomotion.arenas import floors
from flybody.fruitfly.fruitfly import FruitFly

from embodied_fly.observations import (
    append_wing_angles,
    append_wing_velocity,
    wing_angle_indices,
    wing_velocity_indices,
)

PHYSICS_DT = 0.0002
CONTROL_DT = 0.002
SUBSTEPS = 10
FLIGHT_PHYSICS_DT = 0.00005
FLIGHT_CONTROL_DT = 0.0002
NEED_NAMES = ("energy", "hydration", "fatigue", "heat", "injury")


def make_body(preset="walking", *, wing_limits="original"):
    if preset not in ("walking", "flight"):
        raise ValueError(f"Unknown physical preset: {preset}")
    if wing_limits not in ("original", "firm") or (
        wing_limits != "original" and preset != "flight"
    ):
        raise ValueError("Firm wing limits are an explicit flight-only physical pilot")
    physics_dt = FLIGHT_PHYSICS_DT if preset == "flight" else PHYSICS_DT
    control_dt = FLIGHT_CONTROL_DT if preset == "flight" else CONTROL_DT
    fly = FruitFly(
        use_legs=True,
        use_wings=True,
        use_mouth=True,
        use_antennae=True,
        physics_timestep=physics_dt,
        control_timestep=control_dt,
        joint_filter=0.01,
        adhesion_filter=0.007,
    )
    if preset == "flight":
        # Pinned upstream Flying coefficients, with complete limbs and floor
        # contacts retained. Only wing filters are removed; terrestrial actuator
        # dynamics stay unchanged. No external body forces or wing generator.
        for axis in ("yaw", "roll", "pitch"):
            fly.mjcf_model.find("default", axis).general.gainprm[0] = 18
        wing_joint = fly.mjcf_model.find("default", "wing").joint
        wing_joint.damping = 0.007769230
        wing_joint.stiffness = 0.01
        if wing_limits == "firm":
            # Same ranges and actuation; firmer compliant mechanical stops.
            # 0.2 ms is four flight physics steps, above MuJoCo's 2-step floor.
            wing_joint.solreflimit = (0.0002, 1)
        for geom in fly.mjcf_model.find_all("geom"):
            if "fluid" in geom.name:
                geom.fluidshape = "ellipsoid"
                geom.fluidcoef = (1, 0.5, 1.5, 1.7, 1)
        for actuator in fly.mjcf_model.find_all("actuator"):
            if "wing_" in actuator.name:
                actuator.dyntype = "none"
                actuator.dynprm = (1,)
    arena = floors.Floor(size=(3, 3), reflectance=0.08)
    root = arena.mjcf_model
    root.compiler.boundmass = 0
    root.compiler.boundinertia = 0
    spawn = root.worldbody.add("site", pos=(0, 0, 0.1278))
    fly.create_root_joints(arena.attach(fly, spawn))
    spawn.remove()
    root.option.timestep = physics_dt
    root.visual.map.znear = 0.001
    root.visual.map.zfar = 50
    root.statistic.extent = 4.01
    root.visual.quality.shadowsize = 4096
    root.visual.headlight.ambient = (0.22, 0.22, 0.22)
    root.visual.headlight.diffuse = (0.6, 0.6, 0.6)
    arena._ground_texture.rgb1 = (0.095, 0.10, 0.11)
    arena._ground_texture.rgb2 = (0.14, 0.15, 0.16)
    arena._ground_texture.markrgb = (0.20, 0.22, 0.24)
    for ground in arena.ground_geoms:
        ground.friction = (0.5,)
        ground.solref = (0.001, 1)
        ground.solimp = (0.95, 0.99, 0.01)
    fly.mjcf_model.find("default", "adhesion-collision").geom.friction = (1.0,)
    # Same adjacent wing/leg exclusions as upstream walking tasks. Wing/floor,
    # foot/floor and the remaining self contacts stay physical.
    for body in fly.mjcf_model.find_all("body"):
        if any(part in body.name for part in ("coxa", "femur", "tibia", "tarsus", "claw")):
            for wing in ("wing_left", "wing_right"):
                fly.mjcf_model.contact.add("exclude", body1=body.name, body2=wing)
    physics = mjcf.Physics.from_mjcf_model(root)
    model = physics.model.ptr
    model.vis.global_.offwidth = 1600
    model.vis.global_.offheight = 1000
    # Explicit limits for the constant-gain wing/adhesion channels, in addition
    # to upstream position-servo force limits and bounded input/activation.
    for i in range(model.nu):
        if not model.actuator_forcelimited[i]:
            lo, hi = model.actuator_ctrlrange[i]
            gain = model.actuator_gainprm[i, 0]
            model.actuator_forcelimited[i] = True
            model.actuator_forcerange[i] = (lo * gain, hi * gain)
    return physics, fly


class FlyEnvironment:
    def __init__(self, preset="walking", *, wing_limits="original"):
        self.preset = preset
        self.wing_limits = wing_limits
        self.control_dt = FLIGHT_CONTROL_DT if preset == "flight" else CONTROL_DT
        self.physics, self.fly = make_body(preset, wing_limits=wing_limits)
        self.model = self.physics.model.ptr
        self.data = self.physics.data.ptr
        self.substeps = round(self.control_dt / self.model.opt.timestep)
        self.thorax_id = self.model.body("walker/thorax").id
        self.joint_ids = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
        self.qpos_indices = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_indices = self.model.jnt_dofadr[self.joint_ids]
        self.action_names = [self.model.actuator(i).name for i in range(self.model.nu)]
        self.joint_names = [self.model.joint(i).name for i in self.joint_ids]
        self.wing_velocity_indices = wing_velocity_indices(self.model)
        self.wing_angle_indices = wing_angle_indices(self.model)
        self.wing_joint_ids = np.array(
            [
                self.model.joint(name).id
                for name in (
                    "walker/wing_yaw_left",
                    "walker/wing_roll_left",
                    "walker/wing_pitch_left",
                    "walker/wing_yaw_right",
                    "walker/wing_roll_right",
                    "walker/wing_pitch_right",
                )
            ]
        )
        self.low = self.model.actuator_ctrlrange[:, 0].copy()
        self.high = self.model.actuator_ctrlrange[:, 1].copy()
        self.walking_inactive = np.array(
            [
                any(
                    part in name
                    for part in ("wing_", "antenna", "rostrum", "haustellum", "labrum")
                )
                for name in self.action_names
            ]
        )
        self.passive_action = (
            2
            * (np.clip(np.zeros(self.model.nu), self.low, self.high) - self.low)
            / (self.high - self.low)
            - 1
        ).astype(np.float32)
        self.foot_geoms = np.array(
            [
                self.model.geom(f"walker/tarsal_claw_T{leg}_{side}_collision").id
                for leg in (1, 2, 3)
                for side in ("left", "right")
            ]
        )
        self.support_geoms = {
            i
            for i in range(self.model.ngeom)
            if any(part in self.model.geom(i).name for part in ("tarsus", "tarsal", "claw"))
        }
        self.reset()

    def reset(self, yaw=0.0):
        mujoco.mj_resetData(self.model, self.data)
        # Initialization only. Subsequent root motion comes exclusively from mj_step.
        self.data.qpos[3:7] = (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2))
        for jid in self.joint_ids:
            name = self.model.joint(jid).name
            if "wing_" in name:
                address = self.model.jnt_qposadr[jid]
                self.data.qpos[address] = self.model.qpos_spring[address]
        self.command = np.zeros(3, np.float32)  # forward cm/s, lateral cm/s, yaw rad/s
        self.needs = np.zeros(len(NEED_NAMES), np.float32)
        self.previous_action = np.zeros(self.model.nu, np.float32)
        self.maximum_disallowed_ground_force = 0.0
        self.maximum_wing_limit_violation = np.zeros(6)
        mujoco.mj_forward(self.model, self.data)
        self.mean_sensors = self.data.sensordata.copy()
        return self.observation()

    def observation(self, extended_wing_velocity=False, *, wing_angles=False):
        if wing_angles and not extended_wing_velocity:
            raise ValueError("Wing-angle extension requires the existing velocity extension")
        rotation = self.data.xmat[self.thorax_id].reshape(3, 3)
        # Preserve v1 checkpoint input coordinates: MuJoCo mjOBJ_BODY uses the
        # principal inertia frame, NOT the anatomical xmat axes. Six components
        # still contain complete measured local velocity. Use anatomical_velocity
        # for command tracking; do not silently reinterpret trained features.
        body_velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.thorax_id, body_velocity, 1
        )
        foot_touch = np.zeros(6, np.float32)
        for contact in self.data.contact:
            for geom in (contact.geom1, contact.geom2):
                foot_touch[self.foot_geoms == geom] = 1.0
        ranges = self.model.jnt_range[self.joint_ids]
        q = self.data.qpos[self.qpos_indices]
        normalized_q = (
            2 * (q - ranges[:, 0]) / np.maximum(ranges[:, 1] - ranges[:, 0], 1e-4) - 1
        )
        # Causal body feedback only. No phase, time, ghost or future trajectory.
        observation = np.concatenate(
            (
                np.clip(normalized_q, -5, 5),
                np.clip(self.data.qvel[self.qvel_indices] / 100, -10, 10),
                self.actuator_activation(),
                body_velocity / np.array([20, 20, 20, 10, 10, 10]),
                rotation[2],
                foot_touch,
                self.previous_action,
                self.command / (10, 10, 10),
                self.needs,
            )
        ).astype(np.float32)
        if extended_wing_velocity:
            observation = append_wing_velocity(
                observation, self.data.qvel, self.wing_velocity_indices
            )
        if wing_angles:
            observation = append_wing_angles(
                observation, self.data.qpos, self.wing_angle_indices
            )
        return observation

    def actuator_activation(self):
        """One effective input per actuator, retaining the 78-channel schema.

        Filtered actuators expose their activation; unfiltered flight wings expose
        the directly applied control. Walking values exactly match v1 data.act.
        This preserves dimensions, not a claim of flight checkpoint compatibility.
        """
        effective = self.data.ctrl.copy()
        filtered = self.model.actuator_actadr >= 0
        effective[filtered] = self.data.act[self.model.actuator_actadr[filtered]]
        return effective

    def anatomical_velocity(self):
        """Angular then linear velocity in thorax axes (forward x, upright z)."""
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.thorax_id, velocity, 1
        )
        return velocity

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (self.model.nu,) or not np.isfinite(action).all():
            raise ValueError("Invalid actor output")
        self.previous_action = np.clip(action, -1, 1)
        self.data.ctrl[:] = self.low + (self.previous_action + 1) * 0.5 * (
            self.high - self.low
        )
        sensor_total = np.zeros(self.model.nsensordata)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
            ranges = self.model.jnt_range[self.wing_joint_ids]
            angles = self.data.qpos[self.wing_angle_indices]
            violation = np.maximum(angles - ranges[:, 1], ranges[:, 0] - angles)
            np.maximum(
                self.maximum_wing_limit_violation,
                violation,
                out=self.maximum_wing_limit_violation,
            )
            sensor_total += self.data.sensordata
            for cid, contact in enumerate(self.data.contact):
                body1 = self.model.geom_bodyid[contact.geom1]
                body2 = self.model.geom_bodyid[contact.geom2]
                if (body1 == 0) != (body2 == 0):
                    robot_geom = contact.geom2 if body1 == 0 else contact.geom1
                    if robot_geom not in self.support_geoms:
                        force = np.empty(6)
                        mujoco.mj_contactForce(self.model, self.data, cid, force)
                        self.maximum_disallowed_ground_force = max(
                            self.maximum_disallowed_ground_force, float(abs(force[0]))
                        )
            if np.any(self.data.warning.number) or not np.isfinite(self.data.qpos).all():
                raise RuntimeError("MuJoCo numerical failure")
        self.mean_sensors = sensor_total / self.substeps
        return self.observation()

    def walking_action(self, action):
        """First curriculum stage: raw-zero inactive channels, no gait supplied."""
        result = np.asarray(action, dtype=np.float32).copy()
        result[self.walking_inactive] = self.passive_action[self.walking_inactive]
        return result

    def advance(self, action, duration):
        """Hold a bounded command for an explicit controller interval.

        Useful for checking a 500 Hz walking actor with the finer flight physics
        and motor clock. Every underlying substep still checks contacts/warnings.
        """
        repeats = round(duration / self.control_dt)
        if repeats < 1 or not np.isclose(
            repeats * self.control_dt, duration, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Controller interval must be a positive multiple of the motor interval"
            )
        for _ in range(repeats):
            observation = self.step(action)
        return observation

    def report(self):
        model = self.model
        return {
            "units": "CGS retained from FlyBody; SI conversions shown explicitly",
            "body_mass_kg": float(model.body_mass.sum() * 0.001),
            "bodies_including_world": model.nbody,
            "degrees_of_freedom": model.nv,
            "actuators": model.nu,
            "observation_size": len(self.observation()),
            "physical_preset": self.preset,
            "wing_limit_profile": self.wing_limits,
            "wing_limit_solref": model.jnt_solref[self.wing_joint_ids].tolist(),
            "physics_hz": 1 / model.opt.timestep,
            "control_hz": 1 / self.control_dt,
            "activation_states": model.na,
            "activation_observation": "78 effective actuator inputs; filtered state or direct control",
            "root_is_free": bool(model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE),
            "source": "TuragaLab/flybody@d015e9bfe441bd90ae431bac24c55cb74bdbce26",
            "joints": [
                {"name": model.joint(i).name, "range_rad": model.jnt_range[i].tolist()}
                for i in self.joint_ids
            ],
            "actuation": [
                {
                    "name": model.actuator(i).name,
                    "transmission_type": int(model.actuator_trntype[i]),
                    "ctrl_range": model.actuator_ctrlrange[i].tolist(),
                    "force_range_CGS": model.actuator_forcerange[i].tolist(),
                    "gear": model.actuator_gear[i].tolist(),
                }
                for i in range(model.nu)
            ],
        }

    def preview(self, path: Path):
        from PIL import Image, ImageDraw

        path.parent.mkdir(parents=True, exist_ok=True)
        option = mujoco.MjvOption()
        option.geomgroup[3:] = 0
        panels = []
        with mujoco.Renderer(self.model, height=600, width=800) as renderer:
            for azimuth, elevation, distance in ((135, -25, 0.95), (90, -80, 0.95)):
                camera = mujoco.MjvCamera()
                camera.lookat[:] = self.data.xpos[self.thorax_id]
                camera.azimuth, camera.elevation, camera.distance = (
                    azimuth,
                    elevation,
                    distance,
                )
                renderer.update_scene(self.data, camera=camera, scene_option=option)
                panels.append(Image.fromarray(renderer.render()))
        board = Image.new("RGB", (1600, 690), "#111519")
        for i, panel in enumerate(panels):
            board.paste(panel, (800 * i, 65))
        draw = ImageDraw.Draw(board)
        draw.text(
            (24, 20),
            "EMBODIED FLY  |  Complete anatomy / static preview",
            fill="#ffc31f",
            font_size=26,
        )
        draw.text(
            (24, 664),
            "78 bounded actuators   |   legs + wings + mouth + antennae   |   zero physics steps",
            fill="#e6e1db",
            font_size=17,
        )
        board.save(path)
        path.with_suffix(".json").write_text(json.dumps(self.report(), indent=2) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()
    environment = FlyEnvironment()
    environment.preview(args.preview)
    print(
        json.dumps(
            {
                k: v
                for k, v in environment.report().items()
                if k not in ("joints", "actuation")
            },
            indent=2,
        )
    )
