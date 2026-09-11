"""Vectorized MuJoCo state, fault scheduling, sensor interface and task reward."""

import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from mjbatch import Batch

from .bodies import (
    ACTION_SCALE,
    CONTROL_DT,
    DT,
    LEGS,
    LIMITS,
    PRESETS,
    STAND,
    allowed_support_names,
    build_model,
    initialize,
    joint_mapping,
)
from .foot_clearance import clearance_cost
from .gait_balance import ContactTiming, body_motion_cost
from .limb_loss import curriculum
from .paired import training_pairs
from .rear_overlap import overlap_cost
from .stride import StrideTracker
from .visible_steps import VisibleSteps

OBS_DIM, CONTEXT_DIM, HISTORY = 66, 20, 25


def rotation(q):
    w, x, y, z = q.T
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        1,
    ).reshape(-1, 3, 3)


class Group:
    def __init__(
        self, body, n, threads, terrain, sensing, timestep, physics_backend="mjbatch"
    ):
        self.body = body
        self.model = build_model(body, terrain, timestep, sensing)
        if physics_backend == "warp":
            from .warp_batch import WarpBatch

            self.batch = WarpBatch(self.model, n)
        else:
            self.batch = Batch(self.model, n, num_threads=threads)
        self.n = n
        self.qpos, self.qvel, self.ctrl = (
            self.batch.bind(k) for k in ("qpos", "qvel", "ctrl")
        )
        self.torque = self.batch.bind("actuator_force")
        self.warnings = self.batch.bind("warning")
        self.slot, self.qadr, self.dadr, self.aadr = joint_mapping(self.model)
        self.gain, self.bias, self.force_range = (
            self.batch.expand(k)
            for k in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
        )
        self.original_gain, self.original_bias = self.gain.copy(), self.bias.copy()
        sensor_ids = [
            i
            for i in range(self.model.nsensor)
            if self.model.sensor(i).name.startswith("support_")
        ]
        self.support_names = [
            self.model.sensor(i).name.removeprefix("support_") for i in sensor_ids
        ]
        first = self.model.sensor_adr[sensor_ids[0]]
        self.support_data = self.batch.bind("sensordata")[
            :, first : first + 4 * len(sensor_ids)
        ].reshape(n, -1, 4)
        self.support_allowed = np.array(
            [name in allowed_support_names(body) for name in self.support_names]
        )
        self.tip_geoms = [self.model.geom(n).id for n in allowed_support_names(body)]
        self.tip_slots = [LEGS.index(n[:2]) for n in allowed_support_names(body)]
        self.tip_sensors = [
            self.support_names.index(n) for n in allowed_support_names(body)
        ]
        self.geom_positions = self.batch.bind("geom_xpos")
        self.rays = (
            [self.batch.sensor(f"range_{i}") for i in range(9)] if sensing else []
        )
        data = mujoco.MjData(self.model)
        initialize(self.model, data)
        self.initial_qpos = data.qpos.copy()

    def set_strength(self, ids, strength):
        strength = strength[:, self.slot]
        cap = LIMITS[self.slot][None, :] * strength
        self.force_range[ids, :, 0] = -cap
        self.force_range[ids, :, 1] = cap
        active = (strength > 0)[:, :, None]
        self.gain[ids] = self.original_gain[ids] * active
        self.bias[ids] = self.original_bias[ids] * active
        if hasattr(self.batch, "mark_model_dirty"):
            self.batch.mark_model_dirty()


class DogEnv:
    def __init__(
        self,
        num_envs=512,
        seed=0,
        bodies=None,
        terrain="flat",
        randomize=True,
        faults=True,
        threads=12,
        timestep=DT,
        history=HISTORY,
        sensing=None,
        randomize_strength=True,
        reward_profile="adaptive",
        support_weight=2.0,
        support_substeps=False,
        stride_weight=0.0,
        balance_weight=0.0,
        body_motion_weight=0.0,
        damage_action_rate_weight=0.0,
        damage_angular_rate_weight=0.0,
        damage_joint_accel_weight=0.0,
        damage_flight_weight=0.0,
        damage_clearance_weight=0.0,
        clearance_scope="all",
        visible_step_weight=0.0,
        rear_overlap_weight=0.0,
        retention_curriculum=False,
        pair_level=None,
        limb_stage=None,
        physics_backend="mjbatch",
        warp_execution="concurrent",
    ):
        if physics_backend not in ("mjbatch", "warp"):
            raise ValueError("Unknown physics backend")
        self.physics_backend = physics_backend
        if warp_execution not in ("serial", "concurrent"):
            raise ValueError("Unknown Warp execution schedule")
        self.warp_execution = warp_execution
        self.n, self.rng = num_envs, np.random.default_rng(seed)
        self.randomize, self.faults, self.terrain = randomize, faults, terrain
        if reward_profile not in ("adaptive", "walk"):
            raise ValueError(reward_profile)
        self.randomize_strength = randomize_strength
        self.retention_curriculum = retention_curriculum
        self.pair_level = pair_level
        self.limb_stage = limb_stage
        if limb_stage and (retention_curriculum or pair_level or bodies is not None):
            raise ValueError(
                "Limb-loss curriculum is separate from partial-damage stages"
            )
        if pair_level is not None and (not retention_curriculum or num_envs < 16):
            raise ValueError(
                "Paired stages require retention curriculum and at least 16 environments"
            )
        self.reward_profile = reward_profile
        self.support_weight = support_weight
        self.support_substeps = support_substeps
        if stride_weight < 0:
            raise ValueError("Stride weight must be nonnegative")
        self.stride_weight = stride_weight
        if (
            min(
                balance_weight,
                body_motion_weight,
                damage_action_rate_weight,
                damage_angular_rate_weight,
                damage_joint_accel_weight,
                damage_flight_weight,
                damage_clearance_weight,
                visible_step_weight,
                rear_overlap_weight,
            )
            < 0
        ):
            raise ValueError("Gait regularization weights must be nonnegative")
        self.balance_weight = balance_weight
        self.body_motion_weight = body_motion_weight
        self.damage_action_rate_weight = damage_action_rate_weight
        self.damage_angular_rate_weight = damage_angular_rate_weight
        self.damage_joint_accel_weight = damage_joint_accel_weight
        self.damage_flight_weight = damage_flight_weight
        self.rear_overlap_weight = rear_overlap_weight
        self.visible_step_weight = visible_step_weight
        self.damage_clearance_weight = damage_clearance_weight
        if clearance_scope not in ("all", "surviving_rear"):
            raise ValueError("Unknown clearance scope")
        self.clearance_scope = clearance_scope
        if (damage_clearance_weight or visible_step_weight) and terrain != "flat":
            raise ValueError("Foot clearance currently requires flat ground")
        self.timestep = timestep
        self.decimation = round(CONTROL_DT / timestep)
        if abs(self.decimation * timestep - CONTROL_DT) > 1e-10:
            raise ValueError("Inexact decimation")
        self.bodies = bodies or [
            PRESETS[k]
            for k in ("healthy", "short_fl", "short_fr", "short_rl", "short_rr")
        ]
        if retention_curriculum and (
            len(self.bodies) != 5 or self.bodies[0] != PRESETS["healthy"]
        ):
            raise ValueError("Retention curriculum requires the five default bodies")
        if num_envs < len(self.bodies):
            raise ValueError("At least one environment per body")
        if pair_level:
            self.bodies += training_pairs(pair_level)
        if limb_stage:
            self.bodies, counts = curriculum(limb_stage, num_envs)
            self.faults = self.randomize_strength = False
        self.groups, self.slices = [], []
        start = 0
        for i, body in enumerate(self.bodies):
            if limb_stage:
                n = counts[i]
            elif pair_level:
                small_n = num_envs // 16
                n = num_envs - 8 * small_n if i == 0 else small_n
            elif retention_curriculum:
                # 75% intact geometry: two thirds remain healthy, one third gets
                # a motor fault. The other 25% has a shortened calf, one per leg.
                short_n = max(1, num_envs // 16)
                n = num_envs - 4 * short_n if i == 0 else short_n
            else:
                n = num_envs // len(self.bodies) + (i < num_envs % len(self.bodies))
            self.groups.append(
                Group(
                    body,
                    n,
                    max(1, round(threads * n / num_envs))
                    if pair_level or limb_stage
                    else max(1, threads // len(self.bodies)),
                    terrain,
                    True if sensing is None else sensing,
                    timestep,
                    physics_backend,
                )
            )
            self.slices.append(slice(start, start + n))
            start += n
        self.pool = (
            ThreadPoolExecutor(max_workers=len(self.groups))
            if len(self.groups) > 1 and physics_backend == "mjbatch"
            else None
        )
        self.q = np.zeros((num_envs, 12))
        self.dq = self.q.copy()
        self.torque = self.q.copy()
        self.pos = np.zeros((num_envs, 3))
        self.vel = self.pos.copy()
        self.gyro = self.pos.copy()
        self.up = self.pos.copy()
        self.valid = np.zeros_like(self.q)
        self.scan = np.ones((num_envs, 9), np.float32) * 2
        self.action = np.zeros((num_envs, 12), np.float32)
        self.commands = np.zeros((num_envs, 3), np.float32)
        self.strength = np.ones((num_envs, 12))
        self.context = np.ones((num_envs, CONTEXT_DIM), np.float32)
        self.steps = np.zeros(num_envs, int)
        self.fault_at = np.zeros(num_envs, int)
        self.fault_joint = np.zeros(num_envs, int)
        self.fault_strength = np.ones(num_envs)
        self.history = np.zeros((num_envs, history, OBS_DIM), np.float32)
        self.start_x = np.zeros(num_envs)
        self.last_x = np.zeros(num_envs)
        self.max_x = np.zeros(num_envs)
        self.returns = np.zeros(num_envs)
        self.bad_support_force = np.zeros(num_envs)
        self.support_cost = np.zeros(num_envs)
        self.tip_positions = np.zeros((num_envs, 4, 3))
        self.tip_forces = np.zeros((num_envs, 4))
        self.tip_radii = np.zeros((num_envs, 4))
        for g, s in zip(self.groups, self.slices, strict=True):
            self.tip_radii[s, g.tip_slots] = g.model.geom_size[g.tip_geoms, 0]
        self.visible_steps = VisibleSteps(num_envs)
        self.stride = StrideTracker(num_envs)
        self.contact_timing = ContactTiming(num_envs)
        self.reset(np.arange(num_envs))
        if physics_backend == "warp":
            # Compile/capture without stepping; charge cold work to setup, not
            # the bounded training budget or synchronized warm benchmark.
            for group in self.groups:
                group.batch.graph(1 if support_substeps else self.decimation)

    def refresh(self):
        self.q[:] = STAND
        self.dq[:] = 0
        self.torque[:] = 0
        for g, s in zip(self.groups, self.slices, strict=True):
            self.q[s, g.slot] = g.qpos[:, g.qadr]
            self.dq[s, g.slot] = g.qvel[:, g.dadr]
            self.torque[s, g.slot] = g.torque[:, g.aadr]
            self.pos[s] = g.qpos[:, :3]
            rot = rotation(g.qpos[:, 3:7])
            self.vel[s] = np.einsum("nji,nj->ni", rot, g.qvel[:, :3])
            self.gyro[s] = g.qvel[:, 3:6]
            self.up[s] = rot[:, 2, :]
            if g.rays:
                values = np.concatenate(g.rays, axis=1)
                self.scan[s] = np.where(values >= 0, np.minimum(values, 2), 2)
            force = np.linalg.norm(g.support_data[:, ~g.support_allowed, 1:4], axis=-1)
            self.bad_support_force[s] = force.sum(1)
            weight = float(g.model.body_mass.sum() * 9.81)
            # Both load and contact persistence matter. A weak incidental brush
            # costs less than using a knee to carry the body. No contact is hidden.
            self.support_cost[s] = 0.25 * np.minimum(force / 5, 1).sum(
                1
            ) + 2 * np.minimum(force.sum(1) / weight, 2)
            self.tip_positions[s] = 0
            self.tip_forces[s] = 0
            self.tip_positions[s, g.tip_slots] = g.geom_positions[:, g.tip_geoms]
            self.tip_forces[s, g.tip_slots] = np.linalg.norm(
                g.support_data[:, g.tip_sensors, 1:4], axis=-1
            )

    def reset(self, ids):
        ids = np.asarray(ids, dtype=int)
        if not len(ids):
            return
        n = len(ids)
        self.action[ids] = 0
        self.steps[ids] = 0
        self.returns[ids] = 0
        self.commands[ids] = [0.55, 0, 0]
        self.strength[ids] = 1
        if self.randomize:
            self.commands[ids, 0] = self.rng.uniform(0.3, 0.8, n)
            self.commands[ids, 1] = self.rng.uniform(-0.1, 0.1, n)
            self.commands[ids, 2] = self.rng.uniform(-0.3, 0.3, n)
            if self.randomize_strength and not self.retention_curriculum:
                affected = self.rng.random(n) < 0.5
                j = self.rng.integers(0, 12, n)
                self.strength[ids[affected], j[affected]] = self.rng.uniform(
                    0.5, 1, affected.sum()
                )
        self.fault_at[ids] = self.rng.integers(100, 300, n) if self.faults else 100000
        self.fault_joint[ids] = self.rng.integers(0, 12, n)
        self.fault_strength[ids] = self.rng.uniform(0.4, 0.8, n)
        if self.retention_curriculum:
            self.fault_at[ids] = 100000
            intact = ids[ids < self.slices[0].stop]
            affected = intact[self.rng.random(len(intact)) < 1 / 3]
            # Half start weak; half lose torque during a continuous walk.
            split = self.rng.random(len(affected)) < 0.5
            initial, delayed = affected[split], affected[~split]
            self.strength[initial, self.fault_joint[initial]] = self.fault_strength[
                initial
            ]
            self.fault_at[delayed] = self.rng.integers(100, 250, len(delayed))
        for g, s in zip(self.groups, self.slices, strict=True):
            global_ids = ids[(ids >= s.start) & (ids < s.stop)]
            local = global_ids - s.start
            if not len(local):
                continue
            g.batch.reset(local)
            g.qpos[local] = g.initial_qpos
            g.qvel[local] = 0
            if self.randomize:
                g.qpos[local[:, None], g.qadr] += self.rng.uniform(
                    -0.035, 0.035, (len(local), len(g.slot))
                )
                g.qpos[local, 1] = self.rng.uniform(-0.15, 0.15, len(local))
                g.qvel[local, :] = self.rng.uniform(
                    -0.05, 0.05, (len(local), g.model.nv)
                )
            g.ctrl[local] = STAND[g.slot]
            g.set_strength(local, self.strength[global_ids])
            g.batch.forward(local)
            self.valid[global_ids[:, None], g.slot] = 1
            self.context[global_ids, :4] = g.body.calf
            for leg in (*g.body.absent, *g.body.absent_legs):
                self.context[global_ids, LEGS.index(leg)] = 0
        self.context[ids, 4:16] = self.strength[ids]
        self.context[ids, 16:] = 1
        self.refresh()
        self.visible_steps.reset(ids, self.tip_positions, self.tip_forces)
        self.stride.reset(ids, self.tip_positions, self.tip_forces)
        self.contact_timing.reset(ids, self.tip_forces)
        self.start_x[ids] = self.pos[ids, 0]
        self.max_x[ids] = self.last_x[ids] = self.pos[ids, 0]
        self.history[ids] = self.obs()[ids, None, :]

    def obs(self):
        # Fixed nominal normalization; no private morphology/strength in this path.
        return np.concatenate(
            (
                (self.q - STAND) * self.valid,
                self.dq * 0.05,
                self.gyro * 0.25,
                self.up,
                self.action,
                self.commands,
                self.valid,
                self.scan * 0.5,
            ),
            1,
        ).astype(np.float32)

    def critic_obs(self):
        return np.concatenate(
            (self.obs(), self.context, self.vel, self.pos[:, 2:3]), 1
        ).astype(np.float32)

    def step(self, action):
        old_dq = self.dq.copy()
        old_tips = self.tip_positions.copy() if self.damage_clearance_weight else None
        action = np.clip(np.asarray(action), -3, 3)
        old_action = self.action.copy()
        events = np.flatnonzero(self.steps == self.fault_at)
        if len(events):
            self.strength[events, self.fault_joint[events]] = self.fault_strength[
                events
            ]
            self.context[events, 4:16] = self.strength[events]
        for g, s in zip(self.groups, self.slices, strict=True):
            local = events[(events >= s.start) & (events < s.stop)] - s.start
            if len(local):
                g.set_strength(local, self.strength[local + s.start])
            g.ctrl[:] = STAND[g.slot] + ACTION_SCALE * action[s][:, g.slot]

        def advance(g):
            if not self.support_substeps:
                g.batch.step(nstep=self.decimation)
                g.support_peaks = np.linalg.norm(g.support_data[:, :, 1:4], axis=-1)
                g.support_impulses = g.support_peaks * CONTROL_DT
                return
            g.support_peaks = np.zeros((g.n, len(g.support_names)))
            g.support_impulses = np.zeros_like(g.support_peaks)
            for _ in range(self.decimation):
                g.batch.step(nstep=1)
                force = np.linalg.norm(g.support_data[:, :, 1:4], axis=-1)
                np.maximum(g.support_peaks, force, out=g.support_peaks)
                g.support_impulses += force * self.timestep

        if (
            self.physics_backend == "warp"
            and self.warp_execution == "concurrent"
            and not self.support_substeps
        ):
            for group in self.groups:
                group.batch.step_async(self.decimation)
            for group in self.groups:
                group.batch.wait_step()
                group.support_peaks = np.linalg.norm(
                    group.support_data[:, :, 1:4], axis=-1
                )
                group.support_impulses = group.support_peaks * CONTROL_DT
        elif self.pool:
            list(self.pool.map(advance, self.groups))
        else:
            for group in self.groups:
                advance(group)
        if any(np.any(g.warnings[:, :, 1]) for g in self.groups):
            raise FloatingPointError(
                "MuJoCo numerical warning; refusing an automatically corrected trajectory"
            )
        self.steps += 1
        previous_scan = self.scan.copy()
        self.refresh()
        # Scan delivery alternates 2/3 control steps: exact average 20 Hz.
        new_scan = (self.steps * CONTROL_DT // 0.05) != (
            (self.steps - 1) * CONTROL_DT // 0.05
        )
        self.scan[~new_scan] = previous_scan[~new_scan]
        self.action[:] = action
        verror = np.sum((self.vel[:, :2] - self.commands[:, :2]) ** 2, 1)
        track = np.exp(-verror / 0.25)
        turn = np.exp(-((self.gyro[:, 2] - self.commands[:, 2]) ** 2) / 0.25)
        upright = self.up[:, 0] ** 2 + self.up[:, 1] ** 2
        progress = np.clip((self.pos[:, 0] - self.last_x) / CONTROL_DT, -1, 1)
        effort = np.sum((self.torque / LIMITS) ** 2, 1)
        rate = np.sum((action - old_action) ** 2, 1)
        reward = (
            2 * track
            + 0.5 * turn
            + 0.3
            + 0.3 * progress
            - 1.0 * upright
            - 0.025 * effort
            - 0.006 * rate
        )
        reward -= 0.02 * np.sum(self.gyro[:, :2] ** 2, 1)
        if self.limb_stage == "front":
            # Escape the valid-but-stationary solution in the focused extension.
            reward += 0.7 * progress * (self.context[:, :4] < 1).any(1)
        reward -= 0.1 * self.vel[:, 2] ** 2
        reward -= self.support_weight * self.support_cost
        if self.balance_weight:
            # Privileged training-only gating: do not demand an intact gait from
            # shortened/missing limbs or weakened motors. Actor inputs unchanged.
            healthy = (self.context[:, :4] == 1).all(1) & (self.strength >= 0.99).all(1)
            moving = np.linalg.norm(self.commands[:, :2], axis=1) > 0.15
            reward -= (
                self.balance_weight
                * healthy
                * moving
                * self.contact_timing.update(self.tip_forces)
            )
        if self.body_motion_weight:
            reward -= self.body_motion_weight * body_motion_cost(self.vel, self.gyro)
        # Soft training costs only: no action filter, phase template or change to
        # the physical servo. Ignore semantic outputs for nonexistent joints.
        damaged = (self.valid < 1).any(1)
        reward -= damaged * (
            self.damage_action_rate_weight
            * np.sum(((action - old_action) * self.valid) ** 2, 1)
            + self.damage_angular_rate_weight * np.sum(self.gyro[:, :2] ** 2, 1)
            + self.damage_joint_accel_weight
            * np.sum(((self.dq - old_dq) * self.valid / CONTROL_DT) ** 2, 1)
        )
        if self.damage_flight_weight:
            # Restore support continuity independently of healthy-only stride
            # shaping. Only existing feet / designated distal stumps count.
            # This is a soft reward at 50 Hz, not a gait clock or a constraint.
            moving = np.linalg.norm(self.commands[:, :2], axis=1) > 0.15
            unsupported = ~(self.tip_forces > 1.0).any(1)
            reward -= self.damage_flight_weight * damaged * moving * unsupported
        if self.rear_overlap_weight:
            reward -= self.rear_overlap_weight * overlap_cost(
                self.tip_forces, self.valid, self.commands
            )
        if self.damage_clearance_weight:
            reward -= self.damage_clearance_weight * clearance_cost(
                old_tips,
                self.tip_positions,
                self.tip_radii,
                self.valid,
                self.commands,
                scope=self.clearance_scope,
            )
        if self.stride_weight or self.visible_step_weight:
            # The task command expressed in world coordinates supplies direction,
            # not a desired foot trajectory or a phase shared across the legs.
            world_command = np.empty((self.n, 2))
            for g, s in zip(self.groups, self.slices, strict=True):
                rot = rotation(g.qpos[:, 3:7])
                world_command[s] = np.einsum(
                    "nij,nj->ni", rot[:, :2, :2], self.commands[s, :2]
                )
            speed = np.linalg.norm(world_command, axis=1)
            direction = world_command / np.maximum(speed[:, None], 1e-8)
            if self.visible_step_weight:
                reward += self.visible_step_weight * self.visible_steps.update(
                    self.tip_positions,
                    self.tip_forces,
                    self.tip_radii,
                    self.valid,
                    direction,
                    speed > 0.15,
                )
            stride_reward = self.stride.update(
                self.tip_positions, self.tip_forces, direction, speed > 0.15
            )
            if self.limb_stage:
                stride_reward *= (self.context[:, :4] == 1).all(1)
            reward += self.stride_weight * stride_reward
        if self.reward_profile == "walk":
            # Healthy-only posture preferences, never a prescribed gait phase or
            # target foot trajectory. These terms leave the adaptive task intact.
            posture = (
                40 * (self.pos[:, 2] - 0.30) ** 2
                + 3 * upright
                + 0.08 * np.sum((self.q - STAND) ** 2, 1)
                + 0.6 * np.sum((self.q[:, ::3] - STAND[::3]) ** 2, 1)
                + 0.014 * rate
                + 0.08 * np.sum(self.gyro[:, :2] ** 2, 1)
            )
            if self.limb_stage:
                posture *= (self.context[:, :4] == 1).all(1)
            reward -= posture
        finite = np.isfinite(self.q).all(1) & np.isfinite(self.vel).all(1)
        fell = (self.up[:, 2] < 0.15) | (self.pos[:, 2] < 0.09) | ~finite
        timeout = self.steps >= 500
        reward[fell] -= 10 if self.limb_stage else 2
        self.last_x[:] = self.pos[:, 0]
        self.max_x = np.maximum(self.max_x, self.pos[:, 0])
        self.returns += reward
        self.history[:, :-1] = self.history[:, 1:]
        self.history[:, -1] = self.obs()
        return (
            reward.astype(np.float32),
            fell | timeout,
            fell,
            {
                "track": track,
                "turn": turn,
                "speed": self.vel[:, 0],
                "progress": self.pos[:, 0] - self.start_x,
                "faults": len(events),
                "bad_support_force_n": self.bad_support_force.copy(),
            },
        )

    def close(self):
        # Native pools are released when their Batch objects are collected.
        if self.pool:
            self.pool.shutdown()
        self.groups.clear()


def benchmark(seconds=3, num_envs=512, threads=12, **kwargs):
    setup = time.perf_counter()
    env = DogEnv(num_envs=num_envs, threads=threads, **kwargs)
    setup = time.perf_counter() - setup
    t = time.perf_counter()
    steps = 0
    while time.perf_counter() - t < seconds:
        _, done, _, _ = env.step(np.zeros((env.n, 12)))
        env.reset(np.flatnonzero(done))
        steps += env.n
    elapsed = time.perf_counter() - t
    result = {
        "setup_seconds": setup,
        "seconds": elapsed,
        "control_transitions": steps,
        "transitions_per_second": steps / elapsed,
        "num_envs": num_envs,
        "threads": threads,
        "bodies": len(env.groups),
        "timestep": env.timestep,
        "full_body_contacts": True,
    }
    env.close()
    return result
