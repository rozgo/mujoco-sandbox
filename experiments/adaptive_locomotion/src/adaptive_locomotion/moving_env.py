"""Moving-support coordinates, bounded physical motion and shared balance policy."""

import numpy as np

from .bodies import CONTROL_DT, LIMITS, PRESETS
from .env import rotation
from .limb_loss import LOSS_BODIES
from .moving_scene import AXES
from .standing_env import StandingEnv, moving

# Joint order x/y/z/yaw/pitch/roll. Frequencies in Hz, angular amplitudes radians.
MOTIONS = {
    "still": ([0, 0, 0, 0, 0, 0], [0.2] * 6),
    "translate": ([0.30, 0.18, 0, 0, 0, 0], [0.20, 0.27, 0.2, 0.2, 0.2, 0.2]),
    "yaw": ([0, 0, 0, 0.65, 0, 0], [0.2, 0.2, 0.2, 0.17, 0.2, 0.2]),
    "heave": ([0, 0, 0.10, 0, 0, 0], [0.2, 0.2, 0.40, 0.2, 0.2, 0.2]),
    "rock": ([0, 0, 0, 0, 0.18, 0.14], [0.2, 0.2, 0.2, 0.2, 0.24, 0.31]),
    "combined": (
        [0.22, 0.14, 0.07, 0.45, 0.14, 0.11],
        [0.19, 0.23, 0.35, 0.14, 0.21, 0.28],
    ),
}


def to_local(points, origin, matrix):
    return np.einsum("nji,nj->ni", matrix, points - origin)


def point_velocity(points, origin, linear, angular):
    return linear + np.cross(angular, points - origin)


class MovingEnv(StandingEnv):
    def __init__(
        self,
        num_envs=512,
        seed=21,
        moving_profile="mixed",
        motion="combined",
        relative=True,
        motion_scale=1.0,
        cases=None,
        **kwargs,
    ):
        if motion not in MOTIONS or not 0 <= motion_scale <= 1.5:
            raise ValueError("Invalid motion preset or scale")
        self.moving_ready = False
        self.relative = relative
        self.motion, self.motion_scale = motion, motion_scale
        self.deck_origin = np.zeros((num_envs, 3))
        self.deck_rot = np.tile(np.eye(3), (num_envs, 1, 1))
        self.deck_linear = np.zeros((num_envs, 3))
        self.deck_angular = np.zeros((num_envs, 3))
        self.local_anchor = np.zeros((num_envs, 3))
        self.relative_pos = np.zeros((num_envs, 3))
        self.relative_vel = np.zeros((num_envs, 3))
        self.relative_gyro = np.zeros((num_envs, 3))
        self.deck_world = np.zeros(num_envs, bool)
        self.amplitudes = np.zeros((num_envs, 6))
        self.frequencies = np.zeros((num_envs, 6))
        self.phases = np.zeros((num_envs, 6))
        if cases is None:
            # Half moving; static rehearsal spans all bodies and existing terrain.
            static = [(PRESETS["healthy"], "flat")] + [(b, "flat") for b in LOSS_BODIES]
            static += [
                (PRESETS["healthy"], s)
                for s in ("pads", "slope_x_12", "slope_y_12", "gap_fl", "gap_fr")
            ]
            if moving_profile == "mixed":
                cases = static + [(PRESETS["healthy"], "moving")]
                half = num_envs // 2
                kwargs["case_counts"] = [
                    half // len(static) + (i < half % len(static))
                    for i in range(len(static))
                ] + [num_envs - half]
            else:
                cases = [(PRESETS["healthy"], "moving")]
        kwargs.pop("profile", None)
        super().__init__(num_envs, seed, cases=cases, profile="mixed", **kwargs)
        self.deck_sensors = []
        for g, sl, (_, surface) in zip(
            self.groups, self.slices, self.cases, strict=True
        ):
            if surface == "moving":
                self.deck_world[sl] = True
                g.deck_aadr = np.array([g.model.actuator(f"deck_{a}").id for a in AXES])
                g.deck_qadr = np.array(
                    [g.model.joint(f"deck_{a}").qposadr[0] for a in AXES]
                )
                g.deck_dadr = np.array(
                    [g.model.joint(f"deck_{a}").dofadr[0] for a in AXES]
                )
                self.deck_sensors.append(
                    [
                        g.batch.sensor(f"deck_{s}")
                        for s in ("framepos", "framequat", "framelinvel", "frameangvel")
                    ]
                )
            else:
                self.deck_sensors.append(None)
        self.moving_ready = True
        self.reset(np.arange(self.n))

    def reset(self, ids):
        super().reset(ids)
        if not self.moving_ready:
            return
        ids = np.asarray(ids, int)
        self.amplitudes[ids], self.frequencies[ids] = MOTIONS[self.motion]
        if self.schedule:
            for i in ids[self.deck_world[ids]]:
                name = self.rng.choice(list(MOTIONS))
                self.amplitudes[i], self.frequencies[i] = MOTIONS[name]
        self.amplitudes[ids] *= self.motion_scale
        self.phases[ids] = 0
        if self.randomize:
            self.amplitudes[ids] *= self.rng.uniform(0.75, 1.05, (len(ids), 1))
            self.frequencies[ids] *= self.rng.uniform(0.85, 1.15, (len(ids), 1))
            self.phases[ids] = self.rng.uniform(-np.pi, np.pi, (len(ids), 6))
        for g, sl in zip(self.groups, self.slices, strict=True):
            if not hasattr(g, "deck_aadr"):
                continue
            local = ids[(ids >= sl.start) & (ids < sl.stop)] - sl.start
            if len(local):
                g.qvel[local[:, None], g.deck_dadr] = 0
                g.ctrl[local[:, None], g.deck_aadr] = 0
                g.batch.forward(local)
        self.refresh()
        self.local_anchor[ids] = self.relative_pos[ids]
        self.history[ids] = self.obs()[ids, None, :]

    def refresh(self):
        super().refresh()
        if not self.moving_ready:
            return
        self.relative_vel[:] = self.vel
        self.relative_gyro[:] = self.gyro
        for g, sl, sensors in zip(
            self.groups, self.slices, self.deck_sensors, strict=True
        ):
            if sensors is None:
                continue
            origin, quat, linear, angular = sensors
            self.deck_origin[sl], self.deck_rot[sl] = origin, rotation(quat)
            self.deck_linear[sl], self.deck_angular[sl] = linear, angular
            self.relative_pos[sl] = to_local(self.pos[sl], origin, self.deck_rot[sl])
            robot_rot = rotation(g.qpos[:, 3:7])
            carrier = point_velocity(self.pos[sl], origin, linear, angular)
            self.relative_vel[sl] = np.einsum(
                "nji,nj->ni", robot_rot, g.qvel[:, :3] - carrier
            )
            self.relative_gyro[sl] = self.gyro[sl] - np.einsum(
                "nji,nj->ni", robot_rot, angular
            )

    def support_obs(self):
        features = super().support_obs()
        if not self.moving_ready or not self.relative:
            return features
        for g, sl, sensors in zip(
            self.groups, self.slices, self.deck_sensors, strict=True
        ):
            if sensors is None:
                continue
            robot_rot = rotation(g.qpos[:, 3:7])
            local_error = self.relative_pos[sl] - self.local_anchor[sl]
            local_error[:, 2] = 0
            error_world = np.einsum("nij,nj->ni", self.deck_rot[sl], local_error)
            features[sl, 8:11] = self.relative_vel[sl]
            features[sl, 11:13] = (
                2 * np.einsum("nji,nj->ni", robot_rot, error_world)[:, :2]
            )
            # Seven formerly unused channels: actual current support twist and
            # height. Ideal support-state sensing; no desired/future motion input.
            features[sl, 13:16] = np.einsum(
                "nji,nj->ni", robot_rot, self.deck_linear[sl]
            )
            features[sl, 16:19] = np.einsum(
                "nji,nj->ni", robot_rot, self.deck_angular[sl]
            )
            features[sl, 19] = self.relative_pos[sl, 2] - 0.30
        features[moving(self.commands)] = 0
        return np.clip(features, -3, 3)

    def targets(self):
        t = self.steps[:, None] * CONTROL_DT
        # Raised-cosine startup: zero position/velocity at reset even with phases.
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.minimum(t / 1.5, 1))
        return (
            self.amplitudes
            * ramp
            * np.sin(2 * np.pi * self.frequencies * t + self.phases)
        )

    def step(self, action):
        old_action, old_tip = self.action.copy(), self.tip_positions.copy()
        old_origin, old_rot = self.deck_origin.copy(), self.deck_rot.copy()
        target = self.targets()
        for g, sl in zip(self.groups, self.slices, strict=True):
            if hasattr(g, "deck_aadr"):
                g.ctrl[:, g.deck_aadr] = target[sl]
        reward, done, fell, info = super().step(action)
        mask = self.deck_world & ~moving(self.commands)
        drift = np.linalg.norm((self.relative_pos - self.local_anchor)[:, :2], axis=1)
        height = self.relative_pos[:, 2]
        # Exact rigid transport of each old tip removes platform displacement;
        # compare tangential residual, including the angular omega-cross-r effect.
        old_local = np.einsum("nji,nkj->nki", old_rot, old_tip - old_origin[:, None])
        new_local = np.einsum(
            "nji,nkj->nki",
            self.deck_rot,
            self.tip_positions - self.deck_origin[:, None],
        )
        tip_speed = np.linalg.norm(
            (new_local - old_local)[:, :, :2] / CONTROL_DT, axis=2
        )
        slip = ((self.tip_forces > 5) * np.minimum(tip_speed**2, 1)).sum(1)
        height_error = np.maximum(0.25 - height, 0) + np.maximum(height - 0.35, 0)
        effort = ((self.torque / LIMITS) ** 2).sum(1)
        rate = (((self.action - old_action) * self.valid) ** 2).sum(1)
        standing_reward = (
            2 * np.exp(-(self.relative_vel[:, :2] ** 2).sum(1) / 0.0064)
            + np.exp(-(np.maximum(drift - 0.025, 0) ** 2) / 0.01)
            + 0.5 * np.exp(-(self.relative_gyro**2).sum(1) / 0.09)
            + 0.5
            - 3 * (self.up[:, :2] ** 2).sum(1)
            - 80 * height_error**2
            - 0.025 * effort
            - 0.025 * rate
            - 0.3 * self.relative_vel[:, 2] ** 2
            - 0.3 * slip
            - self.idle_support_weight * self.support_cost
            - self.idle_drift_weight * np.maximum(drift - 0.025, 0)
        )
        numerical = ~np.isfinite(self.q).all(1) | ~np.isfinite(self.vel).all(1)
        balance_fell = (
            (height < 0.10) | (self.up[:, 2] < 0.30) | (drift > 0.55) | numerical
        )
        standing_reward[balance_fell] -= 10
        self.returns[mask] += standing_reward[mask] - reward[mask]
        reward[mask] = standing_reward[mask]
        # Override stationary-world termination only in moving worlds. Original
        # observation baseline still uses these fair physical evaluation bounds.
        fell[mask] = balance_fell[mask]
        done[mask] = balance_fell[mask] | (self.steps[mask] >= self.idle_episode_steps)
        info.update(
            relative_drift_m=drift,
            relative_clearance_m=height,
            relative_speed_m_s=np.linalg.norm(self.relative_vel[:, :2], axis=1),
            relative_slip_m_s=tip_speed,
            deck_target=target,
        )
        return reward, done, fell, info
