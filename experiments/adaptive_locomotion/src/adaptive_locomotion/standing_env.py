"""Command-conditioned balance tasks with real, finite support geometry.

No live pose correction, support constraint, action blending or deployed teacher.
The surface map is used only for reset and reward/acceptance, never actor input.
"""

import numpy as np

from .bodies import ACTION_SCALE, CONTROL_DT, LIMITS, PRESETS, STAND
from .env import DogEnv, rotation
from .limb_loss import LOSS_BODIES
from .standing_surfaces import AGGRESSIVE, GENTLE, SURFACES


def moving(commands):
    return (np.linalg.norm(commands[:, :2], axis=1) > 0.05) | (
        np.abs(commands[:, 2]) > 0.05
    )


class StandingEnv(DogEnv):
    def __init__(
        self,
        num_envs=512,
        seed=12,
        profile="healthy",
        cases=None,
        schedule=True,
        surface_level="gentle",
        **kwargs,
    ):
        if profile not in ("healthy", "mixed"):
            raise ValueError(profile)
        self.balance_ready = False
        self.profile, self.schedule = profile, schedule
        surfaces = {
            "gentle": GENTLE,
            "aggressive": ("flat", *AGGRESSIVE),
            "all": SURFACES,
        }[surface_level]
        self.surface_level = surface_level
        if cases is None:
            # One flat group per body, plus healthy terrain groups. Flat worlds
            # mix walking, standing and walk-stop-walk episodes.
            bodies = [PRESETS["healthy"]] + (
                list(LOSS_BODIES) if profile == "mixed" else []
            )
            cases = [(b, "flat") for b in bodies] + [
                (PRESETS["healthy"], s) for s in surfaces[1:]
            ]
            flat_n = num_envs // 2
            terrain_n = num_envs - flat_n
            counts = [
                flat_n // len(bodies) + (i < flat_n % len(bodies))
                for i in range(len(bodies))
            ]
            counts += [
                terrain_n // (len(surfaces) - 1) + (i < terrain_n % (len(surfaces) - 1))
                for i in range(len(surfaces) - 1)
            ]
        else:
            counts = [
                num_envs // len(cases) + (i < num_envs % len(cases))
                for i in range(len(cases))
            ]
        self.cases = cases
        self.hold_anchor = np.zeros((num_envs, 2))
        self.reference_height = np.zeros(num_envs)
        self.flat_world = np.zeros(num_envs, bool)
        self.command_kind = np.zeros(num_envs, int)
        self.walk_command = np.zeros((num_envs, 3), np.float32)
        self.stop_at = np.zeros(num_envs, int)
        self.resume_at = np.zeros(num_envs, int)
        self.episode_limit = np.full(num_envs, 500, int)
        for k in (
            "bodies",
            "terrain",
            "limb_stage",
            "retention_curriculum",
            "pair_level",
            "faults",
            "randomize_strength",
        ):
            kwargs.pop(k, None)
        super().__init__(
            num_envs,
            seed,
            bodies=[b for b, _ in cases],
            terrain="flat",
            terrain_per_group=[f"stand_{s}" for _, s in cases],
            group_counts=counts,
            faults=False,
            randomize_strength=False,
            healthy_posture_only=True,
            **kwargs,
        )
        self.standing_ranges = []
        for g, sl, (_, surface) in zip(self.groups, self.slices, cases, strict=True):
            self.standing_ranges.append(
                [g.batch.sensor(f"standing_range_{i}") for i in range(4)]
            )
            self.reference_height[sl] = g.initial_qpos[2] - 0.30
            self.flat_world[sl] = surface == "flat"
        self.balance_ready = True
        self.reset(np.arange(self.n))

    def reset(self, ids):
        ids = np.asarray(ids, int)
        randomize = self.randomize
        # The parent flat walking reset moves the trunk sideways independently
        # of its feet. That is invalid at a platform edge. Start from IK-placed
        # collision-safe poses, with small joint/velocity variation instead.
        self.randomize = False
        try:
            super().reset(ids)
        finally:
            self.randomize = randomize
        if not self.balance_ready or not len(ids):
            return
        for g, sl in zip(self.groups, self.slices, strict=True):
            chosen = ids[(ids >= sl.start) & (ids < sl.stop)]
            local = chosen - sl.start
            if not len(local):
                continue
            if randomize:
                g.qpos[local[:, None], g.qadr] += self.rng.uniform(
                    -0.008, 0.008, (len(local), len(g.slot))
                )
                g.qvel[local] = self.rng.uniform(-0.02, 0.02, (len(local), g.model.nv))
            g.ctrl[local] = g.qpos[local[:, None], g.qadr]
            g.batch.forward(local)
            self.action[chosen[:, None], g.slot] = (
                g.qpos[local[:, None], g.qadr] - STAND[g.slot]
            ) / ACTION_SCALE
        self.refresh()
        self.hold_anchor[ids] = self.pos[ids, :2]
        self.command_kind[ids] = 0
        flat = ids[self.flat_world[ids]]
        if self.schedule:
            # 20% pure stand, 45% walk, 35% walk-stop-walk on flat.
            self.command_kind[flat] = self.rng.choice(
                3, len(flat), p=[0.20, 0.45, 0.35]
            )
        self.walk_command[ids] = [0.55, 0, 0]
        if randomize:
            self.walk_command[ids, 0] = self.rng.uniform(0.35, 0.70, len(ids))
        self.stop_at[ids] = self.rng.integers(75, 151, len(ids))
        self.resume_at[ids] = self.stop_at[ids] + self.rng.integers(125, 226, len(ids))
        self.commands[ids] = 0
        self.update_commands(ids)
        self.history[ids] = self.obs()[ids, None, :]

    def set_commands(self, commands, ids=None):
        ids = np.arange(self.n) if ids is None else np.asarray(ids, int)
        commands = np.broadcast_to(np.asarray(commands, np.float32), (len(ids), 3))
        stopped = moving(self.commands[ids]) & ~moving(commands)
        self.hold_anchor[ids[stopped]] = self.pos[ids[stopped], :2]
        self.commands[ids] = commands

    def update_commands(self, ids=None):
        if not self.schedule:
            return
        ids = np.arange(self.n) if ids is None else np.asarray(ids, int)
        kind, steps = self.command_kind[ids], self.steps[ids]
        walking = (kind == 1) | (
            (kind == 2) & ((steps < self.stop_at[ids]) | (steps >= self.resume_at[ids]))
        )
        self.set_commands(self.walk_command[ids] * walking[:, None], ids)

    def support_obs(self):
        features = np.zeros((self.n, 20), np.float32)
        features[:, :4] = self.tip_forces > 1.0
        features[:, 8:11] = self.vel
        for g, sl, rays in zip(
            self.groups, self.slices, self.standing_ranges, strict=True
        ):
            ranges = np.concatenate(rays, axis=1)
            features[sl, 4:8] = (
                np.where(ranges < 0, 1, np.clip(ranges, 0, 1)) - 0.32
            ) * 2
            rot = rotation(g.qpos[:, 3:7])
            error = self.pos[sl, :2] - self.hold_anchor[sl]
            features[sl, 11:13] = 2 * np.einsum("nji,nj->ni", rot[:, :2, :2], error)
        # Balance-only sensor channels leave the approved walking observation
        # interface intact. This is input preprocessing, not policy switching.
        features[moving(self.commands)] = 0
        return np.clip(features, -3, 3)

    def step(self, action):
        was_idle = ~moving(self.commands)
        old_action, old_tip = self.action.copy(), self.tip_positions.copy()
        reward, done, fell, info = super().step(action)
        old_reward = reward.copy()
        drift = np.linalg.norm(self.pos[:, :2] - self.hold_anchor, axis=1)
        height = self.pos[:, 2] - self.reference_height
        height_error = np.maximum(0.25 - height, 0) + np.maximum(height - 0.35, 0)
        tilt = (self.up[:, :2] ** 2).sum(1)
        effort = ((self.torque / LIMITS) ** 2).sum(1)
        rate = (((self.action - old_action) * self.valid) ** 2).sum(1)
        tip_speed = np.linalg.norm(
            (self.tip_positions - old_tip)[:, :, :2] / CONTROL_DT, axis=2
        )
        slip = ((self.tip_forces > 5) * np.minimum(tip_speed**2, 1)).sum(1)
        # No gait phase, pose imitation, mandatory four-foot support or reward for
        # lifting a particular foot. The missing support itself changes dynamics.
        standing_reward = (
            2 * np.exp(-(self.vel[:, :2] ** 2).sum(1) / 0.0064)
            + np.exp(-(np.maximum(drift - 0.025, 0) ** 2) / 0.01)
            + 0.5 * np.exp(-(self.gyro**2).sum(1) / 0.09)
            + 0.5
            - 3 * tilt
            - 80 * height_error**2
            - 0.025 * effort
            - 0.025 * rate
            - 0.3 * self.vel[:, 2] ** 2
            - 0.3 * slip
            - self.support_weight * self.support_cost
        )
        balance_fell = (height < 0.10) | (self.up[:, 2] < 0.30) | (drift > 0.55)
        standing_reward[fell | balance_fell] -= 10
        reward[was_idle] = standing_reward[was_idle]
        new_fell = was_idle & balance_fell
        fell |= new_fell
        done |= new_fell
        self.returns += reward - old_reward
        info.update(
            idle=was_idle,
            hold_drift_m=drift,
            clearance_m=height,
            standing_reward=standing_reward,
        )
        self.update_commands()
        self.history[:, -1] = self.obs()
        return reward.astype(np.float32), done, fell, info
