"""Hover-first physical curriculum; no PID, imitation or deployed helper."""

from collections import deque

import numpy as np

from embodied_fly.hover_ppo import HoverPPOReward, smooth_cost
from embodied_fly.motor_focus import MotorTasks

REFERENCE_HEIGHT_CM = 1.86651647


class HoverOnlyTasks(MotorTasks):
    def __init__(self, env, seed):
        self.ready = False
        self.widening = 0.0
        self.recent = deque(maxlen=64)
        self.curriculum_updates = 0
        super().__init__(env, seed)
        self.task_ids[:] = 2
        self.ready = True
        self.reset(np.arange(env.n))

    def reset(self, ids):
        if not self.ready:
            return super().reset(ids)
        ids = np.asarray(ids, np.int64)
        if not len(ids):
            return
        e, n = self.env, len(ids)
        state = {k: np.repeat(v[None], n, axis=0) for k, v in self.air.items()}
        state["qpos"][:, 2] = REFERENCE_HEIGHT_CM
        # Half remain at the exact approved cold-wing start. Offsets change only
        # on resets, after successful on-policy episodes unlock widening.
        scale = (ids % 2) * self.widening
        state["qpos"][:, 2] += self.rng.uniform(-0.02, 0.02, n) * scale
        state["qvel"][:, :3] += self.rng.uniform(-1, 1, (n, 3)) * scale[:, None]
        e.reset(ids, state=state)
        e.requested_height_cm[ids] = REFERENCE_HEIGHT_CM
        e.requested_xy_cm[ids] = self.air["qpos"][:2]
        e.command[ids] = 0
        e.needs[ids] = 0
        self.start[ids] = self.air["qpos"][:3]
        self.start[ids, 2] = REFERENCE_HEIGHT_CM
        self.heading[ids] = 0

    def observe_episode(self, record):
        self.recent.append(record)

    def advance_curriculum(self):
        if len(self.recent) < 64 or self.widening >= 1:
            return
        if (
            all(not r["failed"] and r["simulated_seconds"] >= 5 for r in self.recent)
            and np.mean([r["return"] / r["simulated_seconds"] for r in self.recent]) > 1.5
        ):
            self.widening = min(1.0, round(self.widening + 0.1, 2))
            self.curriculum_updates += 1
            self.recent.clear()

    def report(self):
        return {
            "world_proportions": {"stand": 0, "walk": 0, "hover": 1},
            "initial_height_cm": REFERENCE_HEIGHT_CM,
            "widening": self.widening,
            "curriculum_updates": self.curriculum_updates,
            "progress_gate": "64 complete episodes >=5 s, none failed, mean reward rate >1.5; then widen by 0.1",
            "maximum_height_offset_cm": 0.02,
            "maximum_initial_linear_speed_cm_s_per_axis": 1,
            "perturbed_fraction": 0.5,
            "cold_stationary_wings": True,
            "changes_only_at_episode_resets": True,
            "teacher_actions": False,
        }


class HoverOnlyReward(HoverPPOReward):
    def __init__(self, env):
        super().__init__(env)
        self.recipe = self.recipe | {
            "version": "hover_only_position_v1",
            "height_scale_cm": 0.1,
            "horizontal_position_cost_rate": 1.0,
            "horizontal_position_scale_cm": 0.1,
            "joint_effort_cost_rate": 0.002,
            "action_difference_or_wing_stillness_penalty": False,
            "position_feedback": "two anatomical target-error inputs, ideal simulator sensing",
        }

    def __call__(self, previous_action):
        _, failed, terms = super().__call__(previous_action)
        e = self.env
        terms["height"] = -2 * smooth_cost(
            (e.fields["qpos"][:, 2] - e.requested_height_cm) / 0.1
        )
        terms["horizontal_position"] = -smooth_cost(
            np.linalg.norm(e.fields["qpos"][:, :2] - e.requested_xy_cm, axis=1) / 0.1
        )
        limits = np.maximum(np.abs(e.model.actuator_forcerange).max(axis=1), 1e-6)
        terms["joint_effort"] = -0.002 * np.mean(
            (e.fields["actuator_force"] / limits) ** 2, axis=1
        )
        return (
            (sum(terms.values()) * e.control_dt - failed).astype(np.float32),
            failed,
            terms,
        )
