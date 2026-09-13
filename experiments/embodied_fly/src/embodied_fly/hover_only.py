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
        self.promotion_rate = 1.5
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
            and np.mean([r["return"] / r["simulated_seconds"] for r in self.recent])
            > self.promotion_rate
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
            "progress_gate": f"64 complete episodes >=5 s, none failed, mean reward rate >{self.promotion_rate}; then widen by 0.1",
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


class HoverBalancedReward(HoverOnlyReward):
    """Bounded physical scores; valid airborne states retain positive return.

    No controller, phase target or imitation. A physical failure earns exactly
    -1 and terminates. Preserve the old cost recipe for its recorded runs.
    """

    def __init__(self, env):
        super().__init__(env)
        self.recipe = {
            k: v
            for k, v in self.recipe.items()
            if "cost_rate" not in k and k != "cost_formula"
        }
        self.recipe.update(
            version="hover_bounded_scores_v2",
            score_formula="1 / sqrt(1 + normalized_error**2)",
            alive_rate=0.5,
            maximum_reward_rate=6.1,
            minimum_valid_airborne_rate=0.498,
            tracking_score_weights={
                "height": 2,
                "horizontal_position": 1,
                "vertical_velocity": 1,
                "horizontal_velocity": 1,
                "upright": 0.5,
                "angular_velocity": 0.1,
            },
            failure_penalty_once=1,
            failure_forbidden_support_above_bodyweights=0.1,
            termination_incentive="Valid airborne rate stays positive; failed step gets exactly -1",
        )

    def __call__(self, previous_action):
        e = self.env
        q, v = e.fields["qpos"], e.fields["qvel"][:, :3]
        up = e.fields["xmat"][:, e.template.thorax_id, 8]
        failed = (q[:, 2] < 0.5) | (up < 0.5) | (e.forbidden_peak / e.body_weight > 0.1)
        score = lambda error: 1 / np.sqrt(1 + np.asarray(error) ** 2)
        limits = np.maximum(np.abs(e.model.actuator_forcerange).max(axis=1), 1e-6)
        effort = np.clip(e.fields["actuator_force"] / limits, -1, 1)
        terms = {
            "alive": np.full(e.n, 0.5),
            "height": 2 * score((q[:, 2] - e.requested_height_cm) / 0.1),
            "horizontal_position": score(
                np.linalg.norm(q[:, :2] - e.requested_xy_cm, axis=1) / 0.1
            ),
            "vertical_velocity": score(v[:, 2] / 5),
            "horizontal_velocity": score(np.linalg.norm(v[:, :2], axis=1) / 0.5),
            "upright": 0.5 * np.clip(up, 0, 1),
            "angular_velocity": 0.1 * score(np.linalg.norm(e.velocity()[:, :3], axis=1)),
            "joint_effort": -0.002 * np.mean(effort**2, axis=1),
        }
        terms = {k: np.where(failed, 0, value) for k, value in terms.items()}
        return (
            (sum(terms.values()) * e.control_dt - failed).astype(np.float32),
            failed,
            terms,
        )
