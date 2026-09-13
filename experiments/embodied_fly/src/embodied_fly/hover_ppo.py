"""Opt-in outcome rewards and reset curriculum for PPO on the unchanged plant."""

import numpy as np

from embodied_fly.motor_focus import MotorTasks


def smooth_cost(error):
    """Quadratic near zero, approximately linear far away, no exponential plateau."""
    return np.sqrt(1 + np.asarray(error) ** 2) - 1


class HoverPPOReward:
    def __init__(self, env):
        self.env = env
        self.recipe = {
            "version": "hover_physical_cost_v1",
            "alive_rate": 2.0,
            "height_cost_rate": 2.0,
            "height_scale_cm": 0.3,
            "vertical_speed_cost_rate": 1.0,
            "vertical_speed_scale_cm_s": 5.0,
            "horizontal_speed_cost_rate": 1.0,
            "horizontal_speed_scale_cm_s": 0.5,
            "tilt_cost_rate": 1.0,
            "angular_speed_cost_rate": 0.1,
            "angular_speed_scale_rad_s": 1.0,
            "forbidden_support_cost_rate": 2.0,
            "failure_height_below_cm": 0.5,
            "failure_upright_below": 0.5,
            "failure_penalty_once": 1.0,
            "cost_formula": "sqrt(1 + normalized_error**2) - 1",
            "height_target": "requested altitude, independent of reset altitude",
            "reward_timing": "rate * actual control_dt; failure penalty once before reset",
            "wing_motion_or_action_imitation": False,
            "reference_controller_or_force_target": False,
        }

    def reset(self, ids):
        pass  # No hidden targets, clocks or reference controller.

    def __call__(self, previous_action):
        e = self.env
        height = e.fields["qpos"][:, 2]
        v = e.fields["qvel"][:, :3]
        angular = e.velocity()[:, :3]
        upright = e.fields["xmat"][:, e.template.thorax_id, 8]
        failed = (height < 0.5) | (upright < 0.5)
        terms = {
            "alive": 2.0 * (~failed),
            "height": -2 * smooth_cost((height - e.requested_height_cm) / 0.3),
            "vertical_velocity": -smooth_cost(v[:, 2] / 5),
            "horizontal_velocity": -smooth_cost(
                np.linalg.norm(v[:, :2] - e.command[:, :2], axis=1) / 0.5
            ),
            "tilt": -(1 - np.clip(upright, -1, 1)),
            "angular_velocity": -0.1 * smooth_cost(np.linalg.norm(angular, axis=1)),
            "support": -2 * np.minimum(e.forbidden_peak / e.body_weight, 5),
        }
        return (
            (sum(terms.values()) * e.control_dt - failed).astype(np.float32),
            failed,
            terms,
        )


class HoverPPOTasks(MotorTasks):
    """Half hover, quarter stand, quarter walk; perturbations only at episode reset.

    Keep half the hover worlds near the inherited cold-wing starts. The other
    half widen linearly with collected simulated experience, never video outcome.
    Neural memory resets only when the physical episode resets.
    """

    def __init__(self, env, seed):
        if env.n % 4:
            raise ValueError("Hover PPO requires a world count divisible by four")
        self.widening = 0.0
        self.curriculum_rng = np.random.default_rng(seed + 771)
        self.curriculum_ready = False
        super().__init__(env, seed)
        self.task_ids[:] = np.tile([0, 1, 2, 2], env.n // 4)
        self.curriculum_ready = True
        self.reset(np.arange(env.n))

    def reset(self, ids):
        super().reset(ids)
        if not self.curriculum_ready:
            return
        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[(self.task_ids[ids] == 2) & (ids % 4 == 3)]
        if not len(ids):
            return
        e, rng, scale = self.env, self.curriculum_rng, self.widening
        state = {k: e.fields[k][ids].copy() for k in ("qpos", "qvel", "act", "ctrl")}
        target = e.requested_height_cm[ids].copy()
        state["qpos"][:, 2] += rng.uniform(-0.2, 0.2, len(ids)) * scale
        state["qvel"][:, 2] = rng.uniform(-5, 5, len(ids)) * scale
        key = np.ix_(np.arange(len(ids)), e.template.wing_angle_indices)
        angles = state["qpos"][key] + rng.uniform(-0.08, 0.08, (len(ids), 6)) * scale
        limits = e.model.jnt_range[e.template.wing_joint_ids]
        state["qpos"][key] = np.clip(angles, limits[:, 0], limits[:, 1])
        state["qvel"][np.ix_(np.arange(len(ids)), e.template.wing_velocity_indices)] += (
            rng.uniform(-5, 5, (len(ids), 6)) * scale
        )
        e.reset(ids, state=state)
        e.requested_height_cm[ids] = target
        e.command[ids] = 0
        e.needs[ids] = 0
        self.start[ids] = state["qpos"][:, :3]

    def report(self):
        return {
            "world_proportions": {"stand": 0.25, "walk": 0.25, "hover": 0.5},
            "hover_start_height_cm": [1.8, 2.2],
            "hover_perturbed_fraction": 0.5,
            "widening": self.widening,
            "full_widening_after_mean_world_experience_seconds": 10,
            "maximum_height_offset_cm": 0.2,
            "maximum_vertical_speed_cm_s": 5,
            "maximum_wing_angle_offset_rad": 0.08,
            "maximum_wing_speed_rad_s": 5,
            "changes_only_at_episode_resets": True,
            "teacher_actions": False,
        }
