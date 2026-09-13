"""Physical ground motor rewards on the canonical fly, without a motor teacher."""

import numpy as np

from embodied_fly.ground_posture import GroundPosture


class GroundMotorReward:
    """Add measured initial-form rewards to command tracking and valid support.

    All terms are rates. The trainer applies one failure penalty and resets the
    failed world; a timeout bootstraps the critic from its final physical state.
    GroundPosture is used only to measure q/qvel, never to generate actions.
    """

    def __init__(self, tasks):
        # Local import keeps the shared PPO mechanics independent of curricula.
        from embodied_fly.ppo import OutcomeReward

        self.env, self.tasks = tasks.env, tasks
        self.tracking = OutcomeReward(self.env, stationary_cost=1.5, stationary_turn_cost=0.25)
        self.posture = GroundPosture(self.env, tasks.ground["qpos"])
        self.recipe = self.tracking.recipe | {
            "ground_wing_position_rate": 2.0,
            "ground_wing_velocity_rate": 1.0,
            "stand_leg_pose_rate": 1.0,
            "ground_body_pose_rate": 0.5,
            "ground_height_rate": 0.5,
            "wing_angle_scale_rad": 0.1,
            "wing_speed_scale_rad_s": 2.0,
            "leg_body_angle_scale_rad": 0.15,
            "height_scale_fraction": 0.1,
            "pose_formula": "weight / (1 + group mean squared error / scale squared)",
            "task_scope": "stand: full initial hinge pose; walk: resting wings and body, legs free",
            "teacher": False,
            "action_mask_or_runtime_override": False,
        }

    def reset(self, ids):
        self.tracking.reset(ids)

    def __call__(self, previous_action):
        reward, failed, terms = self.tracking(previous_action)
        measured = self.posture.measure()
        holding = self.tasks.task_ids == 0
        height_error = self.env.fields["qpos"][:, 2] / self.posture.qref[2] - 1
        extra = {
            "wing_pose": 2 / (1 + measured["wings_angle_mse_rad2"] / 0.1**2),
            "wing_stillness": 1 / (1 + measured["wings_velocity_mse_rad2_s2"] / 2**2),
            "stand_leg_pose": holding / (1 + measured["legs_angle_mse_rad2"] / 0.15**2),
            "body_pose": 0.5 / (1 + measured["body_angle_mse_rad2"] / 0.15**2),
            "body_height": 0.5 / (1 + (height_error / 0.1) ** 2),
        }
        reward += (sum(extra.values()) * self.env.control_dt).astype(np.float32)
        return reward, failed, terms | extra
