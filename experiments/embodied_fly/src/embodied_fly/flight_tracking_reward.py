"""One reward correction: score goal-directed velocity, including braking.

The reference velocity exists only in reward computation. It does not issue a
motor action, change a body force, filter measured motion or bypass the brain.
"""

from dataclasses import asdict, dataclass

import numpy as np

from embodied_fly.hover_only import HoverBalancedReward


@dataclass(frozen=True)
class TrackingConfig:
    approach_seconds: float = 0.5
    maximum_correction_cm_s: float = 0.5

    def __post_init__(self):
        if (
            not np.isfinite([self.approach_seconds, self.maximum_correction_cm_s]).all()
            or min(self.approach_seconds, self.maximum_correction_cm_s) <= 0
        ):
            raise ValueError("Finite positive approach time and correction speed required")


DEFAULT_TRACKING = TrackingConfig()


def desired_velocity(position, target, target_velocity, config=DEFAULT_TRACKING):
    correction = (np.asarray(target) - position) / config.approach_seconds
    speed = np.linalg.norm(correction, axis=-1, keepdims=True)
    correction *= np.minimum(1.0, config.maximum_correction_cm_s / np.maximum(speed, 1e-12))
    return np.asarray(target_velocity) + correction


def velocity_scores(velocity, desired, vertical_scale_cm_s=2.0):
    residual = np.asarray(velocity) - desired
    return {
        "vertical_velocity": 1 / np.sqrt(1 + (residual[..., 2] / vertical_scale_cm_s) ** 2),
        "horizontal_velocity": 1
        / np.sqrt(1 + (np.linalg.norm(residual[..., :2], axis=-1) / 0.5) ** 2),
    }


class FlightTrackingReward(HoverBalancedReward):
    def __init__(self, env, tasks, vertical_speed_scale_cm_s=2.0):
        super().__init__(env, vertical_speed_scale_cm_s)
        self.tasks = tasks
        self.tracking_config = TrackingConfig()
        tasks.reward_description = (
            "bounded physical scores with goal-directed velocity; see reward_recipe"
        )
        self.recipe.update(
            version="flight_tracking_scores_v1",
            velocity_target="analytic current target velocity + bounded position-error correction",
            correction=asdict(self.tracking_config),
            velocity_reference_scope="reward only; no actuator commands or applied forces",
            position_and_stability_weights_unchanged=True,
            task_specific_wing_pattern=False,
        )

    def __call__(self, previous_action):
        _, failed, terms = super().__call__(previous_action)
        e = self.env
        target = np.column_stack((e.requested_xy_cm, e.requested_height_cm))
        desired = desired_velocity(
            e.fields["qpos"][:, :3],
            target,
            self.tasks.target_velocity_cm_s,
            self.tracking_config,
        )
        adjusted = velocity_scores(
            e.fields["qvel"][:, :3], desired, self.vertical_speed_scale_cm_s
        )
        terms.update({key: np.where(failed, 0.0, value) for key, value in adjusted.items()})
        return (sum(terms.values()) * e.control_dt - failed).astype(np.float32), failed, terms
