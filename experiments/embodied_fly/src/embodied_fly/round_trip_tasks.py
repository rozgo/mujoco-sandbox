"""Mixed stationary/closed-flight commands for the same motor actor."""

import numpy as np

from embodied_fly.hover_only import HoverOnlyTasks
from embodied_fly.round_trip import CASES


def batched_targets(seconds, origins, route_ids, amplitudes):
    seconds = np.asarray(seconds)
    knots = np.asarray((0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 8.0, 12.0))
    values = np.asarray((0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0))
    indices = np.clip(np.searchsorted(knots, seconds, side="right") - 1, 0, 6)
    u = np.clip((seconds - knots[indices]) / (knots[indices + 1] - knots[indices]), 0, 1)
    smooth = u**3 * (10 - 15 * u + 6 * u * u)
    displacement = amplitudes * (
        values[indices] + (values[indices + 1] - values[indices]) * smooth
    )
    axes = np.asarray([c[1] for c in CASES])[route_ids]
    signs = np.asarray([c[2] for c in CASES])[route_ids]
    targets = origins.copy()
    targets[np.arange(len(targets)), axes] += signs * displacement
    return targets


class RoundTripTasks(HoverOnlyTasks):
    def __init__(self, env, seed):
        if env.n < 12 or env.n % 2:
            raise ValueError("Round trips require an even number of worlds, at least 12")
        self.round_trip_ready = False
        self.route_ids = np.full(env.n, 6, dtype=np.int64)
        self.amplitudes = np.full(env.n, 0.15)
        self.duration_scale = np.ones(env.n)
        self.episode_counts = np.zeros(env.n, dtype=np.int64)
        self.window_counts = np.zeros((env.n, 3), dtype=np.int64)
        self.window_peaks = np.zeros((env.n, 3))
        self.hold_drift_peak = np.zeros(env.n)
        self.transitions = np.zeros(7, dtype=np.int64)
        self.position_history = np.zeros((51, env.n, 3))
        self.history_head = 0
        super().__init__(env, seed)
        self.round_trip_ready = True
        self.reset(np.arange(env.n))

    def reset(self, ids):
        super().reset(ids)
        if not self.round_trip_ready:
            return
        ids = np.asarray(ids, dtype=np.int64)
        if not len(ids):
            return
        self.route_ids[ids] = np.where(
            ids < self.env.n // 2, 6, (ids - self.env.n // 2 + self.episode_counts[ids]) % 6
        )
        self.episode_counts[ids] += 1
        self.amplitudes[ids] = self.rng.uniform(0.12, 0.18, len(ids))
        self.duration_scale[ids] = self.rng.uniform(0.85, 1.15, len(ids))
        self.window_counts[ids] = 0
        self.window_peaks[ids] = 0
        self.hold_drift_peak[ids] = 0
        self.position_history[:, ids] = self.env.fields["qpos"][ids, :3]
        self.update_targets()

    def update_targets(self):
        e = self.env
        tau = e.ages * e.control_dt / self.duration_scale
        target = batched_targets(tau, self.start, self.route_ids, self.amplitudes)
        e.requested_xy_cm[:] = target[:, :2]
        e.requested_height_cm[:] = target[:, 2]
        return target

    def after_step(self):
        """Advance only commands, then measure post-step state at that time."""
        e = self.env
        target = self.update_targets()
        age = e.ages * e.control_dt
        tau = age / self.duration_scale
        position = e.fields["qpos"][:, :3]
        error = np.linalg.norm(position - target, axis=1) * 10
        self.history_head = (self.history_head + 1) % 51
        self.position_history[self.history_head] = position
        old = self.position_history[(self.history_head + 1) % 51]
        drift = np.linalg.norm(position - old, axis=1) * 10 / (50 * e.control_dt)
        windows = ((tau >= 3.5) & (tau < 4), (tau >= 6.5) & (tau < 7), age >= 11)
        for j, mask in enumerate(windows):
            self.window_counts[mask, j] += 1
            self.window_peaks[mask, j] = np.maximum(self.window_peaks[mask, j], error[mask])
        self.hold_drift_peak[windows[2]] = np.maximum(
            self.hold_drift_peak[windows[2]], drift[windows[2]]
        )
        self.transitions += np.bincount(self.route_ids, minlength=7)

    def episode_metrics(self, i):
        visited = bool(np.all(self.window_counts[i] > 0))
        position = bool(np.all(self.window_peaks[i] < 0.5))
        drift = bool(self.hold_drift_peak[i] < 1.5)
        return {
            "target_sequence": CASES[self.route_ids[i]][0],
            "target_amplitude_mm": float(self.amplitudes[i] * 10),
            "target_time_scale": float(self.duration_scale[i]),
            "waypoint_window_samples": self.window_counts[i].tolist(),
            "waypoint_window_peak_errors_mm": self.window_peaks[i].tolist(),
            "final_hold_100ms_drift_peak_mm_s": float(self.hold_drift_peak[i]),
            "complete_target_sequence_tracking": visited and position and drift,
            "final_origin_error_mm": float(
                np.linalg.norm(self.env.fields["qpos"][i, :3] - self.start[i]) * 10
            ),
        }

    def advance_curriculum(self):
        # Fixed first pilot: no automatic reset widening or difficulty changes.
        return

    def report(self):
        return {
            "curriculum": "closed target trips plus stationary hover",
            "stationary_worlds": self.env.n // 2,
            "round_trip_worlds": self.env.n // 2,
            "route_ids_are_observer_only": True,
            "future_waypoints_in_observation": False,
            "amplitude_mm_range": [1.2, 1.8],
            "time_scale_range": [0.85, 1.15],
            "initial_reset_widening": self.widening,
            "transitions_by_target_sequence": {
                c[0]: int(n) for c, n in zip(CASES, self.transitions)
            },
            "reward": "unchanged bounded physical scores, relative to current requested position",
            "target_timing": "post-step command advance before reward and timeout bootstrap; next actor reads that same target",
            "measurement_only_drift_seconds": 50 * self.env.control_dt,
            "original_raw_speed_gate": "preserved in reference report; this task measures sustained drift from positions",
            "teacher_actions": False,
            "body_pose_assignment": "initialization and episode resets only",
        }
