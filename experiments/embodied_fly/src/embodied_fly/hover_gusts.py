"""Training-only force pulses for learning zero-command velocity recovery."""

import numpy as np


class HoverGusts:
    def __init__(self, env, maximum_speed_cm_s, seed):
        if not np.isfinite(maximum_speed_cm_s) or not 0 < maximum_speed_cm_s <= 3:
            raise ValueError("Gust speed must be finite and in (0, 3] cm/s")
        self.env = env
        self.maximum_speed = maximum_speed_cm_s
        self.rng = np.random.default_rng(seed)
        self.enabled = np.arange(env.n) >= env.n // 4
        self.applied = np.zeros((env.n, 3))
        self.force = np.zeros_like(self.applied)
        self.next_start = np.zeros(env.n, dtype=np.int64)
        self.end = np.zeros(env.n, dtype=np.int64)
        self.events = []
        self.duration = round(0.35 / env.control_dt)
        c = env.wing_forces.config
        self.drag_seconds = np.array(
            [c.horizontal_drag_seconds, c.horizontal_drag_seconds, c.vertical_drag_seconds]
        )
        self.mass = env.wing_forces.mass
        self.recipe = {
            "kind": "training-only physical force pulses",
            "calm_worlds": int((~self.enabled).sum()),
            "disturbed_worlds": int(self.enabled.sum()),
            "maximum_equivalent_drift_cm_s": maximum_speed_cm_s,
            "magnitude_fraction_range": [0.5, 1.0],
            "axes": "uniform world x/y/z, independent random sign",
            "pulse_seconds": 0.35,
            "quiet_seconds_range": [1.0, 2.5],
            "force": "total mass * equivalent drift / existing axis drag timescale",
            "application": "world-frame force at thorax COM; added alongside wing wrench",
            "seed": seed,
            "sampler_resume": "restart from declared seed with physical episode resets",
            "actor_or_critic_observes_schedule": False,
            "evaluation_disturbances": False,
        }

    def reset(self, ids):
        # Called after env.reset has cleared external force on those worlds.
        ids = np.asarray(ids)
        self.applied[ids] = self.force[ids] = 0
        self.end[ids] = 0
        delay = self.rng.integers(
            round(1 / self.env.control_dt), round(2.5 / self.env.control_dt) + 1, len(ids)
        )
        self.next_start[ids] = self.env.ages[ids] + delay

    def before_step(self):
        env = self.env
        starts = np.flatnonzero(self.enabled & (env.ages >= self.next_start))
        for w in starts:
            axis = int(self.rng.integers(3))
            speed = self.maximum_speed * self.rng.uniform(0.5, 1) * self.rng.choice([-1, 1])
            self.force[w] = 0
            self.force[w, axis] = self.mass * speed / self.drag_seconds[axis]
            self.end[w] = env.ages[w] + self.duration
            quiet = int(
                self.rng.integers(round(1 / env.control_dt), round(2.5 / env.control_dt) + 1)
            )
            self.next_start[w] = self.end[w] + quiet
            self.events.append(
                {
                    "world": int(w),
                    "episode_seconds": float(env.ages[w] * env.control_dt),
                    "axis": axis,
                    "equivalent_drift_cm_s": float(speed),
                    "force": float(self.force[w, axis]),
                }
            )
        desired = self.force * ((env.ages < self.end) & self.enabled)[:, None]
        env.fields["xfrc_applied"][:, env.template.thorax_id, :3] += desired - self.applied
        self.applied[:] = desired
