"""Airborne reset curriculum and physical rewards; no runtime flight teacher.

Only the first physical frame of training-split expert captures initializes an
episode. Live actions come from the shared graph actor. This stage establishes
wing control from airborne starts; it does not represent takeoff or landing.
"""

import json

import numpy as np

from embodied_fly.provenance import sha256


class FlightResets:
    def __init__(self, env, path, rng):
        if env.preset not in ("flight", "wing_motion"):
            raise ValueError("Airborne resets require the flight physical preset")
        manifest = json.loads((path / "manifest.json").read_text())
        capture_preset = manifest.get("physical_preset", "flight")
        if capture_preset != env.preset:
            raise ValueError("Reset corpus must match the flight force model")
        if manifest["control_hz"] != 1 / env.control_dt:
            raise ValueError("Capture and flight control clocks differ")
        eligible = [
            item
            for item in manifest["episodes"]
            if not item["failure"]
            and not item.get("physical_failure")
            and item["final_upright"] >= 0.5
        ]
        if len(eligible) < 4:
            raise ValueError("Need four valid separate reset episodes")
        validation = set(
            np.random.default_rng(1193)
            .permutation(len(eligible))[: max(1, len(eligible) // 4)]
            .tolist()
        )
        self.states, commands, sources = [], [], []
        for i, item in enumerate(eligible):
            if i in validation:
                continue
            file = path / f"episode_{item['episode']:03d}.npz"
            with np.load(file, allow_pickle=False) as data:
                state = {
                    name: data[key][0].copy()
                    for name, key in (
                        ("qpos", "qpos"),
                        ("qvel", "qvel"),
                        ("act", "activation"),
                        ("ctrl", "ctrl"),
                    )
                }
                # Original legacy observation indices 375:378 carry command / 10.
                command = data["observation"][0, 375:378].copy() * 10
            for name, value in state.items():
                if value.shape != env.fields[name].shape[1:] or not np.isfinite(value).all():
                    raise ValueError(f"Invalid captured {name}")
            if not np.isfinite(command).all() or state["qpos"][2] <= 0.8:
                raise ValueError(
                    "Airborne reset requires finite command and height above 8 mm"
                )
            self.states.append(state)
            commands.append(command)
            sources.append({"file": file.name, "sha256": sha256(file), "frame": 0})
        self.env, self.rng = env, rng
        self.commands = np.asarray(commands, np.float32)
        self.task_ids = np.zeros(env.n, np.int64)
        self.report = {
            "scope": "First training-split physical frame only; no live reference or teacher",
            "physical_preset": env.preset,
            "manifest_sha256": sha256(path / "manifest.json"),
            "validation_indices_excluded": sorted(validation),
            "sources": sources,
            "commands_cm_s_and_rad_s": self.commands.tolist(),
        }

    def reset(self, ids):
        ids = np.asarray(ids, np.int64)
        if not len(ids):
            return
        choices = self.rng.integers(len(self.states), size=len(ids))
        state = {
            key: np.stack([self.states[i][key] for i in choices]) for key in self.states[0]
        }
        self.env.reset(ids, state=state)
        self.env.command[ids] = self.commands[choices]
        self.task_ids[ids] = choices


class FlightOutcomeReward:
    def __init__(self, env):
        self.env = env
        self.height = np.zeros(env.n)
        self.quaternion = np.zeros((env.n, 4))
        self.recipe = {
            "forward_and_lateral_velocity_tracking_rate": 2.0,
            "horizontal_width_cm_s": 5.0,
            "vertical_velocity_tracking_rate": 2.0,
            "vertical_width_cm_s": 5.0,
            "height_tracking_rate": 1.0,
            "height_width_cm": 0.3,
            "orientation_tracking_rate": 1.0,
            "orientation_error": "1 - squared dot of unit root quaternions; exp(-error / 0.1)",
            "airborne_alive_rate": 0.5,
            "forbidden_support_cost_rate": 2.0,
            "failure_height_below_cm": 0.8,
            "failure_upright_below": 0.5,
            "physical_failure_penalty_once": 1.0,
            "units": f"CGS state; rates times actual {env.control_dt:g} s action interval; terminal penalty once",
            "excluded": "No wing-phase/action imitation or direct lift target in physical reward",
        }

    def reset(self, ids):
        self.height[ids] = self.env.fields["qpos"][ids, 2]
        self.quaternion[ids] = self.env.fields["qpos"][ids, 3:7]

    def __call__(self, previous_action):
        env = self.env
        qpos, velocity = env.fields["qpos"], env.fields["qvel"][:, :3]
        upright = env.fields["xmat"][:, env.template.thorax_id, 8]
        failed = (qpos[:, 2] < 0.8) | (upright < 0.5)
        horizontal = velocity[:, :2] - env.command[:, :2]
        orientation_error = np.clip(
            1 - np.sum(qpos[:, 3:7] * self.quaternion, axis=1) ** 2, 0, 1
        )
        terms = {
            "horizontal_velocity": 2 * np.exp(-np.sum(horizontal**2, axis=1) / 25),
            "vertical_velocity": 2 * np.exp(-((velocity[:, 2] / 5) ** 2)),
            "height": np.exp(-(((qpos[:, 2] - self.height) / 0.3) ** 2)),
            "orientation": np.exp(-orientation_error / 0.1),
            "alive": 0.5 * (~failed),
            "support": -2 * np.minimum(env.forbidden_peak / env.body_weight, 5),
        }
        reward = sum(terms.values()) * env.control_dt - failed.astype(float)
        return reward.astype(np.float32), failed, terms
