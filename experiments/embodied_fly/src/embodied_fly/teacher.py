"""Numerically checked PyTorch inference of the official learned walking teacher."""

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WalkingTeacher(nn.Module):
    def __init__(self, path: Path):
        super().__init__()
        self.manifest = json.loads(path.with_suffix(".json").read_text())
        with np.load(path, allow_pickle=False) as data:
            for i in range(14):
                self.register_buffer(f"v{i}", torch.from_numpy(data[f"v{i}"].copy()))
            golden_x = torch.from_numpy(data["golden_input"].copy())
            golden_y = torch.from_numpy(data["golden_mean"].copy())
        with torch.no_grad():
            prediction = self.forward(golden_x)
            # Float32 TF/PyTorch reduction ordering differs in this deep 512-wide
            # network. Retain and report the actual max error; 2e-4 is <0.012 deg
            # for a direct angular output, before input-range clipping.
            torch.testing.assert_close(prediction, golden_y, rtol=2e-5, atol=2e-4)
            self.golden_max_error = float((prediction - golden_y).abs().max())

    def forward(self, x):
        x = x @ self.v1 + self.v0
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).square().mean(-1, keepdim=True)
        inverse = (variance + self.manifest["layer_norm_epsilon"]).rsqrt() * self.v3
        x = (x * inverse + (self.v2 - mean * inverse)).tanh()
        for i in (4, 6, 8):
            x = F.elu(x @ getattr(self, f"v{i + 1}") + getattr(self, f"v{i}"))
        return x @ self.v11 + self.v10

    @torch.no_grad()
    def act(self, observation):
        values = [
            np.asarray(observation[k]).reshape(-1) for k in self.manifest["observation_shapes"]
        ]
        x = torch.from_numpy(np.concatenate(values).astype(np.float32))[None]
        return self(x)[0].cpu().numpy()


class TeacherOracle:
    """Teacher observes the complete body's measured state through its old schema.

    The teacher alone sees a desired future COM path. The student receives only
    current command/body feedback. This adapter does not step or support a ghost.
    """

    def __init__(self, environment, path):
        from flybody.fly_envs import walk_imitation

        self.environment = environment
        self.policy = WalkingTeacher(path).eval()
        # Instantiate once to recover exact official observation/action ordering.
        template = walk_imitation()
        template.reset()
        fly = template.task.walker
        model = environment.model
        self.joint_ids = [model.joint("walker/" + j.name).id for j in fly.observable_joints]
        self.activation_ids = [
            model.actuator_actadr[model.actuator("walker/" + a.name).id] for a in fly.actuators
        ]
        self.actuator_ids = np.array(
            [model.actuator("walker/" + n).id for n in template.action_spec().name.split("\t")]
        )
        self.appendage_ids = [model.site("walker/" + s.name).id for s in fly.appendages]
        self.sensors = {}
        for kind in ("accelerometer", "force", "gyro", "touch", "velocimeter"):
            indices = []
            for sensor in getattr(fly.mjcf_model.sensor, kind):
                sid = model.sensor("walker/" + sensor.name).id
                start = model.sensor_adr[sid]
                indices.extend(range(start, start + model.sensor_dim[sid]))
            self.sensors[kind] = np.array(indices)

    def set_reference(self, speed, yaw_speed, duration, heading=0.0):
        from flybody.tasks.synthetic_trajectories import constant_speed_trajectory

        self.reference, _ = constant_speed_trajectory(
            n_steps=int(duration / 0.002) + 100,
            speed=speed,
            yaw_speed=yaw_speed,
            init_heading=heading,
            init_pos=(0, 0, 0.1278),
            control_timestep=0.002,
        )

    def observation(self, step):
        from flybody.quaternions import get_dquat_local

        env = self.environment
        model, data = env.model, env.data
        rotation = data.xmat[env.thorax_id].reshape(3, 3)
        pose = data.qpos[:7]
        future = self.reference[step : step + 65]
        observation = {
            "walker/" + k: env.mean_sensors[idx].copy() for k, idx in self.sensors.items()
        }
        observation.update(
            {
                "walker/actuator_activation": data.act[self.activation_ids].copy(),
                "walker/appendages_pos": (
                    (data.site_xpos[self.appendage_ids] - data.xpos[env.thorax_id]) @ rotation
                ).reshape(-1),
                "walker/joints_pos": data.qpos[model.jnt_qposadr[self.joint_ids]].copy(),
                "walker/joints_vel": data.qvel[model.jnt_dofadr[self.joint_ids]].copy(),
                "walker/world_zaxis": rotation[2].copy(),
                "walker/ref_displacement": (future[:, :3] - pose[:3]) @ rotation,
                "walker/ref_root_quat": get_dquat_local(pose[3:], future[:, 3:]),
            }
        )
        return observation

    def act(self, step):
        env = self.environment
        raw_control = np.zeros(env.model.nu, np.float32)
        raw_control[self.actuator_ids] = self.policy.act(self.observation(step))
        raw_control = np.clip(raw_control, env.low, env.high)
        return 2 * (raw_control - env.low) / (env.high - env.low) - 1
