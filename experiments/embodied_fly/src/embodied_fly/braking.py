"""Vectorized inherited teacher labels for braking, using only current body state.

The standing reference is anchored at each world's current xy and heading. This
adapter writes no physical state and is not part of the deployed graph actor.
"""

import numpy as np
import torch

from embodied_fly.teacher import TeacherOracle


class BrakingTeacher:
    def __init__(self, environment, path, device="cpu"):
        self.env = environment
        self.oracle = TeacherOracle(environment.template, path)
        self.policy = self.oracle.policy.to(device).eval()
        self.device = torch.device(device)
        self.xpos = environment.batch.bind("xpos")
        self.sites = environment.batch.bind("site_xpos")
        environment.batch.forward()
        self.low = torch.as_tensor(
            environment.template.low, dtype=torch.float32, device=device
        )
        self.high = torch.as_tensor(
            environment.template.high, dtype=torch.float32, device=device
        )
        self.actuators = torch.as_tensor(self.oracle.actuator_ids, device=device)

    def observation(self):
        from flybody.quaternions import get_dquat_local

        env, oracle = self.env, self.oracle
        model = env.model
        pose = env.fields["qpos"][:, :7]
        rotation = env.fields["xmat"][:, env.template.thorax_id].reshape(-1, 3, 3)
        heading = np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0])
        upright = np.zeros((env.n, 4))
        upright[:, 0], upright[:, 3] = np.cos(heading / 2), np.sin(heading / 2)
        relative = get_dquat_local(pose[:, 3:], upright)
        delta = np.zeros((env.n, 3))
        delta[:, 2] = 0.1278 - pose[:, 2]
        local_delta = np.einsum("ni,nij->nj", delta, rotation)
        appendages = (
            self.sites[:, oracle.appendage_ids] - self.xpos[:, env.template.thorax_id, None]
        )
        observation = {
            "walker/" + k: env.mean_sensors[:, indices]
            for k, indices in oracle.sensors.items()
        }
        observation.update(
            {
                "walker/actuator_activation": env.fields["act"][:, oracle.activation_ids],
                "walker/appendages_pos": np.einsum(
                    "nki,nij->nkj", appendages, rotation
                ).reshape(env.n, -1),
                "walker/joints_pos": env.fields["qpos"][
                    :, model.jnt_qposadr[oracle.joint_ids]
                ],
                "walker/joints_vel": env.fields["qvel"][:, model.jnt_dofadr[oracle.joint_ids]],
                "walker/world_zaxis": rotation[:, 2],
                "walker/ref_displacement": np.repeat(local_delta[:, None], 65, axis=1),
                "walker/ref_root_quat": np.repeat(relative[:, None], 65, axis=1),
            }
        )
        return np.concatenate(
            [
                observation[key].reshape(env.n, -1)
                for key in self.policy.manifest["observation_shapes"]
            ],
            axis=1,
        ).astype(np.float32)

    @torch.no_grad()
    def act(self):
        predicted = self.policy(torch.as_tensor(self.observation(), device=self.device))
        controls = torch.zeros((self.env.n, self.env.model.nu), device=self.device)
        controls[:, self.actuators] = predicted
        controls = controls.clamp(self.low, self.high)
        return 2 * (controls - self.low) / (self.high - self.low) - 1
