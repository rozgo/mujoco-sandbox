from types import SimpleNamespace

import numpy as np
import torch

from embodied_fly.batch import FlyBatch
from embodied_fly.ground_posture import GroundPosture
from embodied_fly.wing_response_learning import response_examples, response_loss


def test_paired_labels_teach_restoring_angle_and_damping_speed_without_pose_writes():
    env = FlyBatch(3, 3, 14, preset="wing_motion")
    posture = GroundPosture(env, env.fields["qpos"][0])
    before = env.fields["qpos"].copy()
    obs, target, scale = response_examples(
        env,
        posture,
        env.observation(),
        np.array([0, 1]),
        np.array([2, 4]),
        np.array([False, True]),
    )
    expected = np.zeros((2, 6))
    expected[0, 2] = expected[1, 4] = -1
    np.testing.assert_allclose(
        (target[:, 0] - target[:, 1]) / scale[:, None], expected, atol=1e-6
    )
    np.testing.assert_array_equal(before, env.fields["qpos"])
    np.testing.assert_array_equal(obs[:, 0, 297:375], env.observation()[:2, 297:375])


def test_response_loss_drives_an_underresponsive_controller_toward_correct_gain():
    env = FlyBatch(3, 3, 14, preset="wing_motion")
    posture = GroundPosture(env, env.fields["qpos"][0])

    class KnownResponse(torch.nn.Module):
        def __init__(self, gain):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(gain))

        def forward(self, observation, state):
            q = observation[:, 389:395] * np.pi
            v = observation[:, 383:389] * 2000
            rest = torch.tensor(posture.qref[env.template.wing_angle_indices], dtype=q.dtype)
            wing = self.gain * (0.02 * (rest - q) - 0.00015 * v) / 0.03
            action = observation.new_zeros((len(q), 78)).index_copy(
                1, torch.tensor(posture.wings), wing
            )
            return SimpleNamespace(action=action)

    bad = KnownResponse(0.1)
    memory = torch.zeros(6, 3)
    loss = response_loss(
        bad, memory, env, posture, env.observation(), np.arange(3), 2, np.random.default_rng(3)
    )
    loss.backward()
    assert bad.gain.grad < 0
    good = KnownResponse(1.0)
    ideal = response_loss(
        good,
        memory,
        env,
        posture,
        env.observation(),
        np.arange(3),
        2,
        np.random.default_rng(3),
    )
    assert ideal < 1e-10 and loss > 0.1
