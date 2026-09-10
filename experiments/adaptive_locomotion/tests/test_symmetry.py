import numpy as np
import torch
from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.limb_loss import LOSS_BODIES
from adaptive_locomotion.symmetry import mirror_action, mirror_observation


def test_reflection_is_involution_and_preserves_missing_joint_mask():
    obs = torch.randn(8, 66)
    obs[:, 45:57] = 1
    obs[:, 48:51] = 0  # FR absent -> FL absent in the reflected body.
    reflected = mirror_observation(obs)
    torch.testing.assert_close(mirror_observation(reflected), obs)
    assert not reflected[:, 45:48].any()
    assert reflected[:, 48:51].all()
    action = torch.randn(8, 12)
    torch.testing.assert_close(mirror_action(mirror_action(action)), action)


def test_reflection_matches_mujoco_sensors_on_the_opposite_physical_body():
    for kind in ("lower", "whole"):
        bodies = [
            next(b for b in LOSS_BODIES if b.name == f"{kind}_{leg}")
            for leg in ("fl", "fr")
        ]
        envs = [
            DogEnv(
                1, bodies=[b], randomize=False, faults=False, threads=1, sensing=True
            )
            for b in bodies
        ]
        try:
            left, right = envs
            lg, rg = left.groups[0], right.groups[0]
            lg.qpos[:, 3:7] = np.array([0.96, 0.1, 0.2, -0.05]) / np.linalg.norm(
                [0.96, 0.1, 0.2, -0.05]
            )
            lg.qvel[:, :6] = [0.1, -0.2, 0.3, 0.2, 0.3, -0.4]
            lg.batch.forward()
            left.refresh()
            rg.qpos[:, :7] = lg.qpos[:, :7] * [1, -1, 1, 1, -1, 1, -1]
            rg.qvel[:, :6] = lg.qvel[:, :6] * [1, -1, 1, -1, 1, -1]
            rg.qpos[:, rg.qadr] = mirror_action(torch.tensor(left.q)).numpy()[
                :, rg.slot
            ]
            rg.batch.forward()
            right.refresh()
            expected = mirror_observation(torch.tensor(left.obs())).numpy()
            np.testing.assert_allclose(right.obs()[:, :57], expected[:, :57], atol=1e-6)
            # Vendor visual meshes are not exactly bilateral; ray hits can
            # differ by sub-millimetres. This is a soft policy prior.
            np.testing.assert_allclose(right.obs()[:, 57:], expected[:, 57:], atol=5e-4)
        finally:
            for env in envs:
                env.close()
