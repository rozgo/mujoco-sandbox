"""Backend safety/equivalence checks, optional physical checks on CUDA hosts."""

import numpy as np
import pytest

from adaptive_locomotion.bodies import LIMITS
from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.evaluate import CASES
from adaptive_locomotion.limb_loss import LOSS_CASES


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="physics backend"):
        DogEnv(num_envs=1, physics_backend="unknown")


@pytest.fixture(scope="module")
def cuda_warp():
    wp = pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("NVIDIA CUDA required for optional Warp physics tests")
    return wp


@pytest.mark.parametrize("case", LOSS_CASES)
def test_warp_physics_and_reset(cuda_warp, case):
    kwargs = {
        "num_envs": 2,
        "bodies": [CASES[case][0]],
        "randomize": False,
        "faults": False,
        "sensing": True,
        "timestep": 0.002,
    }
    cpu = DogEnv(**kwargs)
    gpu = DogEnv(**kwargs, physics_backend="warp")
    try:
        c, g = cpu.groups[0], gpu.groups[0]
        np.testing.assert_allclose(g.qpos, c.qpos, atol=1e-6)
        before = g.qpos.copy()
        g.batch.graph(10)
        g.batch.download()
        np.testing.assert_array_equal(g.qpos, before)
        for _ in range(10):
            c.batch.step(nstep=1)
            g.batch.step(nstep=1)
        np.testing.assert_allclose(g.qpos, c.qpos, atol=0.002, rtol=0)
        np.testing.assert_allclose(g.qvel, c.qvel, atol=0.1, rtol=0)
        for _ in range(100):
            gpu.step(np.zeros((2, 12)))
        assert np.linalg.norm(g.support_data[:, :, 1:4]) > 1
        assert np.isfinite(gpu.obs()).all()
        assert np.all(np.abs(gpu.torque) <= LIMITS + 1e-4)
        before = g.qpos[1].copy()
        velocity = g.qvel[1].copy()
        gpu.reset([0])
        np.testing.assert_array_equal(g.qpos[1], before)
        np.testing.assert_array_equal(g.qvel[1], velocity)
        np.testing.assert_allclose(g.qpos[0], g.initial_qpos, atol=1e-6)
        strength = np.ones((1, 12))
        strength[0, g.slot[0]] = 0
        g.set_strength(np.array([0]), strength)
        g.ctrl[:, 0] += 1
        g.batch.step(nstep=1)
        assert abs(g.torque[0, 0]) < 1e-6
        assert abs(g.torque[1, 0]) > 1e-3
    finally:
        cpu.close()
        gpu.close()
