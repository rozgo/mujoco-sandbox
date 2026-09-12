"""Backend safety/equivalence checks, optional physical checks on CUDA hosts."""

import mujoco
import numpy as np
import pytest

from adaptive_locomotion.bodies import LIMITS
from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.evaluate import CASES
from adaptive_locomotion.limb_loss import LOSS_CASES


def test_unknown_backend_rejected():
    with pytest.raises(ValueError, match="physics backend"):
        DogEnv(num_envs=1, physics_backend="unknown")


def test_concurrent_topologies_match_serial_with_partial_resets(cuda_warp):
    args = {
        "num_envs": 32,
        "seed": 9231,
        "limb_stage": "consolidate",
        "physics_backend": "warp",
    }
    serial = DogEnv(**args, warp_execution="serial")
    parallel = DogEnv(**args, warp_execution="concurrent")
    rng = np.random.default_rng(9232)
    try:
        for step in range(30):
            # Compare local integration from identical fresh states. Contact
            # accumulation on GPU is not bitwise deterministic; tiny differences
            # can grow during a long contact-rich rollout even on one stream.
            serial.reset(np.arange(32))
            parallel.reset(np.arange(32))
            action = rng.normal(0, 0.08, (32, 12)).astype(np.float32)
            a = serial.step(action)
            b = parallel.step(action)
            np.testing.assert_allclose(parallel.obs(), serial.obs(), atol=2e-5, rtol=0)
            np.testing.assert_allclose(a[0], b[0], atol=1e-4, rtol=1e-4)
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_allclose(
                parallel.torque, serial.torque, atol=2e-4, rtol=0
            )
            if step in (9, 19):
                ids = np.array([0, 9, 13, 19, 25, 31])
                before = [g.qpos.copy() for g in parallel.groups]
                serial.reset(ids)
                parallel.reset(ids)
                for g, s, old in zip(
                    parallel.groups, parallel.slices, before, strict=True
                ):
                    keep = ~np.isin(np.arange(s.start, s.stop), ids)
                    np.testing.assert_array_equal(g.qpos[keep], old[keep])
    finally:
        serial.close()
        parallel.close()


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


@pytest.mark.parametrize("case", LOSS_CASES)
def test_warp_bvh_ranges_match_cpu(cuda_warp, case):
    kwargs = {
        "num_envs": 8,
        "bodies": [CASES[case][0]],
        "randomize": False,
        "faults": False,
        "sensing": True,
    }
    cpu, gpu = DogEnv(**kwargs), DogEnv(**kwargs, physics_backend="warp")
    try:
        c, g = cpu.groups[0], gpu.groups[0]
        for seed in (9222, 9223, 9224):
            rng = np.random.default_rng(seed)
            c.qpos[:] = c.initial_qpos
            c.qpos[:, :2] = rng.uniform(-10, 10, (8, 2))
            c.qpos[:, 2] = rng.uniform(0.3, 0.8, 8)
            c.qpos[:, c.qadr] += rng.uniform(-0.3, 0.3, (8, len(c.slot)))
            for q in c.qpos:
                euler = rng.uniform([-0.4, -0.4, -3.14], [0.4, 0.4, 3.14])
                mujoco.mju_euler2Quat(q[3:7], euler, "xyz")
            c.batch.forward()
            g.qpos[:] = c.qpos
            g.batch.forward()
            np.testing.assert_allclose(
                np.concatenate(g.rays, 1), np.concatenate(c.rays, 1), atol=1e-4, rtol=0
            )
    finally:
        cpu.close()
        gpu.close()
