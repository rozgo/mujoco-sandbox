"""Physical support motion, frame invariance and unchanged static controls."""

import mujoco
import numpy as np
import pytest
import torch
from adaptive_locomotion.bodies import PRESETS
from adaptive_locomotion.moving_env import MovingEnv, point_velocity, to_local
from adaptive_locomotion.standing_env import StandingEnv


def test_rigid_carrier_velocity_includes_rotation_at_the_point():
    point = np.array([[2.0, 0, 1]])
    origin = np.array([[1.0, 0, 1]])
    linear = np.array([[0.3, -0.2, 0.1]])
    angular = np.array([[0.0, 0, 2]])
    np.testing.assert_allclose(
        point_velocity(point, origin, linear, angular), [[0.3, 1.8, 0.1]]
    )
    angle = 0.7
    rot = np.array(
        [
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        ]
    )
    local = np.array([[0.2, -0.1, 0.3]])
    world = origin + np.einsum("nij,nj->ni", rot, local)
    np.testing.assert_allclose(to_local(world, origin, rot), local, atol=1e-15)


def test_static_worlds_keep_exact_observations_actions_rewards():
    kwargs = {
        "num_envs": 4,
        "seed": 21,
        "cases": [(PRESETS["healthy"], "flat")],
        "schedule": False,
        "threads": 1,
    }
    a, b = StandingEnv(**kwargs), MovingEnv(**kwargs)
    try:
        # Initialization layers consume different random draws; explicitly
        # match reset randomness before comparing the unchanged static path.
        for env in (a, b):
            env.rng = np.random.default_rng(99)
            env.reset(np.arange(4))
        rng = np.random.default_rng(4)
        for _ in range(20):
            np.testing.assert_array_equal(a.obs(), b.obs())
            np.testing.assert_array_equal(a.support_obs(), b.support_obs())
            action = rng.normal(0, 0.05, (4, 12))
            ra, da, fa, _ = a.step(action)
            rb, db, fb, _ = b.step(action)
            np.testing.assert_array_equal(ra, rb)
            np.testing.assert_array_equal(da, db)
            np.testing.assert_array_equal(fa, fb)
    finally:
        a.close()
        b.close()


def test_platform_contacts_and_robot_torque_limits_survive_motor_addition():
    env = MovingEnv(
        2, moving_profile="healthy", randomize=False, schedule=False, motion="still"
    )
    try:
        g = env.groups[0]
        assert g.model.nu == 18
        assert g.model.joint("floating_base").type[0] == mujoco.mjtJoint.mjJNT_FREE
        limits = g.force_range[:, g.deck_aadr].copy()
        g.set_strength(np.array([0, 1]), np.full((2, 12), 0.5))
        np.testing.assert_array_equal(g.force_range[:, g.deck_aadr], limits)
        np.testing.assert_allclose(
            g.force_range[:, g.aadr, 1],
            np.tile([23.7, 23.7, 45.43], 4)[None, :].repeat(2, 0) * 0.5,
        )
        for _ in range(40):
            env.step(np.zeros((2, 12)))
        assert (env.tip_forces.sum(1) > 90).all()
        assert (env.bad_support_force < 1).all()
        assert (env.deck_origin[:, 2] < 0.50).all()  # finite loaded servo sag
        assert (env.deck_origin[:, 2] > 0.47).all()
        assert g.robot_mass < g.model.body_mass.sum() - 34
    finally:
        env.close()


@pytest.mark.parametrize("backend", ["mjbatch", "warp"])
def test_physical_platform_motion_and_measured_pose(backend):
    if backend == "warp" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    env = MovingEnv(
        2,
        moving_profile="healthy",
        randomize=False,
        schedule=False,
        motion="translate",
        physics_backend=backend,
        substep_support=True,
    )
    try:
        g = env.groups[0]
        initial = g.qpos.copy()
        env.step(np.zeros((2, 12)))
        assert not np.array_equal(initial, g.qpos)
        for _ in range(80):
            env.step(np.zeros((2, 12)))
        assert np.linalg.norm(env.deck_origin[0, :2]) > 0.05
        data = mujoco.MjData(g.model)
        data.qpos[:], data.qvel[:] = g.qpos[0], g.qvel[0]
        mujoco.mj_forward(g.model, data)
        # Sensor pipeline uses its documented pre-integration frame state; one
        # 2 ms step of tolerance, never the scripted target as measured pose.
        np.testing.assert_allclose(
            env.deck_origin[0], data.site("deck_top").xpos, atol=0.003
        )
        assert np.isfinite(env.support_obs()).all()
        assert (
            np.abs(g.torque[:, g.deck_aadr])
            <= g.model.actuator_forcerange[g.deck_aadr, 1] + 1e-3
        ).all()
        assert g.support_peaks.shape[1] == len(g.support_names)
    finally:
        env.close()
