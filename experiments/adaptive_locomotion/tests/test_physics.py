import mujoco
import numpy as np
import pytest

from adaptive_locomotion.bodies import (
    LIMITS,
    PRESETS,
    STAND,
    allowed_support_names,
    build_model,
    initialize,
)
from adaptive_locomotion.env import OBS_DIM, DogEnv


def test_physical_partial_and_absent_calf():
    healthy = build_model(PRESETS["healthy"], sensing=False)
    short = build_model(PRESETS["short_fl"], sensing=False)
    absent = build_model(PRESETS["missing_fl"], sensing=False)
    assert healthy.nu == short.nu == 12 and absent.nu == 11
    assert absent.nv == healthy.nv - 1
    assert absent.body_mass.sum() < short.body_mass.sum() < healthy.body_mass.sum()
    assert short.geom("FL_terminal").pos[2] > healthy.geom("FL_terminal").pos[2]
    assert np.all(short.body_inertia[1:] > 0)
    assert np.all(short.actuator_forcelimited)
    assert np.allclose(short.actuator_forcerange[:, 1], LIMITS)
    assert short.geom("FL_member").contype != 0


@pytest.mark.parametrize("body", ["healthy", "short_fl", "missing_fl"])
def test_scalar_batch_native_servo_agreement(body):
    env = DogEnv(1, bodies=[PRESETS[body]], randomize=False, faults=False, threads=1)
    group = env.groups[0]
    model = group.model
    data = mujoco.MjData(model)
    data.qpos[:] = group.qpos[0]
    data.qvel[:] = group.qvel[0]
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(42)
    for _ in range(30):
        action = rng.uniform(-0.3, 0.3, (1, 12))
        data.ctrl[:] = STAND[group.slot] + 0.65 * action[0, group.slot]
        for _ in range(10):
            mujoco.mj_step(model, data)
        env.step(action)
        np.testing.assert_allclose(group.qpos[0], data.qpos, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(group.qvel[0], data.qvel, atol=1e-9, rtol=1e-9)
    assert env.obs().shape == (1, OBS_DIM)
    env.close()


def test_foot_drop_does_not_bury_terminal_geometry():
    model = build_model(PRESETS["short_fl"], sensing=False)
    data = mujoco.MjData(model)
    initialize(model, data)
    data.qpos[2] += 0.06
    data.ctrl[:] = STAND
    worst = 0.0
    for _ in range(1000):
        mujoco.mj_step(model, data)
        if data.ncon:
            worst = min(worst, float(np.min(data.contact.dist)))
    assert worst > -0.008
    assert not np.any(data.warning.number)


def test_healthy_only_randomization_retains_full_motor_strength():
    env = DogEnv(
        16,
        bodies=[PRESETS["healthy"]],
        randomize=True,
        faults=False,
        randomize_strength=False,
        reward_profile="walk",
        threads=1,
    )
    for _ in range(5):
        env.reset(np.arange(env.n))
        np.testing.assert_array_equal(env.strength, np.ones((16, 12)))
        assert np.all(env.fault_at > 500)
    env.close()


def test_ground_support_sensor_catches_loaded_knee_and_allows_only_distal_stump():
    env = DogEnv(
        1, bodies=[PRESETS["healthy"]], randomize=False, faults=False, threads=1
    )
    g = env.groups[0]
    pose = STAND.copy()
    pose[1::3] = 1.3
    pose[2::3] = -2.7
    pose[10:12] = [0, -1.6]
    g.qpos[0, g.qadr] = pose
    g.qpos[0, 2] = 0.224  # knee loaded; the four terminals are clear
    g.batch.forward()
    knee = g.support_names.index("RR_housing")
    sensed = np.linalg.norm(g.support_data[0, knee, 1:4])
    assert sensed > 1 and not g.support_allowed[knee]
    data = mujoco.MjData(g.model)
    data.qpos[:] = g.qpos[0]
    data.qvel[:] = g.qvel[0]
    data.ctrl[:] = g.ctrl[0]
    mujoco.mj_forward(g.model, data)
    force = np.zeros(6)
    total = np.zeros(3)
    geom = g.model.geom("RR_housing").id
    for i, c in enumerate(data.contact):
        if geom in (c.geom1, c.geom2) and 0 in (
            g.model.geom_bodyid[c.geom1],
            g.model.geom_bodyid[c.geom2],
        ):
            mujoco.mj_contactForce(g.model, data, i, force)
            total += (1 if geom == c.geom2 else -1) * (
                c.frame.reshape(3, 3).T @ force[:3]
            )
    np.testing.assert_allclose(sensed, np.linalg.norm(total), rtol=1e-8, atol=1e-8)
    env.refresh()
    assert env.support_cost[0] > 0
    env.close()
    assert "FL_stump" in allowed_support_names(PRESETS["missing_fl"])
    assert "FR_housing" not in allowed_support_names(PRESETS["missing_fl"])
    assert "FL_terminal" in allowed_support_names(PRESETS["short_fl"])


def test_contact_instrumentation_does_not_change_dynamics():
    models = [build_model(support_sensing=x) for x in (False, True)]
    data = [mujoco.MjData(m) for m in models]
    for m, d in zip(models, data):
        initialize(m, d)
        d.ctrl[:] = STAND
        for _ in range(100):
            mujoco.mj_step(m, d)
    np.testing.assert_array_equal(data[0].qpos, data[1].qpos)
    np.testing.assert_array_equal(data[0].qvel, data[1].qvel)


def test_private_context_and_failure_event_do_not_leak_to_actor():
    env = DogEnv(
        2, bodies=[PRESETS["healthy"]], randomize=False, faults=False, threads=1
    )
    obs = env.obs().copy()
    history = env.history.copy()
    env.context[:] = 0.1
    np.testing.assert_array_equal(env.obs(), obs)
    np.testing.assert_array_equal(env.history, history)
    env.fault_at[0] = 0
    env.fault_strength[0] = 0.25
    env.fault_joint[0] = 0
    env.step(np.zeros((2, 12)))
    assert env.steps[0] == 1 and env.strength[0, 0] == 0.25
    # Fault onset neither resets the rollout nor its history.
    np.testing.assert_array_equal(env.history[0, :-1], history[0, 1:])
    env.close()


def test_zero_strength_removes_active_damping_and_caps_weak_servo():
    env = DogEnv(
        2, bodies=[PRESETS["healthy"]], randomize=False, faults=False, threads=1
    )
    g = env.groups[0]
    strengths = np.ones((2, 12))
    strengths[0, 0] = 0
    strengths[1, 0] = 0.1
    g.set_strength(np.arange(2), strengths)
    for _ in range(20):
        env.step(np.ones((2, 12)) * 3)
    assert abs(g.torque[0, 0]) < 1e-10
    assert abs(g.torque[1, 0]) <= 2.37 + 1e-10
    assert np.all(np.abs(g.torque) <= LIMITS + 1e-10)
    env.close()
