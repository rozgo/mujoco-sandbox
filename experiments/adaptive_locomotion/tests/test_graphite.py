"""Presentation must preserve physical state, sensor readings and model parameters."""

import copy

import mujoco
import numpy as np
import pytest

from adaptive_locomotion.bodies import (
    STAND,
    BodySpec,
    build_model,
    initialize,
    joint_mapping,
)
from adaptive_locomotion.graphite import configure, decorate


@pytest.mark.parametrize("terrain", ["flat", "moving"])
def test_theme_does_not_change_physics_or_sensors(terrain):
    plain = build_model(terrain=terrain)
    themed = copy.copy(plain)
    configure(themed)
    for field in (
        "body_mass",
        "body_inertia",
        "jnt_range",
        "geom_size",
        "geom_pos",
        "geom_quat",
        "geom_contype",
        "geom_conaffinity",
        "geom_friction",
        "actuator_forcerange",
        "actuator_gainprm",
        "actuator_biasprm",
    ):
        np.testing.assert_array_equal(getattr(plain, field), getattr(themed, field))
    a, b = mujoco.MjData(plain), mujoco.MjData(themed)
    initialize(plain, a)
    initialize(themed, b)
    slots, _, _, actuators = joint_mapping(plain)
    for _ in range(500):
        a.ctrl[actuators] = b.ctrl[actuators] = STAND[slots]
        mujoco.mj_step(plain, a)
        mujoco.mj_step(themed, b)
        np.testing.assert_array_equal(a.qpos, b.qpos)
        np.testing.assert_array_equal(a.qvel, b.qvel)
        np.testing.assert_array_equal(a.sensordata, b.sensordata)
    before = b.qpos.copy()
    scene = mujoco.MjvScene(themed, maxgeom=1000)
    decorate(scene, themed, b, BodySpec())
    assert scene.ngeom > 0
    np.testing.assert_array_equal(before, b.qpos)
    assert all(
        g.category == mujoco.mjtCatBit.mjCAT_DECOR for g in scene.geoms[: scene.ngeom]
    )
