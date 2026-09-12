import numpy as np
import pytest
from fly_survival.motor import Motors
from fly_survival.scene import TERRAIN_BIT, build


def test_static_scene_has_real_separate_bodies_and_limited_mechanism():
    a = build(2)
    m, d = a.sim.mj_model, a.sim.mj_data
    assert d.time == 0
    assert np.isclose(m.body("swatter").mass[0], 0.040)
    np.testing.assert_allclose(m.actuator_forcerange[a.swat_actuator], [-12000, 12000])
    assert m.actuator_forcelimited[a.swat_actuator]
    assert len(set(a.fly_body_ids)) == 2
    for bid in a.fly_body_ids:
        assert 0.0008 < m.body_subtreemass[bid] < 0.0013
    assert not d.warning.number.any()


def test_interfly_and_hazard_collision_masks_are_enabled():
    a = build(2)
    m = a.sim.mj_model
    groups = []
    for fly in a.flies:
        groups.append(
            [
                i
                for i in range(m.ngeom)
                if m.geom(i).name.startswith(fly.name + "/") and m.geom_contype[i]
            ]
        )
    g1, g2 = groups[0][0], groups[1][0]
    assert m.geom_contype[g1] & m.geom_conaffinity[g2]
    assert m.geom_contype[g2] & m.geom_conaffinity[g1]
    assert m.geom_conaffinity[g1] & TERRAIN_BIT
    assert m.geom_contype[m.geom("swatter_pad").id] & m.geom_conaffinity[g1]


def test_walking_comes_from_bounded_actuation():
    a = build(1)
    motor = Motors(a)
    m, d = a.sim.mj_model, a.sim.mj_data
    initial = d.xpos[a.fly_body_ids[0]].copy()
    for _ in range(300):
        motor.step(np.ones((1, 2)))
        limited = m.actuator_forcelimited.astype(bool)
        assert (
            d.actuator_force[limited] <= m.actuator_forcerange[limited, 1] + 1e-6
        ).all()
        assert (
            d.actuator_force[limited] >= m.actuator_forcerange[limited, 0] - 1e-6
        ).all()
    assert np.linalg.norm(d.xpos[a.fly_body_ids[0], :2] - initial[:2]) > 1.0
    assert d.xpos[a.fly_body_ids[0], 2] > 0.5
    assert np.isfinite(d.qpos).all()
    assert not d.warning.number.any()


def test_death_removes_active_torque_without_removing_the_body():
    a = build(1)
    motor = Motors(a)
    mass = a.sim.mj_model.body_mass.copy()
    motor.step(np.ones((1, 2)), alive=np.array([False]))
    np.testing.assert_array_equal(a.sim.mj_model.body_mass, mass)
    ids = motor.dead_actuators[0]
    np.testing.assert_allclose(a.sim.mj_data.actuator_force[ids], 0, atol=1e-9)


def test_invalid_agent_count_is_rejected():
    with pytest.raises(ValueError):
        build(9)
