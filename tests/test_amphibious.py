"""Contact, displacement and complete land-water-land behavior."""

import mujoco
import numpy as np
import pytest

from sixlegs.amphibious.control import SWIM_POSE
from sixlegs.amphibious.fluid import Water
from sixlegs.amphibious.scene import LEGS, bed_height, load_scene
from sixlegs.amphibious.simulation import Simulation, run


def test_surface_map_matches_physical_ramps_and_start_contacts():
    sim = Simulation()
    m, d = sim.model, sim.data
    assert sim.control.contacts(d).all()
    assert sim.control.mass == pytest.approx(17.006408)
    assert m.nu == 12
    group = np.ones(6, dtype=np.uint8)
    group[1] = 0  # Decorative transparent water surface.
    geom = np.zeros(1, dtype=np.int32)
    for x in np.linspace(-3.0, 4.0, 60):
        distance = mujoco.mj_ray(
            m,
            d,
            np.array([x, 0.9, 1.0]),
            np.array([0.0, 0.0, -1.0]),
            group,
            1,
            -1,
            geom,
        )
        assert 1.0 - distance == pytest.approx(bed_height(x), abs=1e-6)
    before = d.qpos.copy()
    sim.control.update(d, sim.water, 0.01)
    sim.water.apply(d, (6.0, 6.0))
    assert np.array_equal(before, d.qpos)
    assert not d.qfrc_applied.any()  # No hidden force moves the dry chassis.


def test_forward_float_posture_respects_limits_and_displacement():
    m, d = load_scene()
    d.qpos[:3] = [0.8, 0, 1.0]
    d.qpos[7:] = SWIM_POSE
    mujoco.mj_forward(m, d)
    assert np.all(SWIM_POSE >= m.jnt_range[1:, 0])
    assert np.all(SWIM_POSE <= m.jnt_range[1:, 1])
    for leg in LEGS[:2]:
        direction = -d.body(leg + "_float").xmat.reshape(3, 3)[:, 2]
        assert direction[0] > 0.99
        assert abs(direction[2]) < 0.08
    water = Water(m)
    water.apply(d)
    assert not water.buoyancy.any()
    d.qpos[2] = -1.0
    d.qvel[:6] = [0.3, -0.2, 0.1, 0.1, 0.2, 0.3]
    mujoco.mj_forward(m, d)
    water.apply(d)
    assert np.allclose(water.submerged, 1.0)
    expected = 1000 * 9.81 * sum(volume for _, _, volume in water.parts)
    assert sum(water.buoyancy) == pytest.approx(expected)
    assert water.drag_power < 0.0


def test_buoyancy_holds_weight_and_disabling_it_sinks():
    results = []
    for scale in (1.0, 0.0):
        m, d = load_scene()
        d.qpos[:3] = [0.8, 0, -0.13]
        d.qpos[7:] = SWIM_POSE
        mujoco.mj_forward(m, d)
        water = Water(m, scale)
        for _ in range(7000):
            d.ctrl[:] = np.clip(
                120 * (SWIM_POSE - d.qpos[7:]) - 4 * d.qvel[6:] + d.qfrc_bias[6:],
                m.actuator_ctrlrange[:, 0],
                m.actuator_ctrlrange[:, 1],
            )
            water.apply(d)
            mujoco.mj_step(m, d)
        results.append((d.qpos[2], sum(water.buoyancy), d.ncon))
    assert results[0][1] == pytest.approx(17.006408 * 9.81, rel=0.02)
    assert results[0][2] == 0
    assert results[1][0] < results[0][0] - 0.04
    assert results[1][1] == 0.0
    assert results[1][2] > 0


def test_complete_crossing(tmp_path):
    result = run(tmp_path)
    assert result["success"], result
    assert [e["phase"] for e in result["phases"]] == [
        "APPROACH",
        "WATER ENTRY",
        "FLOATING",
        "EXIT",
        "COMPLETE",
    ]
    assert result["longest_unsupported_float_s"] > 2.0
    assert all(result["final_foot_contacts"])
    assert result["max_actuator_fraction"] <= 1.0 + 1e-12
    assert result["min_body_up"] > 0.9
    assert result["max_ground_penetration_m"] < 0.01
    assert not any(result["warnings"])
