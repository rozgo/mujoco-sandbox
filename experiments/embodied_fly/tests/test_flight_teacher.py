from pathlib import Path

import numpy as np

from embodied_fly.body import FlyEnvironment
from embodied_fly.flight_teacher import FlightTeacherOracle
from embodied_fly.teacher import ConvertedTeacher

ASSETS = Path(__file__).resolve().parents[3] / "assets/embodied_fly/teachers"


def test_flight_teacher_matches_exported_tensorflow_reference():
    teacher = ConvertedTeacher(ASSETS / "flight.npz")
    assert teacher.golden_max_error < 2e-6
    assert len(teacher.manifest["variables"]) == 12


def test_flight_teacher_uses_complete_body_without_live_pose_writes():
    env = FlyEnvironment("flight")
    oracle = FlightTeacherOracle(env, ASSETS / "flight.npz", ASSETS / "wing_pattern_fmech.npy")
    oracle.initialize(0, 0.01)
    assert env.model.nu == 78 and env.model.nv == 108
    adhesion = [
        i
        for i in range(env.model.nu)
        if "adhere" in env.model.actuator(i).name or "adhesion" in env.model.actuator(i).name
    ]
    assert adhesion
    assert not oracle.retracted_controls[adhesion].any()
    for i in range(20):
        before = env.data.qpos.copy()
        action = oracle.act(i)
        np.testing.assert_array_equal(env.data.qpos, before)
        assert action.shape == (78,) and np.isfinite(action).all()
        assert np.max(np.abs(action)) <= 1
        env.step(action)
        assert not env.data.xfrc_applied.any() and not env.data.qfrc_applied.any()
    assert not env.data.warning.number.any()
