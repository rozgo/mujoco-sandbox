"""Robot plant: hold, approach and the physics-only learning scene."""

import mujoco
import numpy as np
import pytest

from sixlegs.neural_insertion.motion import (JOINT_NAMES, evaluate, hold_trial, joints_for_tip, load_scene, run,
                                             start_ready, tour_reference)
from sixlegs.neural_insertion.rl_scene import build_rl_scene


def test_hold_has_no_drift_and_no_saturation():
    report, _ = hold_trial()
    assert report["passed"] and report["tip_drift_m"] < 1e-6
    assert max(report["max_force_fraction"].values()) < .5


def test_single_approach_meets_hover_gate_without_contact():
    model, data = start_ready(*load_scene())
    reference = tour_reference(model, data, [3])
    trace, events = run(model, data, reference)
    report = evaluate(trace, events, reference, [3])
    assert report["passed"]
    assert report["targets"][0]["max_lateral_error_m"] < 1e-5
    assert report["robot_environment_contacts"] == 0 and report["min_needle_phantom_gap_m"] > 5e-4


def test_inverse_kinematics_rejects_targets_outside_travel():
    model, data = load_scene()
    with pytest.raises(ValueError, match="outside travel"):
        joints_for_tip(model, data.site("needle_tip").xpos+(0, 0, -.03))


def test_physics_only_scene_reproduces_the_full_tour_exactly():
    path, meta = build_rl_scene()
    rl = mujoco.MjModel.from_xml_path(str(path))
    rd = mujoco.MjData(rl)
    full, fd = start_ready(*load_scene())
    mujoco.mj_resetDataKeyframe(rl, rd, rl.key("inspection").id)
    start_ready(rl, rd)
    reference = tour_reference(full, fd, [0, 4])
    a, _ = run(full, fd, reference)
    b, _ = run(rl, rd, reference)
    np.testing.assert_array_equal(a["qpos"], b["qpos"])
    assert meta["removed_marking_geoms"] > 800 and rl.ngeom < 100
    assert rl.numeric("surface_dome").data[0] == pytest.approx(.112)
