"""Static geometry gates; no motion or task success is inferred here."""

import mujoco

from sixlegs.neural_insertion.scene import load_scene
from sixlegs.neural_insertion.validation import inspect_scene


def test_static_approach_envelope_and_force_limits(tmp_path):
    report = inspect_scene(tmp_path / "scene.xml")
    assert report["success"], report
    assert report["sampled_poses"] == 120
    assert report["sampled_minimum_needle_surface_gap_mm"] > 4
    assert report["z_actuator_limit_n"] > report["z_supported_weight_n"]
    assert report["simulated_seconds"] == 0


def test_needle_cannot_pass_through_solid_phantom(tmp_path):
    model, data = load_scene(tmp_path / "scene.xml")
    data.joint("insertion").qpos[0] = .018  # Scratch collision probe, not a live move.
    mujoco.mj_forward(model, data)
    needle, tissue = model.geom("needle").id, model.geom("tissue_phantom").id
    contacts = [c for c in data.contact if {c.geom1, c.geom2} == {needle, tissue}]
    assert contacts and min(c.dist for c in contacts) < 0
    assert data.time == 0
