"""Physical scene contracts for the preview milestone; no task controller."""
import hashlib
import json

import mujoco
import numpy as np
import pytest

from sixlegs.scene import ROOT, START, load_scene

@pytest.fixture(scope="module")
def scene():
    return load_scene()


def test_vendor_integrity():
    manifest = json.loads((ROOT/"assets/menagerie/manifest.json").read_text())
    for path, digest in manifest["sha256"].items():
        assert hashlib.sha256((ROOT/"assets/menagerie"/path).read_bytes()).hexdigest() == digest


def test_structure_and_independent_grippers(scene):
    m, _ = scene
    assert m.nu == 34  # 18 leg joints + 14 arm joints + 2 tendon actuators
    assert m.neq == 6  # Two independent closed-linkage grippers
    assert m.jnt_type[m.joint("floating_base").id] == mujoco.mjtJoint.mjJNT_FREE
    for side in ("left", "right"):
        assert m.actuator(side+"_grip_fingers_actuator").trnid[0] == m.tendon(side+"_grip_split").id
        assert m.camera(side+"_wrist").bodyid == m.body(side+"_arm_bracelet_link").id
        for segment in ("front", "middle", "rear"):
            for joint in ("yaw", "hip", "knee"):
                assert m.joint(f"{side}_{segment}_{joint}").limited
    assert 50 < m.body_subtreemass[m.body("chassis").id] < 65
    assert m.body("mug").mass[0] == pytest.approx(.21)
    assert m.body("block").mass[0] == pytest.approx(.12)
    assert m.camera("head").bodyid == m.body("head").id
    assert np.all(m.actuator_forcelimited)
    assert np.all(m.actuator_forcerange[:,0] < 0)
    assert np.all(m.actuator_forcerange[:,1] > 0)


def test_initial_contacts_are_only_six_feet(scene):
    m, d = scene
    assert d.ncon == 6
    for contact in d.contact:
        names = [m.geom(int(g)).name for g in contact.geom]
        assert "floor" in names
        assert any(n.endswith("_foot") for n in names)
        assert contact.dist > -1e-6


def test_sampled_leg_swing_clearance():
    m,d = load_scene()
    rng = np.random.default_rng(7)
    legs = [f"{s}_{p}" for s in ("left","right") for p in ("front","middle","rear")]
    for _ in range(150):
        for leg in legs:
            for joint, span in (("yaw",.25),("hip",.15),("knee",.2)):
                d.joint(leg+"_"+joint).qpos[0] = rng.uniform(-span,span)
        mujoco.mj_forward(m,d)
        for c in d.contact:
            names = [m.geom(int(g)).name for g in c.geom]
            if "floor" not in names:
                assert c.dist >= -1e-5, (names, c.dist)


def test_five_second_physics_hold():
    m,d = load_scene()
    for _ in range(2500):
        mujoco.mj_step(m,d)
        assert np.isfinite(d.qpos).all()
        assert np.all(d.actuator_force <= m.actuator_forcerange[:,1] + 1e-6)
        assert np.all(d.actuator_force >= m.actuator_forcerange[:,0] - 1e-6)
    assert not d.warning.number.any()
    assert np.linalg.norm(d.qpos[:3]-START) < .03
    assert np.max(abs(d.qvel)) < .02
    assert d.body("mug").xpos[2] == pytest.approx(.84,abs=.002)
    assert d.body("block").xpos[2] == pytest.approx(.87,abs=.002)
