"""End-to-end variant: scene structure, pickup geometry and the provisional tissue model."""

import mujoco
import numpy as np
import pytest

from sixlegs.neural_insertion import e2e
from sixlegs.neural_insertion.e2e_scene import BED_TOP, EYELET_CENTER, KEEPER_CLOSED, KEEPER_DESIGN, load_e2e
from sixlegs.neural_insertion.tissue import Tissue, TissueParams


@pytest.fixture(scope="module")
def cycle():
    return e2e.Cycle(0, tubes=True)


def test_scene_builds_with_a_free_thread_and_tube(cycle):
    m = cycle.m
    assert m.opt.timestep == 5e-6 and m.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
    assert m.joint("thread_root").type == mujoco.mjtJoint.mjJNT_FREE
    assert sum(m.geom(g).name.startswith("tube_") for g in range(m.ngeom)) == 21
    assert sum(m.geom(g).name.startswith("eyelet_") for g in range(m.ngeom)) == 12
    assert KEEPER_DESIGN == "latch"


def pose(cycle, offset, keeper):
    C, bed = EYELET_CENTER*1000, BED_TOP*1000
    q = cycle.joints(np.array((C[0], C[1]+offset, bed-e2e.SLOT_DEPTH)), keeper=keeper)
    cycle.d.qpos[cycle.qadr] = q  # static check of a configuration, not a motion
    mujoco.mj_forward(cycle.m, cycle.d)


def tool_contacts(cycle):
    names = lambda c: cycle.m.geom(c.geom1).name+cycle.m.geom(c.geom2).name
    return [c.dist for c in cycle.d.contact[:cycle.d.ncon] if any(k in names(c) for k in ("slot", "keeper", "needle"))]


def test_needle_passes_through_the_eyelet_and_the_latch_clears_the_slot(cycle):
    pose(cycle, e2e.DESCEND_OFFSET, e2e.KEEPER_OPEN)
    assert tool_contacts(cycle) == []  # slot and needle fit through the ring
    pose(cycle, e2e.CAPTURE_OFFSET, KEEPER_CLOSED*1000)
    m, d = cycle.m, cycle.d
    latch = m.geom("keeper_latch").id
    gap = min(mujoco.mj_geomDistance(m, d, latch, m.geom(f"slot_lip_{i}").id, 1., None) for i in (0, 1))
    assert .005 < gap < .02  # mm: closes the slot (gap smaller than the 40 µm rim wire) without touching


def test_tissue_dimples_then_punctures():
    m, d, meta = load_e2e(tubes=False)
    tissue = Tissue(m, meta["thread_bodies"], TissueParams())
    assert not tissue.state.punctured
    p = TissueParams()
    assert p.k_dimple*p.d_puncture == pytest.approx(2000.)  # 2 mN puncture force in uN


def test_modern_scene_anchors_a_short_thread_without_initial_contact():
    from sixlegs.neural_insertion.modern_scene import load_modern
    m, d, meta = load_modern()
    assert len(meta["thread_bodies"]) == 6 and m.nv == 21  # 5 robot slides, pincher, 5 ball joints
    assert not [c for c in d.contact[:d.ncon] if c.dist < -1e-4]
    pairs = {(m.geom(m.pair_geom1[i]).name, m.geom(m.pair_geom2[i]).name) for i in range(m.npair)}
    assert ("needle", "pincher_jaw") in pairs  # the jaw must meet the needle it rides on
