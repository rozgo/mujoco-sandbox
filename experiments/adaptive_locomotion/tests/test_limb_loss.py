import mujoco
import numpy as np

from adaptive_locomotion.bodies import (
    LEGS,
    PRESETS,
    allowed_support_names,
    build_model,
    initialize,
)
from adaptive_locomotion.env import DogEnv
from adaptive_locomotion.limb_loss import LOWER_BODIES, WHOLE_BODIES


def test_removal_changes_topology_mass_and_only_real_support_surfaces():
    healthy = build_model(PRESETS["healthy"])
    for leg, lower, whole in zip(LEGS, LOWER_BODIES, WHOLE_BODIES, strict=True):
        a, b = build_model(lower), build_model(whole)
        assert (a.nu, b.nu) == (11, 9)
        assert (a.nq, b.nq) == (18, 16)
        assert b.body_mass.sum() < a.body_mass.sum() < healthy.body_mass.sum()
        assert mujoco.mj_name2id(a, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf") == -1
        assert mujoco.mj_name2id(b, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_hip") == -1
        assert f"{leg}_stump" in allowed_support_names(lower)
        assert len(allowed_support_names(whole)) == 3
        assert not any(n.startswith(leg) for n in allowed_support_names(whole))
        for m in (a, b):
            assert (m.body_inertia[1:] > 0).all()
            d = mujoco.MjData(m)
            initialize(m, d)
            assert np.isfinite(d.qpos).all()


def test_mixed_topologies_keep_semantic_joint_and_foot_slots():
    env = DogEnv(32, seed=4, threads=4, limb_stage="whole")
    try:
        assert [g.n for g in env.groups] == [8, 2, 2, 2, 2, 4, 4, 4, 4]
        assert (env.strength == 1).all() and (env.fault_at == 100000).all()
        for _ in range(5):
            env.step(np.zeros((32, 12)))
        for i, (g, s) in enumerate(zip(env.groups[-4:], env.slices[-4:], strict=True)):
            assert (env.valid[s, i * 3 : i * 3 + 3] == 0).all()
            assert (env.tip_forces[s, i] == 0).all()
            assert (env.tip_positions[s, i] == 0).all()
            assert (env.torque[s, i * 3 : i * 3 + 3] == 0).all()
            assert len(g.tip_geoms) == 3
        assert env.obs().shape == (32, 66)
    finally:
        env.close()
