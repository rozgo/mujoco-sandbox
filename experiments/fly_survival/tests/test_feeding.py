import mujoco as mj
import numpy as np
from fly_survival.environment import Habitat


def test_stopped_fly_can_drink_from_a_real_patch_and_deplete_it():
    weights = np.zeros((6, 12))
    weights[2, 0] = 10
    h = Habitat(1, weights=weights, neural=False, vision=False, hazards=False)
    q = h.m.jnt_qposadr[h.m.joint("fly_00").id]
    yaw = -np.pi / 2
    h.d.qpos[q : q + 7] = [-10, -8.7, 1.05, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mj.mj_forward(h.m, h.d)
    h.previous_xyz = h.d.xpos[h.arena.fly_body_ids].copy()
    h.needs[0].hydration = 0.3
    h.run(2)
    assert h.needs[0].water_intake > 0.2
    assert h.needs[0].hydration > 0.45
    assert h.resources.remaining[2] < 4.8
    assert not h.d.warning.number.any()
