import copy

import mujoco
import numpy as np

from embodied_fly.record import expand_floor_display


def test_enlarging_plane_display_preserves_physics_beyond_former_visible_edge():
    original = mujoco.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="floor" type="plane" size="1 1 .1"/>
      <body pos="5 0 .5"><freejoint/><geom type="sphere" size=".1" mass="1"/></body>
    </worldbody></mujoco>""")
    display = copy.copy(original)
    changes = expand_floor_display(display)
    assert len(changes) == 1 and original.geom_size[0, 0] == 1
    assert display.geom_size[0, 0] == 30
    a, b = mujoco.MjData(original), mujoco.MjData(display)
    for _ in range(500):
        mujoco.mj_step(original, a)
        mujoco.mj_step(display, b)
        np.testing.assert_array_equal(a.qpos, b.qpos)
        np.testing.assert_array_equal(a.qfrc_constraint, b.qfrc_constraint)
    assert a.ncon > 0 and b.ncon > 0
