import mujoco as mj
import numpy as np
from fly_survival.environment import Habitat
from fly_survival.scene import LAMP_POS, build
from fly_survival.senses import Sensors


def test_heat_ray_is_exposed_at_lamp_and_blocked_by_physical_roof():
    a = build(1)
    s = Sensors(a, vision=False)
    assert s.illumination(LAMP_POS, 1.6) > 0.99
    # Put the emitter's XY coordinate underneath a real roof for this geometry test.
    # Translate only the sensor's query source through a temporary module constant.
    import fly_survival.senses as module

    previous = module.LAMP_POS.copy()
    try:
        module.LAMP_POS[:] = [-19, -3]
        assert s.illumination(np.array([-19.0, -3.0]), 1.6) == 0
    finally:
        module.LAMP_POS[:] = previous


def test_physical_swatter_contact_causes_damage_without_numerical_failure():
    weights = np.zeros((6, 12))
    weights[3, 0] = 10
    env = Habitat(1, weights=weights, neural=False, vision=False)
    adr = env.m.jnt_qposadr[env.m.joint("fly_00").id]
    env.d.qpos[adr : adr + 3] = [15, 6, 1.05]
    mj.mj_forward(env.m, env.d)
    env.previous_xyz = env.d.xpos[env.arena.fly_body_ids].copy()
    env.next_swat = 0.2
    report = env.run(1.0)
    assert report["total_impulse_model_units"][0] > 0
    assert report["alive"] == 0
    assert not env.d.warning.number.any()
    assert (
        report["reward"] > -5
    )  # injury penalty uses actual health loss, not overkill impulse
