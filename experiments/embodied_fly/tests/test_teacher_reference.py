import mujoco
import numpy as np

from embodied_fly.body import FlyEnvironment
from embodied_fly.teacher import TeacherOracle


def test_receding_reference_is_translation_and_heading_invariant_without_moving_body():
    environment = FlyEnvironment()
    # Exercise the reference adapter with real MuJoCo poses, without loading a
    # neural teacher or downloading weights. Empty sensor lists are immaterial
    # to the reference fields being checked.
    oracle = object.__new__(TeacherOracle)
    oracle.environment = environment
    oracle.joint_ids = oracle.activation_ids = oracle.appendage_ids = []
    oracle.sensors = {}
    oracle.set_reference(1.0, 0.75, 2.0)
    baseline = oracle.observation(0, reference_mode="receding")
    environment.data.qpos[:2] = (2, -3)
    environment.data.qpos[3:7] = (np.sqrt(0.5), 0, 0, np.sqrt(0.5))
    mujoco.mj_forward(environment.model, environment.data)
    before = environment.data.qpos.copy()
    transformed = oracle.observation(90, reference_mode="receding")
    for key in ("walker/ref_displacement", "walker/ref_root_quat"):
        np.testing.assert_allclose(transformed[key], baseline[key], atol=1e-12)
    np.testing.assert_array_equal(environment.data.qpos, before)
    assert environment.data.time == 0
