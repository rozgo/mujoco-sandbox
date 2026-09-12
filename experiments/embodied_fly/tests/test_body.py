import mujoco
import numpy as np
import pytest

from embodied_fly.body import FlyEnvironment


@pytest.fixture(scope="module")
def environment():
    return FlyEnvironment()


def test_complete_free_anatomy_and_limits(environment):
    model = environment.model
    assert model.nu == 78
    assert model.nv == 108
    assert model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert all("ghost" not in model.body(i).name for i in range(model.nbody))
    assert model.body_mass.sum() * 0.001 == pytest.approx(9.846214691323196e-7)
    assert np.all(model.actuator_ctrllimited)
    assert np.all(model.actuator_forcelimited)
    assert np.isfinite(model.actuator_forcerange).all()
    for group in ("wing", "labrum", "antenna", "tibia"):
        assert any(group in n for n in environment.action_names)


def test_observation_and_invalid_action_do_not_advance_body(environment):
    observation = environment.reset()
    assert observation.shape == (383,)
    assert np.isfinite(observation).all()
    before = environment.data.qpos.copy()
    with pytest.raises(ValueError):
        environment.step(np.full(78, np.nan))
    np.testing.assert_array_equal(before, environment.data.qpos)
    assert environment.data.time == 0


def test_walking_mask_preserves_active_commands_and_passive_physical_dofs(environment):
    original = np.linspace(-1, 1, environment.model.nu, dtype=np.float32)
    masked = environment.walking_action(original)
    active = ~environment.walking_inactive
    np.testing.assert_array_equal(masked[active], original[active])
    raw = environment.low + (masked + 1) * 0.5 * (environment.high - environment.low)
    np.testing.assert_allclose(raw[~active], 0, atol=1e-6)
    assert active.sum() == 59
    assert environment.model.nv == 108
    assert not np.shares_memory(masked, original)
