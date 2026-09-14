from types import SimpleNamespace

import numpy as np

from embodied_fly.hover_gusts import HoverGusts


def test_pulses_add_without_accumulation_leave_calm_worlds_and_clear_after_reset():
    force = np.ones((8, 1, 6))
    env = SimpleNamespace(
        n=8,
        ages=np.zeros(8, dtype=int),
        control_dt=0.002,
        fields={"xfrc_applied": force},
        template=SimpleNamespace(thorax_id=0),
        wing_forces=SimpleNamespace(
            mass=2,
            config=SimpleNamespace(horizontal_drag_seconds=0.12, vertical_drag_seconds=0.08),
        ),
    )
    gusts = HoverGusts(env, 2.0, 420)
    gusts.reset(np.arange(8))
    gusts.next_start[:] = 0
    gusts.before_step()
    expected = force.copy()
    np.testing.assert_array_equal(force[:2], 1)
    np.testing.assert_array_equal(force[:, :, 3:], 1)
    assert len(gusts.events) == 6
    assert np.count_nonzero(gusts.applied, axis=1).tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
    for e in gusts.events:
        assert 1 <= abs(e["equivalent_drift_cm_s"]) <= 2
        assert abs(e["force"]) <= 2 * 2 / 0.08
    gusts.before_step()
    np.testing.assert_array_equal(force, expected)
    # A changing wing wrench or another external force is preserved.
    force += 3
    env.ages[:] = gusts.duration
    gusts.before_step()
    np.testing.assert_allclose(force, 4)
    # Native reset clears force; resetting the sampler must not subtract an old pulse.
    gusts.next_start[3] = env.ages[3]
    gusts.before_step()
    force[3] = 0
    env.ages[3] = 0
    gusts.reset(np.array([3]))
    gusts.before_step()
    np.testing.assert_array_equal(force[3], 0)
