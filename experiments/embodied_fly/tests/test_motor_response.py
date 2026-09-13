import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.motor_response import perturb_commands, perturb_feedback


def test_previous_command_probe_keeps_all_measured_sensors_and_other_actions_unchanged():
    obs = np.zeros((3, 397), np.float32)
    obs[:, :297] = np.arange(297)
    wings = np.arange(14, 20)
    result = perturb_commands(obs, wings)
    np.testing.assert_array_equal(
        result[:, :, :297], np.repeat(obs[:, None, :297], 13, axis=1)
    )
    np.testing.assert_array_equal(result[:, :, 375:], 0)
    for i, channel in enumerate(wings):
        assert np.count_nonzero(result[:, 1 + 2 * i, 297:375]) == 3
        np.testing.assert_allclose(result[:, 1 + 2 * i, 297 + channel], 0.02)
        np.testing.assert_allclose(result[:, 2 + 2 * i, 297 + channel], -0.02)


def test_counterfactual_wing_feedback_matches_native_observation_recomputation():
    env = FlyBatch(25, 4, 14, preset="wing_motion")
    original = env.observation()[0:1]
    initial = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    probe = perturb_feedback(env.model, original, initial["qpos"][:1], initial["qvel"][:1])
    np.testing.assert_array_equal(probe[0, 0], original[0])
    for i in range(6):
        for sign, offset in ((1, 0), (-1, 1)):
            initial["qpos"][1 + i * 2 + offset, env.template.wing_angle_indices[i]] += (
                sign * 0.05
            )
            initial["qvel"][13 + i * 2 + offset, env.template.wing_velocity_indices[i]] += (
                sign * 2
            )
    # Explicit diagnostic initialization, not a pose correction during rollout.
    env.reset(np.arange(25), state=initial)
    np.testing.assert_allclose(probe[0], env.observation(), atol=1e-6, rtol=1e-6)
    assert np.isfinite(probe).all()
