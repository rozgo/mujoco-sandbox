import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.ground_posture import GroundPosture
from embodied_fly.physical_contract import physical_contract


def test_ground_targets_restore_wings_preserve_walking_and_leave_hover_free():
    env = FlyBatch(3, 3, 14, preset="wing_motion")
    ref = GroundPosture(env, env.fields["qpos"][0])
    contract = physical_contract(env.model)
    state = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    state["qpos"][:, env.template.wing_angle_indices] += 0.15
    state["qvel"][:, env.template.wing_velocity_indices] = 2
    env.reset(np.arange(3), state=state)
    before = env.fields["qpos"].copy()
    source = np.random.default_rng(42).uniform(-1, 1, (3, 78)).astype(np.float32)
    output = ref.targets(source, np.arange(3))
    np.testing.assert_array_equal(output[2], source[2])
    other = np.setdiff1d(np.arange(78), ref.wings)
    np.testing.assert_array_equal(output[1, other], source[1, other])
    np.testing.assert_array_equal(output[0, other], ref.rest_action[other])
    assert np.all(output[:2, ref.wings] < 0)
    np.testing.assert_array_equal(before, env.fields["qpos"])
    assert physical_contract(env.model) == contract
    # Apply only bounded physical actuation to recover an intentionally displaced
    # wing; no inference masks, qpos corrections or changed wing mechanics.
    initial = ref.measure()["wing_max_deviation_rad"].copy()
    for _ in range(100):
        env.step(ref.targets(source, np.zeros(3, dtype=int)))
    final = ref.measure()
    assert np.all(final["wing_max_deviation_rad"] < initial * 0.1)
    assert np.all(final["wings_velocity_mse_rad2_s2"] < 0.05)
    assert np.all(env.fields["xmat"][:, env.template.thorax_id, 8] > 0.99)
    assert np.all(env.forbidden_peak == 0)


def test_initial_form_covers_all_hinges_and_detects_body_sag():
    env = FlyBatch(1, 1, 14, preset="wing_motion")
    ref = GroundPosture(env, env.fields["qpos"][0])
    assert np.all(sum(ref.groups.values()) == 1)
    assert len(ref.report()["initial_hinge_angles_rad"]) == len(env.template.joint_ids)
    np.testing.assert_allclose(ref.measure()["initial_form_score"], 1)
    state = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    state["qpos"][:, 2] *= 0.8
    for mask in ref.groups.values():
        state["qpos"][:, env.template.qpos_indices[mask]] += 0.2
    env.reset([0], state=state)
    metrics = ref.measure()
    assert metrics["initial_form_score"][0] < 0.01
    np.testing.assert_allclose(metrics["body_height_loss_fraction"], 0.2)
    for group in ref.groups:
        np.testing.assert_allclose(metrics[group + "_angle_mse_rad2"], 0.04)
