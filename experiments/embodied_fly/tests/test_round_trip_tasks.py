import numpy as np

from embodied_fly.batch import FlyBatch
from embodied_fly.round_trip import CASES, target_at
from embodied_fly.round_trip_tasks import RoundTripTasks, batched_targets


def test_vector_schedule_matches_reference_all_routes_and_jitter():
    rng = np.random.default_rng(5)
    n = 1000
    times = rng.uniform(0, 15, n)
    origins = rng.normal(size=(n, 3))
    routes = rng.integers(0, 7, n)
    amplitudes = rng.uniform(0.12, 0.18, n)
    expected = np.stack(
        [
            target_at(t, o, CASES[r][1], CASES[r][2], a)
            for t, o, r, a in zip(times, origins, routes, amplitudes)
        ]
    )
    np.testing.assert_allclose(batched_targets(times, origins, routes, amplitudes), expected)


def test_mixed_commands_do_not_move_body_and_match_post_step_observations():
    env = FlyBatch(12, 2, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    tasks = RoundTripTasks(env, 99)
    assert tasks.route_ids[:6].tolist() == [6] * 6
    assert tasks.route_ids[6:].tolist() == list(range(6))
    assert tasks.task_ids.tolist() == [2] * 12
    assert np.all((tasks.amplitudes >= 0.12) & (tasks.amplitudes <= 0.18))
    env.ages[:] = 1500
    before = {k: v.copy() for k, v in env.fields.items()}
    tasks.update_targets()
    for k, v in before.items():
        np.testing.assert_array_equal(env.fields[k], v)
    np.testing.assert_allclose(env.requested_xy_cm[:6], tasks.start[:6, :2])
    previous_observation = env.observation().copy()
    env.step(np.repeat(tasks.air_action[None], 12, axis=0))
    tasks.after_step()
    assert env.ages[0] == 1501
    expected = batched_targets(
        env.ages * env.control_dt / tasks.duration_scale,
        tasks.start,
        tasks.route_ids,
        tasks.amplitudes,
    )
    np.testing.assert_allclose(env.requested_xy_cm, expected[:, :2], atol=1e-7)
    np.testing.assert_allclose(env.requested_height_cm, expected[:, 2], atol=1e-7)
    assert not np.array_equal(env.observation(), previous_observation)
    assert sum(tasks.report()["transitions_by_target_sequence"].values()) == 12
    untouched = env.fields["qpos"][6].copy()
    tasks.reset([7])
    assert tasks.route_ids[7] == 2
    assert env.ages[7] == 0
    np.testing.assert_array_equal(env.fields["qpos"][6], untouched)
    assert not tasks.episode_metrics(7)["complete_target_sequence_tracking"]


def test_origin_only_cannot_pass_and_drift_is_measurement_only():
    env = FlyBatch(12, 2, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    tasks = RoundTripTasks(env, 100)
    tasks.duration_scale[:] = 1
    # Synthetic observer samples, no controller and no live trajectory claim.
    for seconds in (3.75, 6.75, 11.5):
        env.ages[:] = round(seconds / env.control_dt)
        tasks.after_step()
    assert tasks.episode_metrics(0)["complete_target_sequence_tracking"]
    assert not tasks.episode_metrics(6)["complete_target_sequence_tracking"]
    assert all(n > 0 for n in tasks.episode_metrics(6)["waypoint_window_samples"])
    tasks.reset(np.arange(12))
    tasks.duration_scale[:] = 1
    for seconds in (3.75, 6.75, 11.5):
        env.ages[:] = round(seconds / env.control_dt)
        target = tasks.update_targets()
        env.fields["qpos"][:, :3] = target  # test fixture, never live controller
        tasks.position_history[:] = target
        tasks.after_step()
    assert all(
        tasks.episode_metrics(i)["complete_target_sequence_tracking"] for i in range(12)
    )
    env.fields["qpos"][:, 0] += 0.1
    tasks.after_step()
    assert not any(
        tasks.episode_metrics(i)["complete_target_sequence_tracking"] for i in range(12)
    )


def test_evaluation_requires_intermediate_reach_and_counts_physical_failure():
    from embodied_fly.round_trip_evaluate import metrics

    times = np.arange(6000) * 0.002
    origins = np.repeat([[0.0, 0.0, 1.86651647]], 7, axis=0)
    targets = np.stack(
        [
            batched_targets(np.full(7, t + 0.002), origins, np.arange(7), np.full(7, 0.15))
            for t in times
        ]
    )
    arrays = {
        "time": times,
        "post_target": targets,
        "actual_position": targets.copy(),
        "actual_velocity": np.zeros_like(targets),
        "upright": np.ones((6000, 7)),
        "forbidden": np.zeros((6000, 7)),
    }
    assert all(c["passed"] for c in metrics(arrays, origins))
    arrays["actual_position"][:] = origins
    result = metrics(arrays, origins)
    assert not any(c["passed"] for c in result[:6])
    assert result[6]["passed"]
    arrays["forbidden"][100, 6] = 0.2
    assert not metrics(arrays, origins)[6]["passed"]
