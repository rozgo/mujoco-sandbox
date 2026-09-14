import numpy as np
import pytest

from embodied_fly.batch import FlyBatch
from embodied_fly.flight_tracking_reward import (
    FlightTrackingReward,
    TrackingConfig,
    desired_velocity,
    velocity_scores,
)
from embodied_fly.hover_only import HoverBalancedReward
from embodied_fly.round_trip_tasks import RoundTripTasks, batched_target_motion


def test_requested_velocity_is_true_derivative_with_timing_jitter():
    rng = np.random.default_rng(55)
    times = rng.uniform(0, 14, 400)
    origins = rng.normal(size=(400, 3))
    routes = rng.integers(0, 7, 400)
    amplitudes = rng.uniform(0.12, 0.18, 400)
    scales = rng.uniform(0.85, 1.15, 400)
    _, velocity = batched_target_motion(times, origins, routes, amplitudes, scales)
    before = batched_target_motion(times - 1e-6, origins, routes, amplitudes, scales)[0]
    after = batched_target_motion(times + 1e-6, origins, routes, amplitudes, scales)[0]
    np.testing.assert_allclose(velocity, (after - before) / 2e-6, atol=1e-8)
    _, held = batched_target_motion(np.full(400, 12.0), origins, routes, amplitudes, scales)
    np.testing.assert_array_equal(held, 0)


def test_goal_velocity_is_continuous_bounded_and_rotationally_consistent():
    rng = np.random.default_rng(6)
    pos = rng.normal(size=(80, 3))
    target = rng.normal(size=(80, 3))
    reference = rng.normal(size=(80, 3))
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    correction = desired_velocity(pos, target, reference) - reference
    assert np.max(np.linalg.norm(correction, axis=-1)) <= 0.5 + 1e-12
    np.testing.assert_allclose(
        desired_velocity(pos @ rotation, target @ rotation, reference @ rotation),
        desired_velocity(pos, target, reference) @ rotation,
        atol=1e-12,
    )
    np.testing.assert_array_equal(desired_velocity(pos, pos, reference), reference)
    for distance in (0.01, 0.1, 0.249, 0.25, 0.251, 1):
        wanted = desired_velocity(
            np.zeros((1, 3)), np.array([[distance, 0, 0]]), np.zeros((1, 3))
        )
        np.testing.assert_allclose(wanted, [[min(2 * distance, 0.5), 0, 0]])
    for bad in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            TrackingConfig(approach_seconds=bad)


def test_returning_is_better_than_stopping_or_moving_away_at_equal_error():
    for axis in range(3):
        for sign in (-1, 1):
            position = np.zeros((3, 3))
            position[:, axis] = sign * 0.8
            target = np.zeros_like(position)
            reference = np.zeros_like(position)
            velocity = np.zeros_like(position)
            velocity[0, axis] = -sign * 0.5
            velocity[2, axis] = sign * 0.5
            terms = velocity_scores(velocity, desired_velocity(position, target, reference))
            rate = sum(terms.values())
            assert rate[0] > rate[1] > rate[2]
    # At a stationary target the optimum is zero velocity.
    velocity = np.array([[0, 0, 0], [0.5, 0, 0], [-0.5, 0, 0]])
    scores = velocity_scores(velocity, np.zeros_like(velocity))
    assert sum(scores.values())[0] > sum(scores.values())[1]
    assert sum(scores.values())[1] == sum(scores.values())[2]


def test_real_reward_changes_only_velocity_scores_and_never_physical_state():
    env = FlyBatch(12, 2, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    tasks = RoundTripTasks(env, 7)
    old, new = HoverBalancedReward(env, 2), FlightTrackingReward(env, tasks, 2)
    baseline = old(env.previous_action)
    initial = new(env.previous_action)
    np.testing.assert_allclose(initial[0], baseline[0], atol=1e-8)
    env.ages[:] = round(1.5 / env.control_dt)
    tasks.update_targets()
    saved = {k: v.copy() for k, v in env.fields.items()}
    old_terms = old(env.previous_action)[2]
    reward, failed, terms = new(env.previous_action)
    for k, v in saved.items():
        np.testing.assert_array_equal(env.fields[k], v)
    for k in old_terms:
        if k not in ("vertical_velocity", "horizontal_velocity"):
            np.testing.assert_array_equal(terms[k], old_terms[k])
    np.testing.assert_allclose(reward, sum(terms.values()) * env.control_dt - failed)
    env.forbidden_peak[0] = env.body_weight * 0.11
    reward, failed, terms = new(env.previous_action)
    assert failed[0] and reward[0] == -1
    assert all(term[0] == 0 for term in terms.values())
    assert all(reward[1:] >= 0.498 * env.control_dt)
    assert "goal-directed" in tasks.report()["reward"]
