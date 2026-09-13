from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy import sparse
from test_motor_focus import tiny_brain

from embodied_fly.batch import FlyBatch
from embodied_fly.brain import EmbodiedBrain
from embodied_fly.hover_migrate import transfer
from embodied_fly.hover_only import HoverBalancedReward, HoverOnlyReward, HoverOnlyTasks
from embodied_fly.observations import actor_observation
from embodied_fly.physical_contract import physical_contract


def test_transfer_preserves_actor_and_graph_with_neutral_observations():
    old = FlyBatch(1, 1, 14, preset="wing_position")
    new = FlyBatch(1, 1, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    actor = tiny_brain().double().eval()
    actor.set_motor_only()
    parent = {
        "physical_contract": physical_contract(old.model),
        "motor_only": True,
        "sensor_extension_size": 14,
        "observation_size": 397,
        "state_dict": actor.state_dict(),
        "config": {},
        "optimizer_state_dict": {"old": 1},
    }
    child, _ = transfer(parent, old.model, new.model, "test")
    graph = sparse.csr_matrix(
        (np.ones(4, np.float32), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6)
    )
    other = (
        EmbodiedBrain(
            graph, [0, 1], [2, 3], [4, 5], 399, 78, sensor_extension_size=16, motor_only=True
        )
        .double()
        .eval()
    )
    other.load_state_dict(child["state_dict"])
    a, b = actor.initial_state(2), other.initial_state(2)
    for _ in range(20):
        obs = torch.randn(2, 397, dtype=torch.float64)
        x, y = (
            actor(obs, a),
            other(torch.cat([obs, torch.randn(2, 2, dtype=torch.float64)], -1), b),
        )
        torch.testing.assert_close(x.action, y.action, rtol=1e-12, atol=1e-12)
        a, b = x.state.detach(), y.state.detach()
    assert "optimizer_state_dict" not in child and "optimizer_state_dict" in parent
    y.action.square().sum().backward()
    assert other.sensor_extension.weight.grad[:, -2:].abs().sum() > 0


def test_hover_only_feedback_reward_and_performance_curriculum():
    env = FlyBatch(4, 2, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    tasks = HoverOnlyTasks(env, 902)
    assert tasks.task_ids.tolist() == [2] * 4
    reward = HoverOnlyReward(env)
    before = {k: v.copy() for k, v in env.fields.items()}
    r, failed, terms = reward(env.previous_action)
    for k, v in before.items():
        np.testing.assert_array_equal(env.fields[k], v)
    np.testing.assert_allclose(r, sum(terms.values()) * 0.002 - failed)
    env.fields["qpos"][0, 0] += 0.2
    env.batch.forward()
    assert reward(env.previous_action)[0][0] < r[0]
    single = env.template
    for k in ("qpos", "qvel", "act", "ctrl"):
        getattr(single.data, k)[:] = env.fields[k][0]
    single.requested_height_cm = env.requested_height_cm[0]
    single.requested_xy_cm[:] = env.requested_xy_cm[0]
    mujoco.mj_forward(single.model, single.data)
    np.testing.assert_allclose(
        actor_observation(single, SimpleNamespace(sensor_extension_size=16)),
        env.observation()[0],
        atol=1e-6,
    )
    for _ in range(64):
        tasks.observe_episode({"failed": True, "simulated_seconds": 5, "return_": 10})
    tasks.advance_curriculum()
    assert tasks.widening == 0
    for _ in range(64):
        tasks.observe_episode({"failed": False, "simulated_seconds": 5, "return": 9})
    tasks.advance_curriculum()
    assert tasks.widening == 0.1
    untouched = env.fields["qpos"][0].copy()
    tasks.reset([3])
    np.testing.assert_array_equal(env.fields["qpos"][0], untouched)
    assert abs(env.fields["qpos"][3, 2] - env.requested_height_cm[3]) <= 0.002


def test_bounded_hover_scores_never_make_valid_airborne_failure_cheaper():
    env = FlyBatch(4, 2, 16, preset="wing_position", wing_response="instant", physics_hz=1000)
    HoverOnlyTasks(env, 901)
    old, new = HoverOnlyReward(env), HoverBalancedReward(env)
    original = {k: v.copy() for k, v in env.fields.items()}
    ideal = new(env.previous_action)[0].copy()
    for k, v in original.items():
        np.testing.assert_array_equal(env.fields[k], v)
    # Gross but finite position/velocity errors still permit positive recovery
    # return. The old recipe could make >0.1 s continued flight cost more than -1.
    env.fields["qpos"][1, 0] += 10
    env.fields["qvel"][2, :3] = 100
    env.fields["qvel"][2, 3:6] = 100
    env.batch.forward()
    reward, failed, terms = new(env.previous_action)
    assert not failed.any()
    assert np.all(reward >= 0.498 * env.control_dt)
    assert np.all(reward <= 6.1 * env.control_dt)
    assert reward[1] < ideal[1] and reward[2] < ideal[2]
    assert old(env.previous_action)[0][1] * 50 < -1
    np.testing.assert_allclose(reward, sum(terms.values()) * 0.002 - failed)
    env.forbidden_peak[0] = env.body_weight * 0.11
    env.fields["qpos"][1, 2] = 0.4
    reward, failed, _ = new(env.previous_action)
    assert failed[:2].all()
    np.testing.assert_array_equal(reward[:2], -1)
