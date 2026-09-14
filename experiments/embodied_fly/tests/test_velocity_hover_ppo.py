from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_brain import make_brain

from embodied_fly.ppo import joint_log_probability, motor_distribution
from embodied_fly.velocity_hover import HoverReward, reward_rates
from embodied_fly.velocity_hover_ppo import replay, replay_audit


def test_vertical_reward_adjustment_changes_only_vertical_term():
    velocity = np.array([[0, 0, 0], [0.7, -1.2, 0.5], [3, 2, -4]])
    angular = np.full((3, 3), 0.1)
    upright = np.array([1.0, 0.9, 0.85])
    old = reward_rates(velocity, angular, upright, horizontal_scale=2)
    new = reward_rates(velocity, angular, upright, horizontal_scale=2, vertical_weight=3)
    np.testing.assert_allclose(new["vertical"], old["vertical"] * 1.5)
    for key in ("alive", "horizontal", "angular", "upright"):
        np.testing.assert_array_equal(old[key], new[key])
    assert new["vertical"][0] == 3
    for invalid in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="vertical reward weight"):
            reward_rates(velocity, angular, upright, vertical_weight=invalid)


def test_wider_horizontal_reward_recognizes_reducing_large_drift_without_moving_optimum():
    speeds = np.array([[0, 0, 0], [2.5, 0, 0], [3.0, 0, 0]])
    old = reward_rates(speeds, np.zeros((3, 3)), np.ones(3))
    new = reward_rates(speeds, np.zeros((3, 3)), np.ones(3), horizontal_scale=2.0)
    assert new["horizontal"][0] == old["horizontal"][0] == 1
    assert new["horizontal"][0] > new["horizontal"][1] > new["horizontal"][2]
    assert new["horizontal"][1] - new["horizontal"][2] > 7 * (
        old["horizontal"][1] - old["horizontal"][2]
    )
    for key in ("alive", "vertical", "angular", "upright"):
        np.testing.assert_array_equal(old[key], new[key])


@pytest.mark.parametrize("vertical_weight", [2.0, 3.0])
def test_hover_reward_ordering_and_failure_cannot_earn_remaining_alive_bonus(vertical_weight):
    def rates(velocity, angular, upright):
        return reward_rates(velocity, angular, upright, vertical_weight=vertical_weight)

    stable = sum(rates(np.zeros((1, 3)), np.zeros((1, 3)), np.ones(1)).values())[0]
    climbing = sum(rates(np.array([[0, 0, 3.0]]), np.zeros((1, 3)), np.ones(1)).values())[0]
    poor = sum(rates(np.full((1, 3), 100), np.full((1, 3), 100), np.array([0.85])).values())[0]
    assert stable == 3 + vertical_weight and stable > climbing > poor >= 1
    gamma = np.exp(-0.002 / 2)

    def outcome(rates, fail_at=None):
        return sum(
            gamma**t * (-2 if t == fail_at else rate * 0.002) for t, rate in enumerate(rates)
        )

    # A recoverable bad interval followed by stable hover beats persistent drift;
    # both beat immediate failure. Delaying failure also beats early termination.
    assert outcome([poor] * 100 + [stable] * 400) > outcome([climbing] * 500)
    assert outcome([poor] * 500) > outcome([poor], 0)
    assert outcome([poor] * 101, 100) > outcome([poor], 0)


def test_reward_filter_is_causal_reset_local_and_not_part_of_dynamics():
    qpos = np.zeros((2, 7))
    qpos[:, 2] = 2
    qvel = np.zeros((2, 6))
    env = SimpleNamespace(
        n=2,
        fields={
            "qpos": qpos,
            "qvel": qvel,
            "xmat": np.tile(np.eye(3).reshape(1, 1, 9), (2, 1, 1)),
        },
        velocity=lambda: np.zeros((2, 6)),
        template=SimpleNamespace(thorax_id=0),
        forbidden_peak=np.zeros(2),
        body_weight=1,
        control_dt=0.002,
    )
    reward = HoverReward(env)
    for i in range(50):
        qvel[:, 2] = np.sin(2 * np.pi * 30 * i * 0.002)
        reward()
    np.testing.assert_allclose(reward.mean()[:, 2], 0, atol=1e-14)
    qvel[0, 2] = 7
    previous_state = qvel.copy()
    reward.reset([0])
    reward()
    assert reward.mean()[0, 2] == 7
    np.testing.assert_array_equal(qvel, previous_state)
    qpos[0, 2] = 0.4
    r, failed, _ = reward()
    assert failed[0] and r[0] == -2 and r[1] > 0


def test_actual_ppo_replay_reproduces_sampling_with_episode_resets_and_nonzero_memory():
    torch.manual_seed(32)
    actor = make_brain().train()
    actor.set_motor_only(True)
    active = torch.ones(3, dtype=torch.bool)
    log_std = torch.full((3,), np.log(0.003))
    data = {"obs": torch.randn(12, 2, 4), "done": torch.zeros(12, 2, dtype=torch.bool)}
    data["done"][5, 0] = True
    data["done"][9, 1] = True
    memory = torch.randn_like(actor.initial_state(2)) * 0.01
    states, latent, means, logps = [], [], [], []
    with torch.no_grad():
        for t in range(12):
            if t % 4 == 0:
                states.append(memory.clone())
            result = actor(data["obs"][t], memory)
            dist = motor_distribution(result, active, log_std, 0.0005)
            z = dist.sample()
            latent.append(z)
            means.append(result.action)
            logps.append(
                joint_log_probability(result, dist, z, result.activity, motor_only=True)
            )
            memory = actor.reset_worlds(result.state, data["done"][t])
    data.update(
        latent=torch.stack(latent), mean_action=torch.stack(means), logp=torch.stack(logps)
    )
    assert replay_audit(actor, data, states, 4, active, log_std)["passed"]
    logp, _, _ = replay(actor, data, states[1], 4, 8, active, log_std, True)
    torch.testing.assert_close(logp.detach(), data["logp"][4:8], rtol=0, atol=1e-5)
    (-logp.mean()).backward()
    for parameter in actor.core.parameters():
        assert torch.isfinite(parameter.grad).all() and parameter.grad.norm() > 0
