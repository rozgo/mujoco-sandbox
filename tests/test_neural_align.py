"""Native alignment environment: determinism, physics, shaping and baselines."""

import numpy as np
import pytest

from sixlegs.neural_insertion.align_env import AlignEnv, EVALUATION_SEEDS, evaluate, scripted


def test_same_seed_and_actions_reproduce_exactly():
    a, b = AlignEnv(42), AlignEnv(42)
    assert np.array_equal(a.reset(), b.reset())
    rng = np.random.default_rng(0)
    for _ in range(60):
        action = rng.uniform(-1, 1, 3).astype(np.float32)
        ra, rb = a.step(action), b.step(action)
        assert np.array_equal(ra[0], rb[0]) and ra[1] == rb[1] and ra[2] == rb[2]


def test_zero_action_holds_the_tip_still_under_the_native_servo():
    env = AlignEnv(7)
    env.reset()
    start = env.state()["tip"].copy()
    for _ in range(100):
        env.step(np.zeros(3, np.float32))
    assert np.max(np.abs(env.state()["tip"]-start)) < 1e-6


def test_shaping_reward_is_the_change_in_log_distance_potential():
    env = AlignEnv(11)
    env.reset()
    s0 = env.state()
    total = 0.
    for _ in range(30):
        _, reward, done, _ = env.step(np.full(3, .3, np.float32))
        total += reward
        assert not done
    s1 = env.state()
    potential = [-np.log10((np.linalg.norm(s["tip"]-s["goal"])+2e-6)/1e-3) for s in (s0, s1)]
    smooth = .002*(3*.3**2)  # only the first step changes the action
    assert total == pytest.approx(potential[1]-potential[0]-smooth, abs=.06)  # vessel penalty may add a few steps


def test_driving_into_the_phantom_ends_the_episode_as_a_collision():
    env = AlignEnv(3)
    env.reset()
    done, info = False, None
    while not done:
        _, reward, done, info = env.step(np.array([0, 0, -1], np.float32))
    assert info["collision"] == 1 and info["success"] == 0 and reward < -1


def test_scripted_reference_succeeds_on_evaluation_seeds():
    summary, results = evaluate(scripted, seeds=EVALUATION_SEEDS[:10])
    assert summary["success"] == 1 and summary["collision"] == 0
    assert max(r["final_lateral_um"] for r in results) < 10


def test_out_of_range_actions_are_clipped_before_any_reward_term():
    a, b = AlignEnv(5), AlignEnv(5)
    a.reset(), b.reset()
    for scale in (1., 50., 1e4):
        ra = a.step(np.array([scale, -scale, .5], np.float32))
        rb = b.step(np.array([1., -1., .5], np.float32))
        assert ra[1] == rb[1] and np.array_equal(ra[0], rb[0])
        assert abs(ra[1]) < 5


def test_trained_checkpoint_succeeds_on_held_out_seeds():
    from sixlegs.neural_insertion.align_policy import PUFFER, Policy
    from sixlegs.neural_insertion.scene import ROOT
    weights = ROOT/"assets/neural_insertion/align_v3_policy.bin"
    if not (PUFFER/"src/puffercpu.c").exists() or weights.read_bytes()[:7] == b"version":
        pytest.skip("needs the pinned PufferLib checkout and the LFS checkpoint")
    policy = Policy(weights)
    # A fresh episode starts at tick 0, which also zeroes the recurrent state.
    summary, _ = evaluate(lambda env, obs: policy.act(obs, env.state()["tick"] == 0), seeds=EVALUATION_SEEDS[:10])
    assert summary["success"] == 1 and summary["collision"] == 0
