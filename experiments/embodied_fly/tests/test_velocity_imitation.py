import numpy as np
import torch
from scipy import sparse

from embodied_fly.fresh_velocity_actor import initialize
from embodied_fly.velocity_exercise import STAGES
from embodied_fly.velocity_imitation import (
    sample_windows,
    sequence_loss,
    validate_resume_recipe,
)


def test_replay_windows_cover_every_stage_and_keep_validation_separate():
    for episodes in (range(8), [8, 9]):
        worlds, indices, valid, stages = sample_windows(
            np.random.default_rng(11), episodes, 64, 128, 64, 35600
        )
        assert set(stages) == set(range(len(STAGES)))
        assert set(worlds) <= set(episodes)
        assert indices.shape == valid.shape == (192, 64)
        assert np.all(indices >= 0) and np.all(indices < 35600)
        assert np.all(np.diff(indices[64:], axis=0) == 1)


def test_replay_warmup_and_recomputed_gradients_match_direct_actor():
    graph = sparse.csr_matrix((np.ones(4), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6))
    actor = initialize(graph, [0, 1], [2, 3], [4, 5], 21)
    torch.manual_seed(51)
    observations = torch.randn(8, 2, 391) * 0.1
    targets = torch.randn(5, 2, 78) * 0.1
    valid = torch.ones((8, 2), dtype=torch.bool)
    valid[:2, 0] = False
    wings = torch.arange(14, 20)
    loss, *_ = sequence_loss(actor, observations, targets, valid, 3, wings, True)
    loss.backward()
    expected_gradients = {
        k: p.grad.clone() for k, p in actor.named_parameters() if p.grad is not None
    }
    actor.zero_grad()
    state = actor.initial_state(2)
    with torch.no_grad():
        for step in range(3):
            result = actor(observations[step], state)
            state = torch.where(valid[step][None], result.state, state)
    errors = []
    for step in range(3, 8):
        result = actor(observations[step], state)
        state = result.state
        errors.append((result.action - targets[step - 3]).square())
    errors = torch.stack(errors).mean((0, 1))
    other = np.r_[0:14, 20:78]
    direct = errors[wings].mean() + 0.1 * errors[other].mean()
    torch.testing.assert_close(loss, direct)
    direct.backward()
    for key, p in actor.named_parameters():
        if key in expected_gradients:
            torch.testing.assert_close(p.grad, expected_gradients[key])
    assert expected_gradients["sensory_encoder.0.weight"].norm() > 0
    assert expected_gradients["core.bias"].norm() > 0


def test_sampler_continuation_resumes_the_next_batch():
    import json

    rng = np.random.default_rng(31)
    args = (range(8), 64, 128, 64, 35600)
    sample_windows(rng, *args)
    serialized = json.dumps(rng.bit_generator.state)
    expected = sample_windows(rng, *args)
    resumed = np.random.default_rng(99)
    resumed.bit_generator.state = json.loads(serialized)
    for left, right in zip(expected, sample_windows(resumed, *args), strict=True):
        np.testing.assert_array_equal(left, right)


def test_cold_start_samples_preserve_all_stages_and_zero_context():
    worlds, indices, valid, stages = sample_windows(
        np.random.default_rng(31), range(8), 64, 128, 64, 35600, cold_starts=8
    )
    assert set(stages[:56]) == set(range(len(STAGES)))
    assert set(worlds[-8:]) == set(range(8))
    assert not valid[:64, -8:].any()
    assert valid[64:, -8:].all()
    np.testing.assert_array_equal(indices[64:, -8:], np.tile(np.arange(128)[:, None], (1, 8)))


def test_continuation_only_allows_declared_sampling_change():
    import pytest

    old = {"lr": 1e-4, "batch": 64}
    same = dict(old, cold_starts=0)
    new = dict(old, cold_starts=8)
    assert not validate_resume_recipe(old, same)
    with pytest.raises(ValueError):
        validate_resume_recipe(old, new)
    assert validate_resume_recipe(old, new, True)
    with pytest.raises(ValueError):
        validate_resume_recipe(old, dict(new, lr=3e-4), True)
