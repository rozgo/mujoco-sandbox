import json

import numpy as np
import pytest
import torch

from embodied_fly.provenance import sha256
from embodied_fly.wing_observability import metrics, ridge_readout
from embodied_fly.wing_readout import load_feature_corpus, replace_wing_rows


def test_readout_uses_training_statistics_and_recovers_a_known_held_out_mapping():
    rng = np.random.default_rng(491)
    train = rng.normal(size=(200, 12))
    test = rng.normal(3, 2, (60, 12))  # changed test distribution must not refit normalization
    matrix = rng.normal(size=(12, 6))
    prediction = ridge_readout(train, train @ matrix + 4, test, alpha=1e-8)
    np.testing.assert_allclose(prediction, test @ matrix + 4, atol=2e-6)
    result = metrics(test @ matrix + 4, prediction, (train @ matrix + 4).mean(0))
    np.testing.assert_allclose(result["r2_against_training_mean"], 1, atol=1e-10)


def test_wing_calibration_leaves_other_actions_and_all_upstream_state_exactly_intact():
    weights, bias = torch.randn(78, 256), torch.randn(78)
    original = {
        "motor_decoder.3.weight": weights,
        "motor_decoder.3.bias": bias,
        "core.bias": torch.randn(20),
    }
    wings = np.array([2, 3, 4, 9, 10, 11])
    other = [i for i in range(78) if i not in wings]
    changed = replace_wing_rows(original, weights[wings] + 1, bias[wings] - 2, wings)
    hidden = torch.randn(5, 256)
    old = (hidden @ weights.T + bias).tanh()
    new = (
        hidden @ changed["motor_decoder.3.weight"].T + changed["motor_decoder.3.bias"]
    ).tanh()
    torch.testing.assert_close(old[:, other], new[:, other], rtol=0, atol=0)
    torch.testing.assert_close(original["core.bias"], changed["core.bias"], rtol=0, atol=0)
    assert not torch.equal(old[:, wings], new[:, wings])


def test_correction_corpora_keep_whole_episode_validation_out_and_verify_feature_origin(
    tmp_path,
):
    loaded = []
    for corpus_id in (0, 1):
        hidden = np.full((3, 4, 256), corpus_id, np.float32)
        target = np.full((3, 4, 6), corpus_id, np.float32)
        hidden[:, 1] = target[:, 1] = 99  # unmistakable held-out episode
        cache = tmp_path / f"features_{corpus_id}.npz"
        report = tmp_path / f"features_{corpus_id}.json"
        np.savez(
            cache,
            hidden=hidden,
            target=target,
            wing_channels=np.arange(6),
            training_indices=np.array([0, 2, 3]),
            validation_indices=np.array([1]),
        )
        report.write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "parent",
                    "frozen_feature_cache": {"sha256": sha256(cache)},
                }
            )
        )
        corpus = load_feature_corpus(cache, report, "parent", "cpu")
        assert torch.all(corpus["x"] == corpus_id) and torch.all(corpus["y"] == corpus_id)
        assert torch.all(corpus["test_x"] == 99) and torch.all(corpus["test_y"] == 99)
        assert corpus["report"]["unique_training_frames"] == 9
        loaded.append(corpus)
        with pytest.raises(ValueError, match="do not belong"):
            load_feature_corpus(cache, report, "different_parent", "cpu")
    assert sum(c["report"]["unique_training_frames"] for c in loaded) == 18
