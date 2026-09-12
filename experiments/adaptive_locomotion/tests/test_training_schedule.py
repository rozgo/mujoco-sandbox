"""Changing scheduling controls must preserve the original update arithmetic."""

import json

import pytest
import torch

from adaptive_locomotion.train import train


def test_fixed_update_limit_and_explicit_minibatches_preserve_legacy_training(tmp_path):
    states = []
    for label, size in (("legacy", None), ("explicit", 8)):
        output = tmp_path / label
        train(
            output,
            seconds=30,
            seed=9233,
            num_envs=8,
            mode="blind",
            bodies="healthy",
            device="cpu",
            epochs=2,
            horizon=4,
            minibatch_size=size,
            max_iterations=1,
        )
        report = json.loads((output / "training.json").read_text())
        assert report["iterations"] == 1
        assert report["transitions"] == 32
        assert report["optimizer_steps"] == 8
        assert report["stop_reason"] == "iteration_limit"
        states.append(torch.load(output / "policy.pt", weights_only=False)["state"])
    for key in states[0]:
        torch.testing.assert_close(states[0][key], states[1][key], atol=0, rtol=0)


@pytest.mark.parametrize("option", ["minibatch_size", "max_iterations"])
def test_invalid_training_schedule_fails_before_setup(tmp_path, option):
    with pytest.raises(ValueError, match="positive"):
        train(tmp_path, **{option: 0})
