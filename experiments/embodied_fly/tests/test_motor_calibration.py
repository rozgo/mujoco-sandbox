import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from embodied_fly.motor_calibration import decoder_from_state, load_corpus
from embodied_fly.motor_features import episode_split, replay_group
from embodied_fly.provenance import sha256


def test_variable_length_replay_matches_separate_histories_and_never_records_padding():
    class Actor:
        motor_ids = torch.arange(2)

        def initial_state(self, count):
            return torch.zeros(2, count)

        def __call__(self, observation, state, time_scale):
            state = state + observation.T * time_scale
            return SimpleNamespace(state=state, action=state.T.repeat(1, 39))

    episodes = [
        {"observation": np.full((n, 2), value, np.float32)}
        for n, value in ((2, 1), (7, 3), (5, -2))
    ]
    grouped = list(replay_group(Actor(), episodes, "cpu", 0.1))
    assert sum(len(row[0]) for row in grouped) == 14
    for index, episode in enumerate(episodes):
        individual = list(replay_group(Actor(), [episode], "cpu", 0.1))
        for active, frame, motor, action in grouped:
            if index not in active:
                continue
            local = list(active).index(index)
            np.testing.assert_array_equal(motor[local], individual[frame][2][0])
            np.testing.assert_array_equal(action[local], individual[frame][3][0])
    assert grouped[-1][0].tolist() == [1] and grouped[-1][1] == 6


def test_decoder_reconstruction_preserves_the_existing_module_function():
    torch.manual_seed(381)
    module = nn.Sequential(
        nn.LayerNorm(3), nn.Linear(3, 256), nn.Tanh(), nn.Linear(256, 78), nn.Tanh()
    )
    state = {"motor_decoder." + k: v for k, v in module.state_dict().items()}
    state["core.bias"] = torch.zeros(10)
    rebuilt = decoder_from_state(state)
    inputs = torch.randn(37, 3)
    torch.testing.assert_close(module(inputs), rebuilt(inputs), atol=0, rtol=0)


def test_feature_identity_and_whole_episode_exclusion_are_enforced(tmp_path):
    ids = np.repeat(np.arange(4), [2, 3, 4, 5])
    training, validation = episode_split(np.arange(4))
    x = np.repeat(ids[:, None], 3, axis=1).astype(np.float32)
    action = np.zeros((14, 78), np.float32)
    frame = np.concatenate([np.arange(n) for n in [2, 3, 4, 5]])
    state = {
        "motor_decoder.1.weight": torch.zeros(256, 3),
        "motor_decoder.3.weight": torch.zeros(78, 256),
    }

    def write(train_ids):
        np.savez(
            tmp_path / "features.npz",
            motor=x,
            parent_action=action,
            teacher_action=action,
            frame=frame,
            episode=ids,
            training_episodes=train_ids,
            validation_episodes=validation,
            wing_channels=np.arange(14, 20),
        )
        (tmp_path / "report.json").write_text(
            json.dumps(
                {
                    "schema": "frozen-motor-cell-features-v1",
                    "role": "ground",
                    "checkpoint_sha256": "parent",
                    "cache_sha256": sha256(tmp_path / "features.npz"),
                }
            )
        )

    write(training)
    c = load_corpus(tmp_path, "parent", state, "cpu")
    assert set(c["train"]["x"][:, 0].tolist()) == set(training.tolist())
    assert set(c["valid"]["x"][:, 0].tolist()) == set(validation.tolist())
    assert c["identity"]["training_frames"] + c["identity"]["validation_frames"] == 14
    with pytest.raises(ValueError, match="mismatch"):
        load_corpus(tmp_path, "another_parent", state, "cpu")
    write(np.arange(4))
    with pytest.raises(ValueError, match="overlap"):
        load_corpus(tmp_path, "parent", state, "cpu")
