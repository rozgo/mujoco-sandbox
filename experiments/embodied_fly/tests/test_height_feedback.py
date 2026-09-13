import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from scipy import sparse

from embodied_fly.batch import FlyBatch
from embodied_fly.brain import EmbodiedBrain, initialize_extended_actor
from embodied_fly.observations import actor_observation
from embodied_fly.train import load_episodes


def test_altitude_and_command_are_current_independent_and_reset_per_world():
    env = FlyBatch(2, 2, 14, preset="wing_motion")
    env.fields["qpos"][:, 2] = [2, 3]
    env.requested_height_cm[:] = [2.5, 4]
    env.batch.forward()
    obs = env.observation()
    assert obs.shape == (2, 397)
    np.testing.assert_array_equal(obs[:, -2:], [[1, 1.25], [1.5, 2]])
    single = env.template
    for name in ("qpos", "qvel", "act", "ctrl"):
        getattr(single.data, name)[:] = env.fields[name][0]
    single.requested_height_cm = 2.5
    mujoco.mj_forward(single.model, single.data)
    np.testing.assert_allclose(
        actor_observation(single, SimpleNamespace(sensor_extension_size=14)), obs[0], atol=1e-6
    )
    prefix = single.observation(True, wing_angles=True)
    np.testing.assert_array_equal(prefix, obs[0, :395])
    env.fields["qpos"][0, 2] = 6
    env.batch.forward()
    np.testing.assert_array_equal(env.observation()[0, -2:], [3, 1.25])
    env.reset([0])
    assert env.requested_height_cm[0] == 0
    assert env.requested_height_cm[1] == 4
    with pytest.raises(ValueError, match="complete wing"):
        single.observation(height_inputs=True)


def test_height_extension_neutral_migration_preserves_learned_wing_channels():
    graph = sparse.csr_matrix(
        (np.ones(4, np.float32), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6)
    )
    parent = (
        EmbodiedBrain(graph, [0, 1], [2, 3], [4, 5], 395, 78, sensor_extension_size=12)
        .double()
        .eval()
    )
    with torch.no_grad():
        parent.sensor_extension.weight.normal_(0, 0.1)
    child = (
        EmbodiedBrain(graph, [0, 1], [2, 3], [4, 5], 397, 78, sensor_extension_size=14)
        .double()
        .eval()
    )
    assert initialize_extended_actor(child, parent.state_dict())
    assert torch.equal(child.sensor_extension.weight[:, :12], parent.sensor_extension.weight)
    assert not child.sensor_extension.weight[:, 12:].any()
    old_memory, new_memory = parent.initial_state(2), child.initial_state(2)
    for _ in range(20):
        old_obs = torch.randn(2, 395, dtype=torch.float64)
        obs = torch.cat((old_obs, torch.randn(2, 2, dtype=torch.float64)), 1)
        old, new = parent(old_obs, old_memory), child(obs, new_memory)
        torch.testing.assert_close(new.action, old.action, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(new.state, old.state, atol=1e-12, rtol=1e-12)
        old_memory, new_memory = old.state.detach(), new.state.detach()
    new.action.square().mean().backward()
    gradient = child.sensor_extension.weight.grad[:, 12:]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_offline_height_matches_declared_command_and_never_future_state(tmp_path):
    env = FlyBatch(1, 1, 14, preset="wing_motion")
    mujoco.mj_saveModel(env.model, str(tmp_path / "model.mjb"))
    entries = []
    for i in range(4):
        qpos = np.tile(env.fields["qpos"][0], (6, 1))
        qpos[:, 2] = np.arange(6) + 1
        np.savez(
            tmp_path / f"episode_{i:03d}.npz",
            qpos=qpos,
            qvel=np.zeros((6, env.model.nv)),
            observation=np.zeros((6, 383)),
            action=np.zeros((6, 78)),
            activity=np.ones(6),
            requested_height_cm=np.full(6, 2.5),
        )
        entries.append({"episode": i, "failure": None, "final_upright": 1})
    (tmp_path / "manifest.json").write_text(
        json.dumps({"physical_preset": "wing_motion", "role": "flight", "episodes": entries})
    )
    episodes = load_episodes(tmp_path, True, True, True)
    np.testing.assert_array_equal(episodes[0]["observation"][:, -2], (np.arange(6) + 1) / 2)
    np.testing.assert_array_equal(episodes[0]["observation"][:, -1], 1.25)
    assert episodes[0]["observation"].shape == (6, 397)
