import json

import numpy as np
import torch
from scipy import sparse
from test_motor_focus import tiny_brain

from embodied_fly.provenance import sha256


def test_zero_readout_preserves_parent_and_nonzero_readout_uses_only_motor_cells():
    torch.manual_seed(818)
    parent = tiny_brain().eval()
    parent.set_motor_only()
    child = tiny_brain().eval()
    child.load_state_dict(parent.state_dict())
    child.set_motor_only()
    child.enable_wing_residual()
    a, b = parent.initial_state(3), child.initial_state(3)
    with torch.no_grad():
        for _ in range(8):
            obs = torch.randn(3, 397)
            old, new = parent(obs, a), child(obs, b)
            torch.testing.assert_close(old.action, new.action, atol=0, rtol=0)
            torch.testing.assert_close(old.state, new.state, atol=0, rtol=0)
            a, b = old.state, new.state
        child.wing_residual.weight.normal_(0, 0.1)
        child.wing_residual.bias.fill_(0.02)
        old, new = parent(obs, a), child(obs, b)
        torch.testing.assert_close(old.state, new.state, atol=0, rtol=0)
        other = np.r_[0:14, 20:78]
        torch.testing.assert_close(old.action[:, other], new.action[:, other], atol=0, rtol=0)
        assert (old.action[:, 14:20] - new.action[:, 14:20]).abs().max() > 0.001
        motor = new.state[child.motor_ids].T
        normalized = child.motor_decoder[0](motor)
        base = child.motor_decoder[3](
            child.motor_decoder[2](child.motor_decoder[1](normalized))
        )
        expected = (base[:, 14:20] + child.wing_residual(normalized)).tanh()
        torch.testing.assert_close(new.action[:, 14:20], expected, atol=0, rtol=0)
        assert (new.action.abs() <= 1).all()


def test_standard_loader_restores_residual_checkpoint(tmp_path, monkeypatch):
    from embodied_fly import evaluate

    brain = tiny_brain().eval()
    brain.set_motor_only()
    brain.enable_wing_residual()
    with torch.no_grad():
        brain.wing_residual.weight.fill_(0.03)
    (tmp_path / "weights.npz").write_bytes(b"test-graph")
    (tmp_path / "brain.npz").write_bytes(b"test-metadata")
    graph = sparse.csr_matrix(
        (np.ones(4, np.float32), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6)
    )
    monkeypatch.setattr(evaluate, "load_malecns", lambda _: (graph, [0, 1], [2, 3], [4, 5]))
    path = tmp_path / "actor.pt"
    checkpoint = {
        "state_dict": brain.state_dict(),
        "observation_size": 397,
        "sensor_extension_size": 14,
        "action_size": 78,
        "config": {"internal_steps": 4},
        "motor_only": True,
        "wing_residual_enabled": True,
        "graph_sha256": sha256(tmp_path / "weights.npz"),
        "graph_metadata_sha256": sha256(tmp_path / "brain.npz"),
    }
    torch.save(checkpoint, path)
    restored, meta = evaluate.load_actor(path, tmp_path, torch.device("cpu"))
    assert meta["wing_residual_enabled"]
    obs = torch.randn(2, 397)
    with torch.no_grad():
        a = brain(obs, brain.initial_state(2))
        b = restored(obs, restored.initial_state(2))
    torch.testing.assert_close(a.action, b.action, atol=0, rtol=0)
    torch.testing.assert_close(a.state, b.state, atol=0, rtol=0)
    assert json.dumps({"inputs": restored.observation_size, "outputs": restored.action_size})
