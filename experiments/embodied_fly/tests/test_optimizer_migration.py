from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse

from embodied_fly.brain import EmbodiedBrain
from embodied_fly.train import configure_optimizer


def test_sensor_lr_change_retains_every_adam_moment_and_checkpoint_continuation():
    graph = sparse.eye(6, format="csr", dtype=np.float32)
    brain = EmbodiedBrain(graph, [0, 1], [2, 3], [4, 5], 16, 3, sensor_extension_size=12)
    optimizer = torch.optim.Adam(brain.parameters(), lr=3e-5)
    for _ in range(3):
        optimizer.zero_grad()
        sum(p.square().sum() + p.sum() for p in brain.parameters()).backward()
        optimizer.step()
    parent = {"optimizer_state_dict": optimizer.state_dict()}
    expected = {
        p: {key: value.clone() for key, value in state.items()}
        for p, state in optimizer.state.items()
    }
    args = SimpleNamespace(lr=3e-5, sensor_lr_multiplier=100)
    changed, resumed, grouped = configure_optimizer(brain, args, parent, False)
    assert resumed and grouped
    assert changed.param_groups[0]["lr"] == 3e-5
    assert changed.param_groups[1]["lr"] == 0.003
    assert changed.param_groups[1]["params"] == [brain.sensor_extension.weight]
    for p, state in expected.items():
        for key, value in state.items():
            torch.testing.assert_close(value, changed.state[p][key], rtol=0, atol=0)
    before = {p: p.detach().clone() for p in brain.parameters()}
    changed.zero_grad()
    sum(p.square().sum() + p.sum() for p in brain.parameters()).backward()
    changed.step()
    assert (
        brain.sensor_extension.weight - before[brain.sensor_extension.weight]
    ).abs().mean() > 0.002
    assert all(int(state["step"]) == 4 for state in changed.state.values())
    saved = {"optimizer_state_dict": changed.state_dict(), "optimizer_sensor_grouped": True}
    continued, resumed, grouped = configure_optimizer(brain, args, saved, False)
    assert resumed and grouped
    for p, state in changed.state.items():
        for key, value in state.items():
            torch.testing.assert_close(value, continued.state[p][key], rtol=0, atol=0)
