from types import SimpleNamespace

import numpy as np
import torch
from test_motor_focus import tiny_brain

from embodied_fly.batch import FlyBatch
from embodied_fly.physical_contract import physical_contract


def test_continuous_commands_preserve_physics_and_neural_memory(tmp_path, monkeypatch):
    from embodied_fly import motor_sequence

    actor = tiny_brain()
    actor.set_motor_only()
    initial = {k: v.clone() for k, v in actor.state_dict().items()}
    env = FlyBatch(3, 3, 14, preset="wing_position")
    monkeypatch.setattr(
        motor_sequence,
        "load_actor",
        lambda *a: (actor, {"physical_contract": physical_contract(env.model)}),
    )

    def forbidden_reset(*args):
        raise AssertionError("Command changes must not reset recurrent memory")

    monkeypatch.setattr(actor, "reset_worlds", forbidden_reset)
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"test")
    r = motor_sequence.run(
        SimpleNamespace(
            checkpoint=checkpoint,
            graph=tmp_path,
            output=tmp_path / "run",
            device="cpu",
            seed=99103,
            phase_seconds=0.5,
            neural_view=False,
        )
    )
    assert r["transitions"] == 3000 and r["live_resets"] == 0
    assert r["active_actuators"] == 78 and not r["walking_action_mask"]
    assert len(r["boundary_events"]) == 4
    assert all(
        e["physical_state_unchanged_at_boundary"] and e["neural_memory_unchanged_at_boundary"]
        for e in r["boundary_events"]
    )
    assert all(e["preceding_memory_l2"] > 0 for e in r["boundary_events"][1:])
    assert all(torch.equal(actor.state_dict()[k], v) for k, v in initial.items())
    for i in range(3):
        z = np.load(tmp_path / "run" / f"sequence_{i}.npz")
        np.testing.assert_array_equal(z["qpos"][1:], z["post_action_qpos"][:-1])
        np.testing.assert_array_equal(z["observation"][1:, 297:375], z["action"][:-1])
        np.testing.assert_array_equal(z["command"][:, 0], np.repeat([0, 1, 0, 1], 250))
        assert len(z["qpos"]) == 1000 and abs(z["action"]).max() <= 1
