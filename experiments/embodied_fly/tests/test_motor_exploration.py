from types import SimpleNamespace

import numpy as np
import torch
from test_motor_focus import tiny_brain

from embodied_fly.motor_exploration import paired_actions
from embodied_fly.ppo import motor_distribution


def test_sampling_matches_ppo_and_zero_noise_preserves_saturated_actions():
    mean = torch.tensor([[1.0, -0.3, 0.0], [0.5, -0.3, 0.0]])
    epsilon = torch.tensor([[1.0, -2.0, 3.0], [1.0, -2.0, 3.0]])
    output = paired_actions(mean, [0, 0.01], epsilon)
    torch.testing.assert_close(output[0], mean[0], rtol=0, atol=0)
    dist = motor_distribution(
        SimpleNamespace(action=mean),
        torch.ones(3, dtype=torch.bool),
        torch.full((3,), np.log(0.01)),
    )
    torch.testing.assert_close(output[1], (dist.loc + dist.scale * epsilon).tanh()[1])


def test_frozen_probe_runs_matched_physical_cases_without_learning(tmp_path, monkeypatch):
    from embodied_fly import motor_exploration as probe
    from embodied_fly.batch import FlyBatch
    from embodied_fly.physical_contract import physical_contract

    actor = tiny_brain()
    actor.set_motor_only()
    env = FlyBatch(3, 3, 14, preset="wing_position")
    monkeypatch.setattr(
        probe,
        "load_actor",
        lambda *a: (actor, {"physical_contract": physical_contract(env.model)}),
    )
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"test")
    report = probe.run(
        SimpleNamespace(
            resume=parent,
            graph=tmp_path,
            output=tmp_path / "probe",
            device="cpu",
            noise=[0, 0.01],
            seeds=[3],
            seconds=0.006,
        )
    )
    assert (
        report["weights_unchanged"]
        and report["optimizer_updates"] == report["live_resets"] == 0
    )
    assert report["transitions"] == 18 and len(report["results"]) == 6
    capture = np.load(tmp_path / "probe/seed_3.npz")
    np.testing.assert_array_equal(capture["initial_qpos"][:3], capture["initial_qpos"][3:])
    np.testing.assert_array_equal(capture["action"][:, :3], capture["mean_action"][:, :3])
    assert np.any(capture["action"][:, 3:] != capture["mean_action"][:, 3:])
