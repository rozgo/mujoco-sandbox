import json
from types import SimpleNamespace

import numpy as np
import torch

from embodied_fly.ground_readout import (
    balanced_weights,
    calibrated_checkpoint,
    collect,
    fit,
    ridge_delta,
    split_worlds,
    variant_targets,
)


def test_balancing_and_ridge_fit_recover_known_map_without_validation_statistics():
    torch.manual_seed(351)
    x = torch.randn(300, 8, dtype=torch.float64)
    coef = torch.randn(8, 6, dtype=torch.float64) * 0.05
    y = (x @ coef).tanh()
    task = torch.arange(300) % 3
    variant = torch.arange(300) % 25
    weight = balanced_weights(task, variant)
    for i in range(3):
        torch.testing.assert_close(
            weight[task == i].sum(), torch.tensor(1 / 3, dtype=torch.float64)
        )
    fitted = ridge_delta(x, y, weight, torch.zeros_like(coef), 1e-8)
    probe = torch.randn(20, 8, dtype=torch.float64) + 2
    torch.testing.assert_close(probe @ fitted, probe @ coef, rtol=1e-6, atol=1e-7)
    a, b = split_worlds(np.arange(12) % 3)
    assert set(a).isdisjoint(b) and set(b) == {9, 10, 11}


def test_targets_and_checkpoint_keep_physics_context_and_non_wing_parameters():
    q, v = np.zeros((2, 6)), np.zeros((2, 6))
    target = variant_targets(q, v, np.zeros(6), 0.2, 5)
    assert not target[:, 0].any()
    np.testing.assert_allclose(target[:, 1, 0], -0.2 * 0.02 / 0.03)
    np.testing.assert_allclose(target[:, 13, 0], -5 * 0.00015 / 0.03)
    assert not q.any() and not v.any()
    parent = {
        "motor_only": True,
        "physical_contract": {"sha256": "body"},
        "optimizer_state_dict": {"stale": True},
        "config": {"internal_steps": 4},
        "state_dict": {
            "motor_decoder.3.weight": torch.randn(78, 256),
            "motor_decoder.3.bias": torch.randn(78),
            "core.bias": torch.randn(7),
        },
    }
    wings = np.arange(14, 20)
    fitted = calibrated_checkpoint(parent, torch.zeros(6, 256), torch.zeros(6), wings)
    assert fitted["physical_contract"] == parent["physical_contract"]
    assert fitted["motor_only"] and "optimizer_state_dict" not in fitted
    torch.testing.assert_close(
        fitted["state_dict"]["core.bias"], parent["state_dict"]["core.bias"], rtol=0, atol=0
    )
    for k in ("motor_decoder.3.weight", "motor_decoder.3.bias"):
        torch.testing.assert_close(
            fitted["state_dict"][k][:14], parent["state_dict"][k][:14], rtol=0, atol=0
        )


def test_canonical_collection_and_fit_preserve_causal_capture_and_motor_contract(
    tmp_path, monkeypatch
):
    from test_motor_focus import tiny_brain

    from embodied_fly import ground_readout
    from embodied_fly.batch import FlyBatch
    from embodied_fly.physical_contract import physical_contract
    from embodied_fly.provenance import sha256

    actor = tiny_brain()
    actor.set_motor_only()
    contract = physical_contract(FlyBatch(6, 2, 14, preset="wing_motion").model)
    parent = {
        "motor_only": True,
        "observation_size": 397,
        "sensor_extension_size": 14,
        "action_size": 78,
        "physical_contract": contract,
        "config": {"internal_steps": 4, "ground_posture": True},
        "state_dict": actor.state_dict(),
    }
    checkpoint = tmp_path / "parent.pt"
    torch.save(parent, checkpoint)
    monkeypatch.setattr(ground_readout, "load_actor", lambda *args: (actor, parent))
    args = SimpleNamespace(
        checkpoint=checkpoint,
        graph=tmp_path,
        cache=tmp_path / "corpus",
        output=tmp_path / "corpus",
        device="cpu",
        worlds=6,
        threads=2,
        seconds=0.008,
        stride=2,
        seed=73009,
        angle_delta=0.2,
        speed_delta=5,
        alphas=[0.01, 0.001],
    )
    report = collect(args)
    assert report["physical_transitions"] == 24 and report["synthetic_examples"] == 192
    with np.load(args.output / "capture.npz") as c:
        np.testing.assert_array_equal(c["observation"][1:, :, 297:375], c["action"][:-1])
        assert c["qpos"].shape[:2] == (4, 6)
    args.output = tmp_path / "fitted"
    fitted = fit(args)
    new = torch.load(args.output / "actor.pt", weights_only=True)
    assert new["motor_only"] and new["physical_contract"] == contract
    assert new["config"]["ground_posture"]
    assert fitted["new_deployed_parameters"] == 0
    assert fitted["checkpoint_sha256"] == sha256(args.output / "actor.pt")
    assert json.loads((args.cache / "report.json").read_text())["training_updates"] == 0
