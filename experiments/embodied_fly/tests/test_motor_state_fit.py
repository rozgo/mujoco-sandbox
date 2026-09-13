import numpy as np
import pytest

from embodied_fly.motor_state_fit import fit_delta


def test_weighted_readout_recovers_known_mapping_and_regularization_limits_changes():
    rng = np.random.default_rng(82)
    x = rng.normal(size=(1000, 12))
    true = rng.normal(size=(13, 6))
    y = np.column_stack((x, np.ones(len(x)))) @ true
    weights = rng.uniform(1, 8, len(x))
    result = fit_delta(x, y, weights, 1e-9)
    np.testing.assert_allclose(result, true, atol=1e-7)
    limited = fit_delta(x, y, weights, 10)
    assert np.linalg.norm(limited) < np.linalg.norm(result)
    for bad in (0, -1, np.inf):
        with pytest.raises(ValueError):
            fit_delta(x, y, weights, bad)
    weights[0] = 0
    with pytest.raises(ValueError):
        fit_delta(x, y, weights, 0.1)


def test_full_fit_keeps_upstream_and_nonwing_weights_unchanged(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import torch
    from test_motor_focus import tiny_brain

    from embodied_fly import motor_state_fit
    from embodied_fly.provenance import sha256

    actor = tiny_brain().eval()
    actor.set_motor_only()
    parent = {
        "state_dict": actor.state_dict(),
        "motor_only": True,
        "physical_contract": {"test": True},
        "observation_size": 397,
        "sensor_extension_size": 14,
        "action_size": 78,
        "config": {"internal_steps": 4},
        "graph_sha256": "test",
        "graph_metadata_sha256": "test",
    }
    resume = tmp_path / "parent.pt"
    torch.save(parent, resume)
    ground, hover = tmp_path / "ground", tmp_path / "hover"
    ground.mkdir()
    hover.mkdir()
    (ground / "report.json").write_text(json.dumps({"checkpoint_sha256": sha256(resume)}))
    (hover / "report.json").write_text(
        json.dumps({"teacher_present": True, "hover_reference": "state"})
    )
    monkeypatch.setattr(motor_state_fit, "load_actor", lambda *a: (actor, parent))
    rng = np.random.default_rng(4)
    observation = rng.normal(size=(2500, 397)).astype(np.float32)
    action = np.zeros((2500, 78), np.float32)
    action[:, 14:20] = 0.2
    monkeypatch.setattr(
        motor_state_fit,
        "read_capture",
        lambda root, case, contract: (observation, action, {"case": case}),
    )
    output = tmp_path / "fit"
    motor_state_fit.train(
        SimpleNamespace(
            resume=resume,
            graph=tmp_path,
            ground=ground,
            hover=hover,
            output=output,
            device="cpu",
        )
    )
    result = torch.load(output / "actor.pt", weights_only=True)["state_dict"]
    for k, value in parent["state_dict"].items():
        if k in ("motor_decoder.3.weight", "motor_decoder.3.bias"):
            other = np.r_[0:14, 20:78]
            assert torch.equal(value[other], result[k][other])
        else:
            assert torch.equal(value, result[k])
    assert not actor.motor_decoder[3]._forward_hooks
    assert not torch.equal(
        parent["state_dict"]["motor_decoder.3.weight"], result["motor_decoder.3.weight"]
    )
    report = json.loads((output / "report.json").read_text())
    assert report["physical_transitions_collected"] == 0
    assert report["training_frames"] == 6000
    assert report["validation_frames"] == 1500
