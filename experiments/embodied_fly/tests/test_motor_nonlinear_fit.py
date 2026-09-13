import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from test_motor_focus import tiny_brain

from embodied_fly.brain import NonlinearWingReadout
from embodied_fly.motor_nonlinear_fit import read_cache, train
from embodied_fly.provenance import sha256


def test_cached_nonlinear_fit_preserves_original_actor_and_exports_exact_predictions(tmp_path):
    torch.manual_seed(53)
    actor = tiny_brain().eval()
    actor.set_motor_only()
    parent = {
        "state_dict": actor.state_dict(),
        "motor_only": True,
        "observation_size": 397,
        "sensor_extension_size": 14,
        "action_size": 78,
        "config": {"internal_steps": 4},
        "graph_sha256": "test",
        "graph_metadata_sha256": "test",
        "physical_contract": {"fixture": True},
    }
    resume = tmp_path / "parent.pt"
    torch.save(parent, resume)
    cache = tmp_path / "cache"
    cache.mkdir()
    rng = np.random.default_rng(54)
    x = rng.normal(size=(180, 3, 2)).astype(np.float32)
    target = np.repeat((0.4 * np.sin(x[..., :1])), 6, axis=-1)
    fit = np.zeros((180, 3), bool)
    fit[:140] = True
    weights = np.tile([8.0, 8.0, 1.0], (180, 1))
    np.savez_compressed(
        cache / "features.npz",
        hidden=x,
        logits=np.zeros_like(target),
        target=target,
        train_mask=fit,
        validation_mask=~fit,
        frame_weights=weights,
    )
    report = {
        "parent_checkpoint_sha256": sha256(resume),
        "physical_contract": parent["physical_contract"],
        "feature_layer": "motor",
        "features_sha256": sha256(cache / "features.npz"),
        "history_weights": [8.0, 8.0, 1.0],
        "training_frames": int(fit.sum()),
        "validation_frames": int((~fit).sum()),
        "sources": [],
        "validation_split": "fixture",
    }
    (cache / "report.json").write_text(json.dumps(report))
    output = tmp_path / "fit"
    train(
        SimpleNamespace(
            resume=resume,
            cache=cache,
            output=output,
            device="cpu",
            seconds=20,
            max_updates=200,
            hidden=16,
            batch_size=128,
            learning_rate=0.01,
            seed=55,
        )
    )
    child = torch.load(output / "actor.pt", weights_only=True)
    for key, value in parent["state_dict"].items():
        assert torch.equal(value, child["state_dict"][key])
    module = NonlinearWingReadout(2, 16)
    module.load_state_dict(
        {
            k.removeprefix("wing_residual."): v
            for k, v in child["state_dict"].items()
            if k.startswith("wing_residual.")
        },
        strict=True,
    )
    np.testing.assert_allclose(module.feature_mean.numpy(), x[fit].mean(axis=0), atol=1e-7)
    with torch.no_grad():
        prediction = module(torch.from_numpy(x)).tanh().numpy()
    np.testing.assert_array_equal(
        prediction, np.load(output / "predictions.npz")["prediction"]
    )
    result = json.loads((output / "report.json").read_text())
    assert result["selected"]["selection_score"] < result["initial"]["selection_score"] * 0.1
    assert result["updates"] == 200 and result["physical_transitions_collected"] == 0
    loaded = tiny_brain().eval()
    loaded.set_motor_only()
    loaded.enable_wing_residual(16)
    loaded.load_state_dict(child["state_dict"], strict=True)
    assert loaded.wing_residual_hidden == 16
    # Exact provenance checks reject a tampered cache before fitting.
    with (cache / "features.npz").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        read_cache(cache, resume, parent)
