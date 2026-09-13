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


@pytest.mark.parametrize("with_corrections", [False, True])
@pytest.mark.parametrize("feature_layer", ["hidden", "motor"])
def test_full_fit_keeps_upstream_and_nonwing_weights_unchanged(
    tmp_path, monkeypatch, with_corrections, feature_layer
):
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
    monkeypatch.setattr(
        motor_state_fit,
        "correction_capture",
        lambda *args: (
            observation,
            action,
            {"case": "hover_correction", "retained_pre_failure_frames": 125},
        ),
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
            correction_capture=[tmp_path / "correction"] if with_corrections else [],
            startup_weight=4.0 if with_corrections else 1.0,
            feature_layer=feature_layer,
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
    if feature_layer == "hidden":
        assert not torch.equal(
            parent["state_dict"]["motor_decoder.3.weight"], result["motor_decoder.3.weight"]
        )
    else:
        assert all(torch.equal(v, result[k]) for k, v in parent["state_dict"].items())
        assert result["wing_residual.weight"].abs().sum() > 0
        loaded = tiny_brain().eval()
        loaded.set_motor_only()
        loaded.enable_wing_residual()
        loaded.load_state_dict(result, strict=True)
        assert torch.isfinite(
            loaded(torch.as_tensor(observation[:1]), loaded.initial_state(1)).action
        ).all()
    report = json.loads((output / "report.json").read_text())
    assert report["physical_transitions_collected"] == 0
    assert report["training_frames"] == (6105 if with_corrections else 6000)
    assert report["validation_frames"] == (1520 if with_corrections else 1500)


def test_correction_masks_retain_near_failure_training_examples_and_exclude_post_fall():
    from embodied_fly.motor_state_fit import history_masks

    train, valid = history_masks(2500, [2500, 2500, 2500, 115])
    assert not (train & valid).any()
    assert train[100:115, 3].all()
    assert valid[90:100, 3].all()
    assert not train[115:, 3].any() and not valid[115:, 3].any()
    assert np.sum(train[:, 3] | valid[:, 3]) == 115


def test_current_state_corrections_reverse_at_wing_limit_and_stop_before_body_failure(
    tmp_path,
):
    import json

    import mujoco

    from embodied_fly.batch import FlyBatch
    from embodied_fly.motor_focus import MotorTasks
    from embodied_fly.motor_state_fit import correction_capture
    from embodied_fly.physical_contract import physical_contract
    from embodied_fly.provenance import sha256

    env = FlyBatch(1, 1, 14, preset="wing_motion")
    tasks = MotorTasks(env, 501)
    tasks.task_ids[:] = 2
    tasks.reset(np.array([0]))
    state = {k: env.fields[k].copy() for k in ("qpos", "qvel", "act", "ctrl")}
    captured = {
        "qpos": np.repeat(state["qpos"], 2500, axis=0),
        "qvel": np.repeat(state["qvel"], 2500, axis=0),
        "activation": np.repeat(state["act"], 2500, axis=0),
        "ctrl": np.repeat(state["ctrl"], 2500, axis=0),
        "observation": np.repeat(env.observation(), 2500, axis=0),
        "action": np.zeros((2500, 78), np.float32),
        "requested_height_cm": np.full(2500, 2.0),
    }
    # Initialization-only fixture states: canonical free fly, then sweep-limit
    # position, then an explicitly failed height. No claimed physical rollout.
    state["qpos"][0, env.template.wing_angle_indices[[0, 3]]] = -1.45
    env.reset(np.array([0]), state=state)
    env.requested_height_cm[:] = 2
    captured["qpos"][60:80] = state["qpos"][0]
    captured["observation"][60:80] = env.observation()[0]
    captured["qpos"][80:, 2] = 0.4
    np.savez_compressed(tmp_path / "hover.npz", **captured)
    mujoco.mj_saveModel(env.model, str(tmp_path / "model.mjb"))
    contract = physical_contract(env.model)
    report = {
        "physical_contract": contract,
        "control_hz": 500,
        "model_sha256": sha256(tmp_path / "model.mjb"),
        "teacher_present": False,
        "student_present": True,
        "checkpoint_sha256": "fixture",
        "results": [{"case": "hover", "state_sha256": sha256(tmp_path / "hover.npz")}],
    }
    (tmp_path / "report.json").write_text(json.dumps(report))
    source_hash = sha256(tmp_path / "hover.npz")
    obs, labels, identity = correction_capture(tmp_path, contract)
    assert identity["retained_pre_failure_frames"] == 80
    assert identity["excluded_frames"] == 2420
    assert labels[60, 14] > 0.4 and labels[60, 17] > 0.4
    np.testing.assert_array_equal(obs, captured["observation"])
    assert sha256(tmp_path / "hover.npz") == source_hash
