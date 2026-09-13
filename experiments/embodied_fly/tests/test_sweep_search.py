import copy
import json
from types import SimpleNamespace

import numpy as np
import torch
from test_motor_focus import tiny_brain

from embodied_fly.sweep_search import CandidateRows, calibrated_state, select_candidate


def test_batched_search_matches_exported_actor_and_freezes_nonwing_state():
    torch.manual_seed(920)
    parent = tiny_brain().eval()
    parent.set_motor_only()
    rows = [14, 17]
    candidates = [(1, 0), (0.85, -0.02), (1.3, 0.04)]
    originals = {k: v.clone() for k, v in parent.state_dict().items()}
    actors = []
    for gain, bias in candidates:
        child = tiny_brain().eval()
        child.set_motor_only()
        child.load_state_dict(calibrated_state(originals, rows, gain, bias))
        actors.append(child)
    hook = CandidateRows(parent, rows, candidates)
    shared_memory = parent.initial_state(3)
    memories = [a.initial_state(1) for a in actors]
    with torch.no_grad():
        for _ in range(5):
            obs = torch.randn(3, 397)
            shared = parent(obs, shared_memory)
            shared_memory = shared.state
            for i, actor in enumerate(actors):
                single = actor(obs[i : i + 1], memories[i])
                memories[i] = single.state
                torch.testing.assert_close(
                    shared.action[i], single.action[0], atol=1e-6, rtol=1e-6
                )
                torch.testing.assert_close(
                    shared.state[:, i], single.state[:, 0], atol=1e-6, rtol=1e-6
                )
    hook.close()
    for key, value in originals.items():
        assert torch.equal(parent.state_dict()[key], value)
    other = [i for i in range(78) if i not in rows]
    changed = actors[-1].state_dict()
    for key, value in originals.items():
        if key in ("motor_decoder.3.weight", "motor_decoder.3.bias"):
            assert torch.equal(changed[key][other], value[other])
        else:
            assert torch.equal(changed[key], value)


def test_selection_requires_ground_retention_and_meaningful_hover_improvement():
    case = {
        "stable": True,
        "max_forbidden_load": 0.0,
        "wing_rms_rad": 0.05,
        "wing_max_rad": 0.1,
        "speed_rmse_cm_s": 0.1,
        "yaw_rmse_rad_s": 0.1,
        "root_rmse_cm": 1.0,
        "failed_fraction": 0.0,
    }
    baseline = {"gain": 1.0, "bias": 0.0, "cases": [copy.copy(case) for _ in range(4)]}
    better = copy.deepcopy(baseline)
    better["gain"] = 1.15
    for c in better["cases"][2:]:
        c["root_rmse_cm"] = 0.5
    selected, improved = select_candidate([baseline, better])
    assert selected is better and improved
    better["cases"][0]["stable"] = False
    assert select_candidate([baseline, better]) == (baseline, False)
    better["cases"][0]["stable"] = True
    better["cases"][1]["wing_rms_rad"] = 0.5
    assert select_candidate([baseline, better]) == (baseline, False)
    better["cases"][1]["wing_rms_rad"] = 0.05
    for c in better["cases"][2:]:
        c["root_rmse_cm"] = 0.99
    assert select_candidate([baseline, better]) == (baseline, False)
    assert np.isfinite(baseline["hover_cost"])


def test_search_runs_canonical_physics_and_saves_causal_candidate_traces(
    tmp_path, monkeypatch
):
    from embodied_fly import sweep_search
    from embodied_fly.batch import FlyBatch
    from embodied_fly.physical_contract import physical_contract

    actor = tiny_brain().eval()
    actor.set_motor_only()
    env = FlyBatch(1, 1, 14, preset="wing_motion")
    parent = {
        "state_dict": actor.state_dict(),
        "motor_only": True,
        "physical_contract": physical_contract(env.model),
        "observation_size": 397,
        "sensor_extension_size": 14,
        "action_size": 78,
        "config": {"internal_steps": actor.internal_steps},
        "graph_sha256": "test",
        "graph_metadata_sha256": "test",
    }
    resume = tmp_path / "parent.pt"
    torch.save(parent, resume)
    monkeypatch.setattr(sweep_search, "load_actor", lambda *args: (actor, parent))
    output = tmp_path / "run"
    sweep_search.search(
        SimpleNamespace(
            resume=resume,
            output=output,
            graph=tmp_path,
            seconds=0.004,
            seed=2,
            threads=1,
            device="cpu",
        )
    )
    report = json.loads((output / "report.json").read_text())
    assert report["physical_transitions"] == 128
    assert report["physical_contract"] == parent["physical_contract"]
    assert len(report["grid"]) == 16
    assert not actor.motor_decoder[3]._forward_hooks
    for i in range(2):
        with np.load(output / f"batch_{i}.npz") as data:
            np.testing.assert_array_equal(
                data["observation"][1, :, 297:375], data["action"][0]
            )
            for candidate in range(1, 8):
                np.testing.assert_array_equal(
                    data["qpos"][0, :4], data["qpos"][0, candidate * 4 : candidate * 4 + 4]
                )
            assert all(np.isfinite(data[k]).all() for k in data.files)
