from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy import sparse

from embodied_fly.body import FlyEnvironment
from embodied_fly.brain import EmbodiedBrain
from embodied_fly.motor_interventions import intervention_masks
from embodied_fly.physical_contract import physical_contract


def test_interventions_keep_named_groups_separate_and_feed_back_executed_actions(
    tmp_path, monkeypatch
):
    from embodied_fly import motor_interventions

    native = FlyEnvironment("wing_motion")
    masks = intervention_masks(SimpleNamespace(template=native, model=native.model))
    assert [int(m.sum()) for m in masks.values()] == [0, 6, 19, 6, 48, 54, 78]
    assert not (masks["reference_leg_joints"] & masks["reference_foot_adhesion"]).any()
    graph = sparse.eye(6, format="csr", dtype=np.float32)
    actor = EmbodiedBrain(
        graph, [0, 1], [2, 3], [4, 5], 397, 78, sensor_extension_size=14, motor_only=True
    ).eval()
    monkeypatch.setattr(
        motor_interventions,
        "load_actor",
        lambda *a: (actor, {"physical_contract": physical_contract(native.model)}),
    )
    checkpoint = tmp_path / "actor.pt"
    checkpoint.write_bytes(b"tiny test actor")
    args = SimpleNamespace(
        checkpoint=checkpoint,
        graph=tmp_path,
        output=tmp_path / "run",
        teacher=Path(__file__).resolve().parents[3]
        / "assets/embodied_fly/teachers/walking.npz",
        seconds=0.004,
        device="cpu",
    )
    r = motor_interventions.run(args)
    assert not r["policy_acceptance_eligible"] and r["training_seconds"] == 0
    assert len(r["results"]) == 14 and r["warning_count"] == 0
    for item in r["results"]:
        with np.load(args.output / (item["case"] + ".npz")) as d:
            mask = masks[item["intervention"]]
            np.testing.assert_array_equal(
                d["action"], np.where(mask, d["reference_action"], d["student_action"])
            )
            np.testing.assert_array_equal(d["observation"][1:, 297:375], d["action"][:-1])
