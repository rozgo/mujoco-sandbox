import math

import pytest
import torch

from embodied_fly.ppo_step import bounded_step
from embodied_fly.velocity_hover_ppo import baseline_capture_label, policy_update_exceeded


def test_reused_parent_capture_follows_checkpoint_not_latest_file():
    baseline = {
        "checkpoint_sha256": "final",
        "parent_checkpoint_sha256": "parent",
        "snapshots": [{"file": "midpoint_actor.pt", "sha256": "mid"}],
        "evaluations": {"midpoint": {}},
    }
    assert baseline_capture_label(baseline, "mid") == "midpoint"
    assert baseline_capture_label(baseline, "final") == "final"
    assert baseline_capture_label(baseline, "parent") == "parent"
    with pytest.raises(ValueError):
        baseline_capture_label(baseline, "unrelated")


def test_smaller_kl_limit_applies_to_every_stop_and_adam_acceptance():
    assert not policy_update_exceeded(0.006, 0.006, 0.02)
    assert policy_update_exceeded(0.006, 0.001, 0.005)
    assert policy_update_exceeded(0.001, 0.004, 0.005)
    assert not policy_update_exceeded(0.001, 0.003, 0.005)
    for value in (math.nan, math.inf):
        assert policy_update_exceeded(value, 0, 0.005)
        assert policy_update_exceeded(0, value, 0.005)
    outcomes = []
    for cap in (0.02, 0.005):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=0.1)
        p.grad = torch.ones_like(p)
        check = bounded_step(opt, [p], lambda p=p: float(p.detach().square()) * 100, limit=cap)
        assert check["accepted"] and check["accepted_kl"] <= cap
        assert opt.state[p]["step"].item() == 1
        outcomes.append(abs(p.item()))
    assert outcomes[1] < outcomes[0]


@pytest.mark.parametrize("target", [0, -1, math.inf, math.nan])
def test_bad_kl_config_fails_before_training(target):
    with pytest.raises(ValueError):
        policy_update_exceeded(0, 0, target)
