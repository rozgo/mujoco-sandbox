import json

import numpy as np
import pytest

from embodied_fly.velocity_exercise import DURATION, STAGES, command_at
from embodied_fly.velocity_imitation import sample_windows, validate_resume_recipe
from embodied_fly.velocity_recovery import exercise_order, stage_windows
from embodied_fly.velocity_student import completed_stage_metrics


def test_varied_orders_keep_every_skill_brake_and_safe_vertical_path():
    orders = [exercise_order(141000 + i) for i in range(10)]
    assert len({tuple(o) for o in orders}) == 10
    for order in orders:
        assert sorted(order) == list(range(50)) and order[0] == 0 and order[-1] == 49
        height = 0
        for position, stage in enumerate(order):
            _name, duration, command = STAGES[stage]
            if any(command):
                assert order[position + 1] == stage + 1
            height += command[2] * duration
            assert height >= -1e-10
        assert abs(height) < 1e-9
        assert sum(STAGES[s][1] for s in order) == pytest.approx(DURATION)
        json.dumps(order)


def test_ordered_sampler_really_covers_semantic_stages_without_split_leakage():
    orders = [exercise_order(i) for i in range(20)]
    windows = stage_windows(orders)
    worlds, indices, valid, stages = sample_windows(
        np.random.default_rng(5),
        list(range(8)) + list(range(10, 18)),
        64,
        128,
        64,
        35600,
        8,
        windows,
    )
    assert set(stages[:56]) == set(range(50))
    assert not set(worlds) & {8, 9, 18, 19}
    for world, index, stage in zip(worlds[:56], indices[64, :56], stages[:56], strict=True):
        assert command_at(index * 0.002, 1.5, 4.5, orders[world])[1] == stage
    assert not valid[:64, -8:].any()
    assert (indices[64, -8:] == 0).all()


def test_dataset_transition_requires_explicit_flag_and_keeps_optimizer_recipe():
    old = {"lr": 1e-4, "cold_starts": 8, "dataset_report_sha256": "old"}
    new = dict(old, dataset_report_sha256="new")
    with pytest.raises(ValueError):
        validate_resume_recipe(old, new, True)
    assert validate_resume_recipe(old, new, allow_dataset_change=True)
    with pytest.raises(ValueError):
        validate_resume_recipe(old, dict(new, lr=2e-4), True, True)


def test_ordered_scoring_keeps_stage_names_and_failure_windows_matched():
    order = exercise_order(12)
    times = np.arange(35600) * 0.002
    commands = np.array([command_at(t, 1.5, 4.5, order)[0] for t in times])
    arrays = {
        "time": times,
        "command": commands,
        "measured_velocity": commands[:, :3],
        "yaw_rate": commands[:, 3],
    }
    scored = completed_stage_metrics(arrays, order=order)
    assert [s["stage"] for s in scored] == order
    assert all(s["passed"] for s in scored)
    assert [s["stage"] for s in completed_stage_metrics(arrays, 1.9, order)] == []
    assert [s["stage"] for s in completed_stage_metrics(arrays, 2.1, order)] == [0]
    json.dumps(scored)
