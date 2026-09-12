import numpy as np

from embodied_fly.body import CONTROL_DT
from embodied_fly.transitions import summarize_phase


def summary(speed, phase="stop", command=(0, 0, 0), support=0):
    velocity = np.zeros((len(speed), 6))
    velocity[:, 3] = speed
    positions = np.zeros((len(speed), 3))
    positions[:, 0] = np.cumsum(speed) * CONTROL_DT
    return summarize_phase(
        phase,
        command,
        np.zeros(3),
        positions,
        velocity,
        np.ones(len(speed)),
        np.full(len(speed), 0.0012),
        support,
    )


def test_stop_gate_accepts_braking_and_rejects_continued_drift():
    time = np.arange(1000) * CONTROL_DT
    assert summary(np.exp(-time / 0.05))["success"]
    assert not summary(np.full(1000, 0.4))["success"]
    # Stopping late after excessive travel cannot pass by looking only at the tail.
    late_stop = np.zeros(1000)
    late_stop[:100] = 10
    assert not summary(late_stop)["success"]


def test_transition_gate_requires_full_duration_and_permitted_support():
    assert summary(np.ones(1000), "walk", (1, 0, 0))["success"]
    assert not summary(np.ones(500), "walk", (1, 0, 0))["success"]
    assert not summary(np.ones(1000), "walk", (1, 0, 0), support=1)["success"]
