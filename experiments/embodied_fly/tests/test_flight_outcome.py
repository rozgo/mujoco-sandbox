import json
from types import SimpleNamespace

import numpy as np
import pytest

from embodied_fly.flight_outcome import FlightOutcomeReward, FlightResets


def test_airborne_rewards_distinguish_hover_sink_and_failure_with_physical_time():
    qpos = np.zeros((4, 7))
    qpos[:, 2:4] = 1
    qvel = np.zeros((4, 6))
    env = SimpleNamespace(
        n=4,
        fields={
            "qpos": qpos,
            "qvel": qvel,
            "xmat": np.tile(np.eye(3).reshape(1, 1, 9), (4, 1, 1)),
        },
        command=np.zeros((4, 3)),
        template=SimpleNamespace(thorax_id=0),
        forbidden_peak=np.zeros(4),
        body_weight=1,
        control_dt=0.0002,
    )
    reward_fn = FlightOutcomeReward(env)
    reward_fn.reset(np.arange(4))
    qvel[1, 2] = -10  # descending but still within the height gate
    qpos[2, 2] = 0.7  # physical termination, penalized once before environment reset
    env.fields["xmat"][3, 0, 8] = -0.1
    reward, done, terms = reward_fn(np.zeros((4, 78)))
    assert reward[0] == pytest.approx(6.5 * env.control_dt)
    assert reward[0] > reward[1] > reward[2]
    np.testing.assert_array_equal(done, [False, False, True, True])
    assert reward[2] == pytest.approx(
        sum(value[2] for value in terms.values()) * env.control_dt - 1
    )
    assert reward[3] < -0.99


@pytest.mark.parametrize("preset,hz", [("flight", 5000), ("wing_motion", 500)])
def test_airborne_resets_exclude_validation_and_read_only_first_physical_frame(
    tmp_path, preset, hz
):
    manifest = {"control_hz": hz, "physical_preset": preset, "episodes": []}
    for i in range(8):
        qpos = np.zeros((2, 7))
        qpos[0, 2:4] = 1
        qpos[1] = np.nan
        observation = np.zeros((2, 383))
        observation[0, 375] = i / 10
        np.savez(
            tmp_path / f"episode_{i:03d}.npz",
            qpos=qpos,
            qvel=np.zeros((2, 6)),
            activation=np.zeros((2, 2)),
            ctrl=np.zeros((2, 3)),
            observation=observation,
        )
        manifest["episodes"].append({"episode": i, "failure": None, "final_upright": 1})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    fields = {
        name: np.zeros((4, size))
        for name, size in (("qpos", 7), ("qvel", 6), ("act", 2), ("ctrl", 3))
    }
    calls = []

    def reset(ids, *, state):
        calls.append(ids.copy())
        for key, value in state.items():
            fields[key][ids] = value

    env = SimpleNamespace(
        preset=preset,
        n=4,
        control_dt=1 / hz,
        fields=fields,
        command=np.zeros((4, 3)),
        reset=reset,
    )
    resets = FlightResets(env, tmp_path, np.random.default_rng(15))
    assert resets.report["validation_indices_excluded"] == [4, 6]
    assert len(resets.states) == 6
    assert all(np.isfinite(s["qpos"]).all() for s in resets.states)
    assert set(resets.commands[:, 0]) == {0, 1, 2, 3, 5, 7}
    resets.reset([0, 2])
    np.testing.assert_array_equal(fields["qpos"][[0, 2], 2], 1)
    np.testing.assert_array_equal(fields["qpos"][[1, 3]], 0)
    assert len(calls) == 1
    resets.reset([])
    assert len(calls) == 1
    manifest["physical_preset"] = "wing_motion" if preset == "flight" else "flight"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="force model"):
        FlightResets(env, tmp_path, np.random.default_rng(15))
