from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from embodied_fly.body import FlyEnvironment
from embodied_fly import flight_collect


def test_corrective_capture_records_executed_physics_and_causal_previous_action(
    tmp_path, monkeypatch
):
    """A small deterministic test actor checks collection, not learned behavior."""

    class Actor:
        sensor_extension_size = 6

        def initial_state(self, worlds):
            return torch.zeros(1, worlds)

        def __call__(self, obs, state, time_scale):
            assert obs.shape == (1, 389) and time_scale == 0.1
            # Continuously varying actions also expose memory resets inside episodes.
            action = (state.T * 0.01).expand(1, 78).clone()
            return SimpleNamespace(action=action, state=state + 1)

    monkeypatch.setattr(flight_collect, "load_actor", lambda *a: (Actor(), {}))
    fake = tmp_path / "actor.pt"
    fake.write_bytes(b"test actor identity; not a deployable checkpoint")
    (tmp_path / "weights.npz").write_bytes(b"test graph identity")
    assets = Path(__file__).resolve().parents[3] / "assets/embodied_fly/teachers"
    args = Namespace(
        output=tmp_path / "capture",
        teacher=assets / "flight.npz",
        wing_pattern=assets / "wing_pattern_fmech.npy",
        student_checkpoint=fake,
        graph=tmp_path,
        student_fraction=0.15,
        device="cpu",
        seed=46001,
        seconds=0.002,
        episodes=4,
    )
    manifest = flight_collect.collect(args)
    assert manifest["student_present"] and not manifest["policy_acceptance_eligible"]
    env = FlyEnvironment("flight")
    for episode in range(4):
        with np.load(args.output / f"episode_{episode:03d}.npz") as capture:
            actions = capture["executed_action"]
            np.testing.assert_allclose(
                actions,
                0.15 * capture["student_action"] + 0.85 * capture["action"],
                rtol=0,
                atol=1e-7,
            )
            np.testing.assert_allclose(capture["student_action"][:, 0], np.arange(10) * 0.01)
            # Original sensor schema ends with previous action, command, five needs.
            np.testing.assert_array_equal(capture["observation"][1:, -86:-8], actions[:-1])
            env.reset()
            env.data.qpos[:] = capture["qpos"][0]
            env.data.qvel[:] = capture["qvel"][0]
            env.data.act[:] = capture["activation"][0]
            env.data.ctrl[:] = capture["ctrl"][0]
            mujoco.mj_forward(env.model, env.data)
            for i, action in enumerate(actions[:-1]):
                env.step(action)
                np.testing.assert_allclose(env.data.qpos, capture["qpos"][i + 1], atol=1e-12)
                assert not env.data.xfrc_applied.any() and not env.data.qfrc_applied.any()
