from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from embodied_fly import flight_collect
from embodied_fly.body import FlyEnvironment
from embodied_fly.train import load_episodes


@pytest.mark.parametrize(
    "extension_size,warmup,reset_at_release",
    ((6, 0, False), (12, 0, False), (12, 0.0006, False), (12, 0.0006, True)),
)
def test_corrective_capture_records_executed_physics_and_causal_previous_action(
    tmp_path, monkeypatch, extension_size, warmup, reset_at_release
):
    """A small deterministic test actor checks collection, not learned behavior."""

    class Actor:
        sensor_extension_size = extension_size
        observation_size = 383 + extension_size

        def initial_state(self, worlds):
            return torch.zeros(1, worlds)

        def __call__(self, obs, state, time_scale):
            assert obs.shape == (1, 383 + extension_size) and time_scale == 0.1
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
        student_fraction=1.0 if warmup else 0.15,
        teacher_warmup_seconds=warmup,
        reset_memory_at_release=reset_at_release,
        device="cpu",
        seed=46001,
        seconds=0.002,
        episodes=4,
    )
    manifest = flight_collect.collect(args)
    assert manifest["recorded_observation_size"] == 383
    assert manifest["student_observation_size"] == 383 + extension_size
    assert manifest["student_present"] and not manifest["policy_acceptance_eligible"]
    assert manifest["release_diagnostic"] == bool(warmup)
    env = FlyEnvironment("flight")
    for episode in range(4):
        with np.load(args.output / f"episode_{episode:03d}.npz") as capture:
            actions = capture["executed_action"]
            fractions = np.full((len(actions), 1), args.student_fraction)
            fractions[: round(warmup / env.control_dt)] = 0
            np.testing.assert_array_equal(
                capture["executed_student_fraction"], fractions[:, 0]
            )
            np.testing.assert_allclose(
                actions,
                fractions * capture["student_action"] + (1 - fractions) * capture["action"],
                rtol=0,
                atol=1e-7,
            )
            if warmup:
                np.testing.assert_array_equal(actions[:3], capture["action"][:3])
                np.testing.assert_array_equal(actions[3:], capture["student_action"][3:])
                assert manifest["episodes"][episode]["after_teacher_warmup"]["frames"] == 7
            expected = np.arange(10) * 0.01
            if reset_at_release:
                expected[3:] -= 0.03
            np.testing.assert_allclose(capture["student_action"][:, 0], expected, atol=1e-8)
            if warmup:
                np.testing.assert_array_equal(
                    capture["release_neural_state"], 0 if reset_at_release else 3
                )
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
    augmented = load_episodes(args.output, True, True)
    assert len(augmented) == 4
    for i, episode in enumerate(augmented):
        with np.load(args.output / f"episode_{i:03d}.npz") as capture:
            np.testing.assert_array_equal(
                episode["observation"][:, :383], capture["observation"]
            )
            np.testing.assert_allclose(
                episode["observation"][:, 389:395],
                capture["qpos"][:, env.wing_angle_indices] / np.pi,
                rtol=1e-7,
            )
