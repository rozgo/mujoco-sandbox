"""Balance interfaces must preserve physics and the frozen walking conversion."""

import mujoco
import numpy as np
import pytest
import torch
from adaptive_locomotion.bodies import PRESETS, ROOT, build_model, initialize
from adaptive_locomotion.env import OBS_DIM
from adaptive_locomotion.policy import Policy
from adaptive_locomotion.standing_env import StandingEnv
from adaptive_locomotion.standing_surfaces import SURFACES, surface_height
from adaptive_locomotion.train import load_checkpoint, train


def test_support_conversion_preserves_walking_and_initial_balance_actions():
    torch.manual_seed(12)
    net = Policy("blind")
    obs = torch.randn(17, OBS_DIM)
    before, _ = net(obs)
    net.enable_support()
    for support in (None, torch.randn(17, 20)):
        after, _ = net(obs, support=support)
        torch.testing.assert_close(after, before, rtol=1e-5, atol=1e-7)
    assert net.actor[0].in_features == 86
    assert net.critic[0].in_features == 90


@pytest.mark.parametrize("surface", SURFACES)
def test_physical_surface_reset_and_ranges(surface):
    m = build_model(PRESETS["healthy"], "stand_" + surface)
    d = mujoco.MjData(m)
    initialize(m, d)
    assert np.isfinite(d.qpos).all()
    assert not d.ncon or min(d.contact.dist) > -0.001
    assert all(m.sensor(f"standing_range_{i}").dim[0] == 1 for i in range(4))
    for i in range(4):
        site = m.site(f"standing_ray_{i}").id
        # At level orientation, vertical sensor measures physical terrain height.
        origin = d.site_xpos[site]
        ground = float(surface_height(surface, *origin[:2]))
        actual = d.sensor(f"standing_range_{i}").data[0]
        assert actual == pytest.approx(origin[2] - ground, abs=0.003)


def test_idle_reward_has_no_reference_to_forward_progress_or_gait_weights():
    args = {
        "num_envs": 2,
        "seed": 12,
        "randomize": False,
        "schedule": False,
        "cases": [(PRESETS["healthy"], "gap_fr")],
        "threads": 1,
    }
    plain = StandingEnv(**args)
    gait = StandingEnv(
        **args,
        stride_weight=9,
        balance_weight=9,
        body_motion_weight=4,
        visible_step_weight=5,
        reward_profile="walk",
    )
    try:
        for _ in range(5):
            a = plain.action.copy()
            r1 = plain.step(a)[0]
            r2 = gait.step(a)[0]
            np.testing.assert_allclose(r1, r2, atol=1e-6)
        assert plain.support_obs().shape == (2, 20)
        plain.set_commands([0.55, 0, 0])
        assert not plain.support_obs().any()
        plain.hold_anchor[:] = 100
        plain.set_commands([0, 0, 0])
        np.testing.assert_allclose(plain.hold_anchor, plain.pos[:, :2])
    finally:
        plain.close()
        gait.close()


def test_shared_actor_training_smoke(tmp_path):
    baseline = ROOT / "assets/locomotion/checkpoints/adaptive_dog_v1.pt"
    _, parent = load_checkpoint(baseline)
    report = train(
        tmp_path,
        seconds=30,
        num_envs=16,
        mode="support",
        resume=baseline,
        standing_profile="healthy",
        device="cpu",
        epochs=1,
        horizon=4,
        minibatch_size=16,
        max_iterations=1,
        reference=baseline,
        reference_loss_weight=3,
        walking_replay_weight=0.5,
        allowance=parent["cumulative_training_seconds"] + 31,
        extension_reason="Focused one-update interface test.",
    )
    assert report["iterations"] == 1
    assert report["optimizer_steps"] == 4
    trained, saved = load_checkpoint(tmp_path / "policy.pt")
    assert trained.mode == "support"
    assert len(saved["config"]["standing_cases"]) == 8


@pytest.mark.parametrize("surface", ["flat", "gap_fr", "slope_x_18", "steps_20"])
def test_warp_standing_geometry_and_sensors(surface):
    wp = pytest.importorskip("warp")
    pytest.importorskip("mujoco_warp")
    wp.init()
    if not wp.is_cuda_available():
        pytest.skip("NVIDIA CUDA required")
    args = {
        "num_envs": 2,
        "seed": 9300,
        "randomize": False,
        "schedule": False,
        "cases": [(PRESETS["healthy"], surface)],
        "threads": 1,
    }
    cpu = StandingEnv(**args)
    gpu = StandingEnv(**args, physics_backend="warp")
    try:
        np.testing.assert_allclose(cpu.support_obs(), gpu.support_obs(), atol=1e-5)
        for _ in range(4):
            cpu.step(cpu.action)
            gpu.step(gpu.action)
        np.testing.assert_allclose(cpu.pos, gpu.pos, atol=0.003)
        assert np.isfinite(gpu.obs()).all()
        assert np.isfinite(gpu.support_obs()).all()
        assert max(abs(gpu.torque).ravel()) <= 45.4301
    finally:
        cpu.close()
        gpu.close()
