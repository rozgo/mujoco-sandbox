"""Physical flight, independently checked fluid dynamics, and causal forecasts."""

import mujoco
import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("neuralop")

from sixlegs.scene import ROOT
from sixlegs.wind.flow import DOMAIN, SpectralFlow, initial_fields, sample
from sixlegs.wind.operator import load
from sixlegs.wind.scene import DRONE_MASS, LOAD_MASS, load_scene
from sixlegs.wind.simulation import Simulation, run
from sixlegs.wind.weather import Weather


def test_reference_solver_exact_vortex_decay_and_divergence():
    solver = SpectralFlow(64, [900])
    axis = np.arange(64) * 2 * np.pi / 64
    x, y = np.meshgrid(axis, axis)
    initial = np.cos(x) * np.cos(y)
    solver.w_hat = np.fft.rfft2(initial[None])
    solver.f_hat[:] = 0
    solver.mean[:] = 0
    result = solver.advance(1.0)[0]
    exact = initial * np.exp(-solver.viscosity * 2 * (2 * np.pi / DOMAIN) ** 2)
    assert np.max(abs(result - exact)) < 1e-6
    u, v = solver.velocity()[0]
    divergence = np.fft.irfft2(
        1j * solver.kx * np.fft.rfft2(u) + 1j * solver.ky * np.fft.rfft2(v), s=(64, 64)
    )
    assert np.max(abs(divergence)) < 2e-6
    coarse, _, _ = initial_fields([900], 64)
    fine, _, _ = initial_fields([900], 128)
    assert np.allclose(coarse, fine[:, ::2, ::2], atol=1e-6)


def test_scene_and_force_causality():
    m, d = load_scene()
    assert m.body("drone").mass == pytest.approx(DRONE_MASS)
    assert m.body("payload").mass == pytest.approx(LOAD_MASS)
    assert m.nu == 4 and np.allclose(m.actuator_forcerange, [0, 10])
    assert d.ten_length[0] == pytest.approx(0.65)
    sim = Simulation()
    before = sim.data.qpos.copy()
    sim.apply_wind()
    assert np.array_equal(before, sim.data.qpos)
    assert not sim.data.qfrc_applied.any()
    sim.data.qvel[0] = 1.0
    mujoco.mj_forward(sim.model, sim.data)
    sim.apply_wind()
    assert sim.drag_power <= 0 and sim.forces[:4, 0].sum() < 0


def test_calm_delivery_is_complete_without_learned_assistance():
    report = run()
    assert report["success"], report
    assert report["peak_swing_deg"] < 2.0
    assert report["payload_tracking_rmse_m"] < 0.01
    assert report["max_rotor_thrust_n"] <= 10.0
    assert report["max_cable_extension_m"] < 0.002
    assert not any(report["warnings"])


def test_committed_operator_uses_same_weights_on_new_grid():
    import torch

    torch.set_num_threads(4)
    model, meta = load(ROOT / "assets/wind/pino.pt")
    assert meta["training_resolution"] == 64 and meta["physics_resolution"] == 128
    for n in (64, 128, 256):
        w, f, mean = initial_fields([900], n)
        with torch.no_grad():
            out = model(torch.tensor(w), torch.tensor(f), torch.tensor(mean)).numpy()
        assert out.shape == (1, n, n) and np.isfinite(out).all()
        assert abs(out.mean() - w.mean()) < 1e-5
    assert not set(meta["train_seeds"]) & set(meta["validation_seeds"])
    assert not set(range(300, 306)) & set(
        meta["train_seeds"] + meta["validation_seeds"]
    )


def test_spectral_inverse_consistent_with_cpu_and_differentiable():
    import torch

    from sixlegs.wind.operator import device

    dev = device()
    cpu, _ = load(ROOT / "assets/wind/pino.pt")
    accelerated, _ = load(ROOT / "assets/wind/pino.pt", dev)
    w, f, mean = [torch.tensor(a) for a in initial_fields([900], 128)]
    assert cpu.fno.fno_blocks.convs[0].enforce_hermitian_symmetry
    with torch.no_grad():
        expected = cpu(w, f, mean)
    a = w.to(dev).requires_grad_()
    actual = accelerated(a, f.to(dev), mean.to(dev))
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)
    actual.square().mean().backward()
    assert torch.isfinite(a.grad).all()


def test_frozen_and_learned_forecast_have_no_reference_future_access(tmp_path):
    n = 8
    rng = np.random.default_rng(0)
    v = rng.normal(size=(31, 2, n, n)).astype(np.float32)
    omega = np.zeros((7, n, n), dtype=np.float32)
    np.savez(
        tmp_path / "reference.npz", velocity=v, omega=omega, mean=np.zeros(2), seed=900
    )
    predictions = rng.normal(size=(7, 11, 2, n, n)).astype(np.float32)
    np.save(tmp_path / "pino_forecast.npy", predictions)
    points = np.array([[0.1, 0.1], [-0.2, 0.3]])
    leads = np.array([0.3, 1.0])
    for kind in ("persistence", "pino"):
        weather = Weather(tmp_path, kind)
        before = weather.forecast(0.30, leads, points)
        weather.velocity[6:] = 99999.0
        weather.omega[2:] = 99999.0
        after = weather.forecast(0.30, leads, points)
        assert np.array_equal(before, after)
    assert sample(np.zeros((2, 8, 8)), points).shape == (2, 2)


def test_held_out_paired_flights(tmp_path):
    folder = ROOT / "outputs/wind/weather-300"
    if not (folder / "weather.json").exists():
        pytest.skip("Run wind-demo prepare --seed 300 for physical integration test")
    results = {
        kind: run(Weather(folder, kind), tmp_path / kind)
        for kind in ("persistence", "pino")
    }
    for r in results.values():
        assert r["success"], r
        assert not r["gate_collisions"] and not any(r["warnings"])
        assert r["max_rotor_thrust_n"] <= 10.0
        assert r["max_cable_extension_m"] < 0.003
    assert (
        results["pino"]["payload_tracking_rmse_m"]
        < results["persistence"]["payload_tracking_rmse_m"]
    )
