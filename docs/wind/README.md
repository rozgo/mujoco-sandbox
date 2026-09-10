# Learning the wind: a physical delivery experiment

A quadrotor carries a suspended package through two gates, lowers it onto a platform, releases the cable after contact, and climbs away. A learned wind forecast improves the same predictive controller's payload tracking. Every flight is driven by an **independent numerical fluid solver**, including the flights using neural forecasts.

[Watch the complete multi-view video](../../previews/wind/comparison.mp4) · [Static scene](../../previews/wind/overview.png) · [Measured results](RESULTS.md) · [Goal brief](BRIEF.md) · [Time log](../TIME_LOG.md)

![Paired physical deliveries](../../previews/wind/comparison.png)

## Run on macOS

```sh
git lfs install
git lfs pull
uv sync --locked --extra wind
uv run --extra wind wind-demo view
```

The first flight prepares numerical weather and causal forecasts from the committed weights. Later launches reuse those files. On Apple Silicon, PyTorch uses Metal automatically; Linux with compatible NVIDIA drivers uses CUDA; CPU also works. The native macOS viewer automatically relaunches through `mjpython` with uv's Python library path.

**Space** pauses. **1** follows the aircraft; **2** shows the course; **3** is the forward onboard camera; **4** looks down at the payload; **5** is overhead. Close the window to exit. `--static` starts paused; `--kind persistence` selects the baseline. These cameras render real scene geometry but are not controller observations.

```sh
# Regenerate static images without simulating motion.
uv run --extra wind wind-demo preview

# One complete physical flight and its machine-readable report.
uv run --extra wind wind-demo run --kind pino --seed 300

# Reproduce all four forecast methods on six held-out winds.
uv run --extra wind wind-demo compare --seed 300 --seeds 6

# Render the saved physical trajectories; run compare first.
uv run --extra wind wind-demo record --seed 300

# Validate forecasts, resolution transfer, and device agreement.
uv run --extra wind wind-demo validate
uv run --extra wind python scripts/check_wind_backend.py
uv run --extra wind pytest -q
```

On a headless Linux GPU machine, set `MUJOCO_GL=egl` for rendering. Training and wind validation require no display. The complete weather cache is several GB and is ignored by Git. The MP4 and model checkpoints are tracked with Git LFS.

## What is physical

MuJoCo integrates the aircraft and parcel as separate free rigid bodies. Four bounded rotor-site actuators generate thrust and reaction torque. A unilateral spatial tendon supplies cable tension and can go slack. The package collides with the two platforms, gates and floor. Release only occurs after contact with the destination and package speed below 0.15 m/s; the tendon is then slackened. No flight code writes body positions, velocities or poses to make the delivery happen. Video rendering replays saved poses from those physical runs.

| Quantity | Value and rationale |
| --- | --- |
| Aircraft mass / inertia | 2.0 kg; diagonal inertia 0.035, 0.035, 0.055 kg m², representing a compact inspection drone |
| Payload | 0.20 kg, 26 × 26 × 20 cm; a light foam parcel with appreciable wind area |
| Rotor centers | Four sites at x/y = ±0.24 m |
| Rotor thrust | 0–10 N each; 40 N maximum against 21.58 N loaded weight |
| Rotor yaw coefficient | Alternating ±0.018 m torque/thrust ratio |
| Available axis moments | Up to ±4.8 N m roll/pitch or ±0.36 N m yaw individually; these maxima are not simultaneously available |
| Rotor response | First-order lag, 0.03 s |
| Cable | 0.65 m attachment-to-attachment; nominal 0.75 m from aircraft hook to package COM |
| Joints and limits | Two free joints; no artificial position limits. Unilateral cable upper length 0.65 m; no compressive cable force |
| Contact | Friction 0.8, 0.01, 0.001; compliant MuJoCo contact, not a welded parcel |
| Integration | Standard CPU MuJoCo, 500 Hz, `implicitfast`; gravity 9.81 m/s² |
| Control rates | Attitude 100 Hz; predictive controller 10 Hz, 2 s horizon |
| Route | 5.4 m transfer; cruise aircraft height 1.90 m; two gates; 43 s complete flight |

The chassis, supports and rotors are geometric primitives. Rotor disks and blades are visual approximations; thrust comes from site actuators. This is a generic quadrotor, not a vendor-calibrated aircraft.

Wind drag at each rotor location and at the parcel COM is

\[
F = c\,\lVert u(x,t)-v_{point}\rVert\,[u(x,t)-v_{point}].
\]

Total aircraft coefficient is **0.14 kg/m**, divided across four sites; parcel coefficient is **0.045 kg/m**, consistent with a light, bluff package. Air velocity is horizontal; body-relative motion includes the actual point velocity. `mj_applyFT` adds these external forces and their moments to `qfrc_applied`. The drag law is dissipative relative to the air; moving wind can do positive work on the aircraft.

## The numerical and learned wind

The reference solves the periodic, forced, incompressible 2D vorticity equation on a 12 × 12 m domain:

\[
\partial_t\omega+u\partial_x\omega+v\partial_y\omega
=0.04\nabla^2\omega+f(x,y).
\]

A streamfunction recovers divergence-free velocity plus a constant mean flow. The solver uses Fourier differentiation, 2/3 dealiasing and third-order SSP Runge–Kutta. Maximum internal step is 0.01 s at 64², 0.005 s at 128² and 0.0025 s at 256². Tests check exact Taylor–Green vortex decay and numerical divergence independently of the learned model.

Each seed generates a different combination of 12 low Fourier modes, initial vorticity RMS 2 s⁻¹, random phases and steady forcing. Mean x velocity is drawn from −0.4 to 0.4 m/s and mean y velocity from 1.4 to 2.2 m/s. The evolving vortices add spatially and temporally changing crosswinds.

The predictor is the official NeuralOperator **FNO**, with four Fourier layers, width 32 and `n_modes=(16,16)` (603,041 parameter elements, including complex spectral weights). Library convention keeps 16 signed y frequencies and 9 nonnegative x frequencies. Inputs are current vorticity, steady forcing and both mean-velocity components. Output is a mean-preserving residual update for the next 0.25 s. The same weights accept different grids.

| Data partition | Seeds | Use |
| --- | --- | --- |
| Training | 0–47 | 48 trajectories, 64 steps each, 3,072 supervised pairs on 64²; 0–16 s |
| Checkpoint selection | 100–107 | Independent trajectories; one-step error |
| Flow evaluation | 200–203 | 64²/128² at 2, 12 and 28 s; 256² at 12 s; leads 0.25, 1 and 2 s |
| Flight development | 200 | Controller development before held-out flight evaluation |
| Flight evaluation | 300–305 | All four controllers on all six winds, 43 s each |

The data-only FNO and PINO receive **64 epochs each**, the same initialization, supervised data, minibatch order and learning-rate schedule. Both use normalized vorticity and velocity errors. PINO adds a 128² PDE residual during epochs 33–64, for four collocation inputs every fourth minibatch, with coefficient 0.03. Those inputs are Fourier-interpolated from coarse data; there are **no 128² target labels**. The residual uses a trapezoidal time discretization, so it is approximate physics regularization, not an exact continuous-time guarantee.

128² tests demonstrate transfer from coarse supervised data, but that grid is used by PINO's physics loss. **256² is unseen by both losses.** The direct 128² rollout is also compared with Fourier interpolation of a 64² rollout. Similar errors do not establish recovery of new fine-scale detail. Resolution-compatible weights are not a promise of arbitrary-resolution accuracy.

### Reproduce training

```sh
uv run --extra wind wind-demo dataset
uv run --extra wind python scripts/train_wind.py --device auto
uv run --extra wind wind-demo validate --models outputs/wind/model
```

Final weights were trained on an RTX 4090, then evaluated and used for physical flights on an Apple M3 Max. The final matched training runs took 73.4 s for FNO and 82.3 s for PINO, about 2 minutes 36 seconds combined, excluding setup, data generation and validation. Both training branches, losses and selection epochs are recorded in the adjacent JSON files. Different devices can produce different optimization trajectories; committed checkpoints make the evaluation reproducible without retraining.

**Portability fix:** released NeuralOperator 2.0.0 produced different 128² inverse-FFT outputs on CUDA and Metal/CPU. The dependency is pinned to official upstream commit `00b7d86f8d74ff0af55da53eb585fe26df9c71f0`, which explicitly enforces Hermitian symmetry before the real inverse FFT. Models were retrained after the fix; TF32 is disabled. The backend check compares 2 s autoregressive outputs on 64², 128² and 256², and a regression test also checks finite gradients. [Upstream implementation and explanation](https://github.com/neuraloperator/neuraloperator/blob/00b7d86f8d74ff0af55da53eb585fe26df9c71f0/neuralop/layers/spectral_convolution.py).

## A fair control comparison

Every controller observes the same **current full wind field every 0.25 s**, plus exact aircraft and package state. This is privileged simulator state, not perception from cameras. The reference path, controller matrices, gains, actuator caps, initial conditions and actual numerical wind are identical within each seed.

- **Persistence:** hold the last observed spatial wind field fixed over the prediction horizon.
- **FNO:** advance that observation with the data-only operator.
- **PINO:** advance it with the physics-regularized operator.
- **True future:** query numerical future wind as a diagnostic. This is unavailable to the learned controllers and is not a deployable method or a mathematical optimum.

The common controller uses a linearized planar aircraft/sling model with position, speed, swing, swing rate and horizontal-force lag. A finite-horizon quadratic objective penalizes payload error, aircraft error, swing, effort and slew. It solves the unconstrained quadratic problem, then clips horizontal force to ±6 N per axis. A geometric attitude controller converts force to orientation and rotor commands. This is an approximate predictive controller, **not** a fully constrained nonlinear MPC or autonomous route planner.

Forecasts are prepared ahead of the flight for efficient repeated evaluation. Each cache entry depends only on the observation at its timestamp, known forcing and model weights. Future reference frames are never inputs to FNO/PINO. A test corrupts the future reference arrays and verifies the controllers' forecasts are unchanged. File metadata hashes the checkpoints; changed weights invalidate the weather cache. Flight reports preserve those hashes, and recording rejects mismatched flight/weather provenance. Cached playback performance is not reported as live neural inference speed.

## Video and evidence

The 1080p video shows the full 43-second flight at real time, with introductory/final holds and a results card. The main views follow both physical runs. An inset cycles through the course overview, downward payload camera, forward onboard camera, overhead camera and parcel contact close-up. Wind tracers follow the reference velocity; arrows represent applied forces and colored paths show controller predictions.

The wind panel compares a one-second PINO forecast with numerical future truth for evaluation only. Vorticity color range is −4 to +4 s⁻¹; velocity-error color range is 0–0.8 m/s. The plot shows the whole flight, while the reported crossing RMSE uses **t = 6–32 s**. Results include every evaluation seed, not just the first seed shown in the movie.

## Limits and intellectual basis

This is **one-way coupling to a horizontal periodic background field**. The gates and aircraft do not modify the flow. There is no resolved rotor wash, obstacle wake, pressure integration, vertical turbulence, real aerodynamics calibration, cable mass/aerodynamics, camera perception or collision-aware planning. The 2D flow is a controlled synthetic test family, not a high-Reynolds-number turbulence benchmark. A stronger deployment claim would require richer flow data, partial observations, uncertainty handling and real-system validation.

The conceptual basis is [FNO, Li et al.](https://arxiv.org/abs/2010.08895) and [PINO, Li et al.](https://arxiv.org/abs/2111.03794), implemented with the [official NeuralOperator library](https://github.com/neuraloperator/neuraloperator). This is an independent robotics illustration of learned continuum forecasts alongside a rigid-body solver; it is not an official Anima Anandkumar MuJoCo result or a replication of the supplied research brief's performance claims.
