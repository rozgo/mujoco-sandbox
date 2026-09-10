# Version 2: stronger crosswinds and faster delivery

[Watch Version 2](../../../previews/wind/aggressive/comparison.mp4) · [Original video](../../../previews/wind/comparison.mp4) · [Original experiment and physics](../README.md)

The user requested more aggressive dynamics and then explicitly asked to preserve the previous version. **The original MP4, model checkpoints and standard preset are unchanged.** This version uses a separate `aggressive` preset, fresh wind seeds 400–405, and separate output and preview directories.

![Version 2 comparison](../../../previews/wind/aggressive/comparison.png)

## Run either version

```sh
# Original scene, controller and timing.
uv run --extra wind wind-demo view

# Stronger wind and faster physical motion.
uv run --extra wind wind-demo view --profile aggressive

# Reproduce all 24 stress trials, including reported failures.
uv run --extra wind wind-demo compare --profile aggressive

# Render the complete successful seed-400 pair.
uv run --extra wind wind-demo record --profile aggressive
```

The comparison command finishes all four forecast methods on all six seeds and returns a nonzero exit code when any mission fails. Physical failures are saved as reports with termination reasons and partial trajectories; numerical failures still raise errors. A missing or too-short forecast cache is rejected instead of silently shortening the controller's prediction horizon.

## What changed

| Parameter | Original | Version 2 |
| --- | --- | --- |
| Wind velocity and evolution rate | 1× | 1.5× for each underlying flow |
| 5.4 m crossing | 24 s | 12 s |
| Peak commanded cruise speed | 0.3375 m/s | 0.675 m/s |
| Crossing acceleration scale | 1× | 4×, due to the shorter smooth trajectory |
| Crossing metric window | 6–32 s | 6–20 s |
| Complete flight | 43 s | 31 s |
| Video | 62 s | 50 s, with the physical flight at real time |
| Wind seeds | 300–305 | 400–405 |

Masses, cable geometry, contact parameters, controller gains and four 0–10 N rotor limits remain the same. Both sides experience the same wind and follow the same reference trajectory. The main camera is oriented to expose lateral tracking error. Colored trails show the parcel's actual last three seconds of motion, and a white cross shows the horizontal target at the parcel's current height. These are overlays on unmodified physical poses.

## Stronger wind remains consistent with the modeled PDE

For scale s = 1.5, velocity is transformed as **u_new(x,t) = s u_base(x,s t)**. This is a Navier–Stokes similarity transformation with viscosity **nu_new = s nu_base = 0.06 m²/s**, vorticity **omega_new = s omega_base**, and forcing **f_new = s² f_base**. It increases both speed and the rate of gust evolution without replaying the aircraft faster. Reynolds number is preserved; the robot's mass, gravity and actuator time constants stay fixed.

The learned models operate in the original flow coordinates, forecasting three native seconds for each two-second physical controller horizon, then rescale their outputs. No new model training was necessary. Native forecast caches are extended to 3.75 s and the numerical reference to 51 s. The same exact current-field observation is supplied to every controller; the observation clock also scales, to 6 Hz. This is still the simplified, one-way 2D background-flow setup described in the original documentation.

A regression test independently integrates a solver with scaled initial conditions, forcing and viscosity and compares it with the transformed original solution using matched dimensionless integration steps. The initial overly strict comparison using unmatched time steps exposed ordinary RK discretization error; the corrected test isolates the scaling identity and agrees below 1e-5 in vorticity.

## Measured outcome

**Every method achieved 5/6 successful, collision-free deliveries.** Seed 404 exceeds this fixed controller/actuator configuration's capability: the frozen-field run collides and drops below the flight envelope; FNO and PINO drift beyond the lateral envelope during takeoff; the true-future diagnostic eventually places the parcel but contacts a gate, so it also fails the mission criterion. All four encounter rotor saturation. These are physical/control failures, not MuJoCo numerical instability.

The results card explicitly retains the failures. RMSE bars use only the **five shared, complete crossing windows** (400, 401, 402, 403, 405); a run that terminates before the crossing is finished cannot contribute an artificially low full-window error.

| Forecast | Successful missions | Mean XY tracking RMSE, five shared full windows |
| --- | ---: | ---: |
| Frozen field | 5/6 | 19.59 cm |
| FNO | 5/6 | 4.34 cm |
| PINO | 5/6 | 4.51 cm |
| True future diagnostic | 5/6 | 4.47 cm |

PINO reduces error **77.0% on those five complete paired windows**. This is conditional on that subset, not a claim of improved mission success rate or an all-six-case average. These cases are a different, harder experiment than Version 1, so the two aggregate percentages are not a controlled comparison with each other. FNO and PINO again perform similarly.

The movie uses **seed 400**, selected as the first new evaluation seed before the final results. Both flights complete. The baseline parcel's maximum horizontal error reaches **47.7 cm**, versus **11.4 cm** for PINO. Crossing RMSE is **20.17 cm vs 5.99 cm**. Peak sling angles are **46.2° vs 43.4°**: the clearest improvement is the parcel staying closer to its target, not elimination of the wind-induced cable angle. Peak aircraft tilt is **21.2° vs 16.9°**.

[All 24 trial reports, including failures](comparison.json). All **32 tests passed**, including both flight presets, forecast causality and the Navier–Stokes scaling check. The original standard trajectory was rerun and matched exactly, and the original MP4's hash is unchanged. The native macOS viewer passed its smoke test. The scene and motion were inspected from multiple camera angles; all **1,250 frames** of the new MP4 decoded without error. Time spent on this follow-up is recorded separately in the [time log](../../TIME_LOG.md).
