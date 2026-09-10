# Measured wind-delivery results

**24/24 complete deliveries; PINO reduced mean crossing tracking RMSE by 59.4% relative to the frozen-field forecast.**

These are six paired held-out cases (seeds 300–305), not a statistical guarantee for other winds. The same controller and actuator limits are used throughout. RMSE measures horizontal package error during t = 6–32 s.

| Forecast | Mean tracking RMSE | Mean peak swing | Deliveries |
| --- | ---: | ---: | ---: |
| Frozen field | 10.89 cm | 28.81° | 6/6 |
| Data-only FNO | 4.28 cm | 25.36° | 6/6 |
| PINO | 4.42 cm | 25.30° | 6/6 |
| True future (diagnostic) | 4.30 cm | 25.31° | 6/6 |

FNO and PINO give similar flight performance here. The strong finding is the value of forecasting over holding the wind fixed; this experiment does not establish a large advantage for the physics loss. True-future wind is only a diagnostic: model mismatch and approximate control can make an imperfect forecast occasionally score better.

| Wind seed | Frozen field | FNO | PINO | True future |
| --- | ---: | ---: | ---: | ---: |
| 300 | 9.90 cm | 3.88 cm | 3.88 cm | 3.65 cm |
| 301 | 13.33 cm | 6.02 cm | 6.39 cm | 6.31 cm |
| 302 | 8.26 cm | 3.03 cm | 2.96 cm | 2.99 cm |
| 303 | 9.36 cm | 2.70 cm | 2.76 cm | 2.43 cm |
| 304 | 13.43 cm | 3.71 cm | 3.73 cm | 3.72 cm |
| 305 | 11.07 cm | 6.34 cm | 6.79 cm | 6.72 cm |

## Independent flow validation

| Grid | FNO 2 s velocity RMSE | PINO 2 s velocity RMSE |
| --- | ---: | ---: |
| 64² | 0.0643 m/s | 0.0650 m/s |
| 128² | 0.0643 m/s | 0.0650 m/s |
| 256² | 0.0413 m/s | 0.0434 m/s |

64²/128² use four held-out trajectories and three initial times; 256² uses the same four trajectories at 12 s. Training uses 64² labels and PINO also uses a 128² physics residual. The 256² grid is unseen by either loss.

Direct 128² PINO vorticity relative L2 error is 0.057133; Fourier interpolation of the 64² rollout gives 0.057133. These nearly equal values show successful grid transfer, not added fine-scale accuracy.

CPU/CUDA and CPU/Metal agreement is checked over eight autoregressive steps at all three resolutions; maximum absolute vorticity disagreement is below 0.00002 s⁻¹. See [CUDA checks](backend_cuda.json) and [Metal checks](backend_mps.json).

Timing samples in the validation JSON are diagnostic wall measurements against this NumPy reference, including model transfers. They are not a benchmark against optimized CFD or a guaranteed hardware speedup. Final validation ran alongside other preparation work, so no precise speedup claim is made.

## Physical checks

All runs stayed within the 10 N per-rotor cap (observed maximum 9.734 N). Maximum cable constraint extension was 0.400 mm. No gate collisions or MuJoCo numerical warnings occurred. Every release followed physical destination contact.

The movie uses seed 300, the first evaluation seed chosen before the final results. [Full per-flight records](comparison.json) · [Flow validation](operator_validation.json) · [CUDA flow validation](operator_validation_cuda.json) · [Model provenance](manifest.json).
