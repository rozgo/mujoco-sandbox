# Firmer wing stops preserve expert flight but do not rescue the student

Source `48008f1`, native MuJoCo CPU physics at 20 kHz, 5 kHz control.
The six wing limit time constants change from 1 ms to 0.2 ms; all other
physical parameters and the evaluated `motor_wing_readout_01` checkpoint stay
unchanged. The two initial airborne captures and durations were declared before
evaluation. This is a physics diagnostic with **no training**.

| Controller / stops | Task | Duration | Root RMSE | Lowest root height | Worst limit overshoot | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Inherited expert / firm | Hover | 1 s | 0.174 mm | 9.952 mm | 0 rad | Airborne |
| Inherited expert / firm | Forward, 10 cm/s | 0.3 s | 0.102 mm | 9.930 mm | 0.0345 rad | Airborne |
| Student / original | Hover | 0.3 s | 14.422 mm | 0.710 mm | 1.2666 rad | Failed |
| Student / original | Forward, 10 cm/s | 0.3 s | 15.021 mm | 0.616 mm | 1.2293 rad | Failed |
| Student / firm | Hover | 0.3 s | 20.869 mm | 0.515 mm | 0.2140 rad | Failed |
| Student / firm | Forward, 10 cm/s | 0.3 s | 17.200 mm | 0.441 mm | 0.2070 rad | Failed |

Worst student overshoot falls about **83%** in both tasks, measured at every
physics substep. All six runs remain finite with zero MuJoCo warnings. The
expert has no prohibited ground loading; all student flights hit the ground.
RMSE covers the full requested duration, including failure, and does not imply
useful flight before contact. Sparse CUDA execution can vary between runs; these
are individual diagnostic trajectories, not a multi-seed robustness comparison.

Setup times in table order: **4.226557, 4.128939, 3.033902, 3.011464, 3.030136,
2.938492 seconds**. Stepping/capture/state-write times: **8.361059, 2.561729,
5.552881, 5.580582, 5.619500, 5.576117 seconds**. No new video was rendered.
Teacher control is the inherited policy plus wingbeat generator. Student control
uses the complete graph actor without runtime teacher, wing oscillator or
applied root forces. Both use the same approximate aerodynamic-force model.

Every local/GPU state and model hash was verified; the original flight model is
unchanged (`3c38ca5…`). The firm variant is `a94552b…`. Captures retain bounded
actions, finite states, 0.2 ms timestamps and causal previous-action observations.
Detailed records and verification are alongside this summary.

Do not adopt the variant as a flight fix or promote a student. The original
physical preset remains the default. The next controller investigation should
target autonomous feedback stability and the gap between expert-history fit
and self-generated motion, preserving the existing ground references.
