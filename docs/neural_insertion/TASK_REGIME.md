# Thread physics in the task regime

October 6, 2026. **Every task-regime thread fixture passes the unchanged pointwise
gates at a 5 µs RK4 step with a 20 µs contact time constant; settling, drag and
press also pass at 10 µs.** That step is 64× larger than the 78 ns the 6 mm drop
required. Gates come from
[the task-derived criteria](ACCEPTANCE.md); the drop remains a labelled stress
test. Both material presets pass; neither is a measured thread.

## Clock: how large can the thread step be?

The settling fixture (rest start, support 50 µm below, tip lands near 0.03 m/s,
60 ms) ran with the ramped contact law at four time constants, timesteps from
τ/2 down to τ/16, both integrators and both materials. Unit systems were checked
at τ/8. [Manifest](THREAD_CLOCK_RESULTS.json), 80 cases.

| Material | Integrator | Largest passing step | At τ |
| --- | --- | ---: | ---: |
| Illustrative 100 MPa | RK4 or implicitfast | 20 µs | 40 µs |
| Polyimide 2.5 GPa | RK4 or implicitfast | 10 µs | 20 µs |

Penetration grows with τ, so τ = 40 µs fails the 0.2 µm settled gate for
polyimide (0.203 µm). Unit disagreement was at most 4.7e−10 µm everywhere. RK4
had 3–10× more margin than `implicitfast` at equal cost per simulated second.

## Probe fixtures: drag, press and release

Each adds a rigid 150 µm probe on slide joints, driven only through
force-limited actuators at a 50 kHz servo rate, with gravity compensation.
Penetration is checked over every contact at every step.

- **Drag:** a needle lying across the thread with a 20 µN preload slides
  200 µm along its own axis at 2 mm/s. The thread rolls about 100 µm, half
  the needle travel, as a rolling cylinder should.
- **Press:** a 100 µN force ramp, hold and unload; the support reaction must
  match the applied load.
- **Release:** the thread drapes over a hook, which slides out at 8 mm/s; the
  end falls 150 µm and lands near 0.054 m/s.

[Manifest](TASK_FIXTURE_RESULTS.json). All values in µm unless stated.

| Fixture | Timestep, 10 → 5 µs | Timestep, 5 → 2.5 µs | Units (worst) | Peak penetration |
| --- | ---: | ---: | ---: | ---: |
| Drag | 8.6e−7 | 5.1e−7 | 9.6e−7 | 0.73 |
| Press | 8.1e−7 | 2.4e−7 | 9.2e−11 | 0.92 |
| Release, illustrative | 0.89 | 0.21 | 7.3e−7 | 1.08 |
| Release, polyimide | **1.58, fail** | 0.19 | 1.2e−7 | 1.13 |

Press force balance error was at most 6.8e−5 relative (gate 1%). Release
failed at 10 µs and passed at 5 µs; we refined the step rather than relax the
gate, and keep the failed comparison in the record. Drag and press differences
near 1e−6 µm reflect slow, smooth contact, not a loosened measure.

One observation to carry forward: the press reaction lags the applied load by
about 20 ms and overshoots to 137 µN before settling. The probe must compress
the soft onset of both ramped contacts, about 1 µm each at 100 µN, at its
damping-limited speed. That compliance is a property of MuJoCo's soft contact
acting on a 2.4 µg segment, not a calibrated thread or tissue stiffness. It must
be measured before contact forces are trusted quantitatively.

![Clock, penetration and probe-fixture results](../../previews/neural_insertion/task_regime_v1/quality_review.png)

## Decision and cost

Thread phases use **RK4, 5 µs steps, τ = 20 µs, ramped impedance**. The robot
alone keeps its 1 ms `implicitfast` clock; a combined robot-and-thread model
runs at the thread step. Cost on this Mac is about 1.6–2 ms of wall time per
thread step, roughly 350 s per simulated second at 5 µs. That is affordable
for validation but too slow for learning with the thread in the loop. Thread
handling RL needs a cheaper thread (fewer segments or a coupled implicit rod)
or a GPU path; robot-only learning does not.

## Reproduce

```sh
uv run --locked python -m sixlegs.neural_insertion.thread_clock run \
  --output outputs/neural_insertion/thread_clock/recheck_v1 --jobs 10
uv run --locked python -m sixlegs.neural_insertion.task_fixtures run \
  --output outputs/neural_insertion/task_fixtures/recheck_v1 --tau 2e-5 --dt 1e-5 --jobs 10
uv run --locked python -m sixlegs.neural_insertion.task_fixtures run \
  --output outputs/neural_insertion/task_fixtures/dt5us_recheck_v1 --tau 2e-5 --dt 5e-6 --jobs 8
uv run --locked --extra wind python scripts/report_neural_task_regime.py \
  outputs/neural_insertion/thread_clock/recheck_v1 outputs/neural_insertion/task_fixtures/recheck_v1 \
  outputs/neural_insertion/task_fixtures/dt5us_recheck_v1 previews/neural_insertion/task_regime_recheck_v1
```

The 10 µs fixture run exits 1 because of the recorded release failure. Use
fresh output directories.
