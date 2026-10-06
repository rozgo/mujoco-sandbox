# Task-derived acceptance criteria

October 6, 2026. Gates now follow the task phases in the [brief](BRIEF.md).
Earlier studies used a 6 mm, 0.34 m/s thread drop as their contact gate. That
was a numerical stress test, not a task requirement; see [the process
account](PROCESS.md#7-correction-a-stress-test-had-become-an-acceptance-gate).
Its record and sensitivity result stay in [the smoothing study](CONTACT_SMOOTHING.md).

## Thread contact, by task phase

| Task phase | Contact regime | Fixture | Pointwise gates |
| --- | --- | --- | --- |
| Pick from cassette, retainer closes | Quasi-static grasp, mm/s | Normal loading (`press`) | Force balance < 1%; timestep < 1 µm; units < 0.01 µm |
| Transport and align | Held thread, free end may brush surfaces | Settling (`der_settle`) | Timestep < 1 µm; units < 0.01 µm; settled penetration < 0.2 µm |
| Insert to depth | Sliding against needle and support, mm/s | Needle drag (`drag`) | Timestep < 1 µm; units < 0.01 µm |
| Release and withdraw | Hook slides away, short fall, relaxation | Release (`release`) | Timestep < 1 µm; units < 0.01 µm |
| Missed-pickup recovery | Falls 5–13 mm, 0.3–0.5 m/s | 6 mm drop (stress test) | Statistical only, see below |

Status, October 6: with RK4 and a 20 µs contact time constant, all four
fixtures pass at 5 µs for both material presets; settling, drag and press also
pass at 10 µs. Thread phases use 5 µs. See [the results](TASK_REGIME.md).

All fixtures also require finite state, no MuJoCo warnings, actual contact and
peak penetration below 2 µm. Settled penetration below 0.2 µm applies to an
unloaded resting thread; under applied load, penetration is reported against
load, because MuJoCo's soft contact compliance is not a calibrated material law.

The drop keeps its non-pointwise checks: finite state, penetration bound,
dissipated rather than gained energy, and rest position within the re-grasp
tolerance across timesteps and predeclared seeds. A real dropped thread does
not land repeatably to 1 µm either.

## Robot plant

Set before the first motion run and unchanged afterwards:

- **Hold:** 2 s at a fixed command, drift below 1 µm on every axis, no actuator
  at its force limit.
- **Approach:** needle tip to 1 mm above a target site; no contact between robot
  and phantom, support or cassette; final lateral and vertical tip error below
  10 µm within 0.3 s of the reference ending; tracking error and actuator force
  margins reported.

## Learning

Alignment and tracking RL is robot-only at the 1 ms robot clock. Success,
reward terms and evaluation seeds are fixed before training. Task thresholds for
the full insertion cycle are frozen only after the thread material and contact
parameters are measured, as the brief requires.
