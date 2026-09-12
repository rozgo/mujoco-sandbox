# Flight curriculum: required physical setup

Flight remains part of the approved goal, inside the same learned actor. It is
not trained yet. This check records what the pinned FlyBody code actually needs
before collecting wing-control demonstrations or applying flight rewards.

The current full walking model retains six wing joints and all wing actuators.
Its compiled air density is 0.00128 g/cm³ (1.28 kg/m³), viscosity 0.000185
 g/(cm·s) (1.85e-5 Pa·s), and gravity −981 cm/s². However, its two dedicated
wing-fluid geoms currently have zero fluid-model coefficients. Their presence
and visible wing meshes do not establish an operational flight model.

The pinned upstream `Flying` task enables ellipsoid wing aerodynamics with
coefficients `[1.0, 0.5, 1.5, 1.7, 1.0]`, changes all three wing-axis gains to 18,
and uses wing damping 0.007769230 and stiffness 0.01 in its CGS convention.
The walking composition currently uses gains 3/2/1 and damping 0.0005.
Upstream flight uses 20 kHz physics and 5 kHz control, compared with our current
5 kHz / 500 Hz terrestrial curriculum. Its reference wingbeat is 218 Hz.
These are source settings, not locally validated flight results.

Before training flight:

1. Add an explicit flight-ready physical preset, preserve current walking
   reproduction, and measure lift/drag, wing torque saturation and timestep
   convergence with the complete body. Keep meaningful floor contacts and legs;
   upstream flight task defaults disable both, which is unsuitable for takeoff
   and landing in our arena.
2. Validate ground-to-air and air-to-ground contacts. Lift must come from wing
   motion interacting with the fluid model, without root forces or pose writes.
3. Resolve the control-rate change explicitly. Preserve one controller/checkpoint
   and demonstrate that its recurrent timing and terrestrial skills survive the
   higher wing-control rate or a documented internal multirate implementation.
   Merely calling existing walking weights ten times faster is not validation.
4. If using the official flight teacher for training data, capture its complete
   executed wing controls. Its `FlightImitationWBPG` combines a learned residual
   with an external wingbeat generator. That may supply demonstrations, but the
   student must generate its own individual wing controls with no runtime pattern
   generator, and attribution must identify the inherited motion knowledge.
5. Extend the same actor through wing control, stable free flight, takeoff,
   maneuvering and landing while retaining walking/stopping. Then train the
   survival decisions that select and combine those learned skills.

Sources inspected locally at FlyBody revision
`d015e9bfe441bd90ae431bac24c55cb74bdbce26`:

- [Full anatomy and defaults](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/fruitfly/assets/fruitfly.xml).
- [Flight physical setup](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/base.py).
- [Flight constants](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/constants.py).
- [Teacher plus wingbeat generator](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/flight_imitation.py).

## Implemented physical preset and first force diagnostic

`FlyEnvironment("flight")` now enables the pinned wing aerodynamic coefficients,
18-unit wing gains, damping 0.007769230 and stiffness 0.01. It uses 20 kHz physics
and 5 kHz control, removes the six wing actuator filters, and preserves all legs,
mouth, antennae, free root and ground collisions. The walking preset is unchanged.
Terrestrial actuator filters remain unchanged in the flight preset too.

The flight body has 72 filter states instead of 78. The observation still has
383 entries: its activation section exposes one effective input per actuator,
using filter state where present and direct control on unfiltered wings. This
exactly preserves existing walking observations. Equal input/output dimensions
do not establish behavioral compatibility at a tenfold action rate; controller
timing and curriculum transfer still require an explicit design and validation.

A first 30 ms airborne diagnostic uses the upstream *approximate* wingbeat pattern
with its proportional angle-error wing torque, no learned residual and no brain.
The body is initialized airborne once, then moves only through bounded actuators,
aerodynamics, gravity and native MuJoCo stepping. No root force is supplied.

| Condition | Mean upward passive force / weight | Vertical travel | Solver warnings |
| --- | ---: | ---: | ---: |
| Air, 20 kHz physics | 0.2760 | −3.743 mm | 0 |
| Vacuum, 20 kHz physics | 0 | −4.988 mm | 0 |
| Air, 40 kHz physics | 0.2816 | −3.722 mm | 0 |

Halving the timestep changed final height by 0.0203 mm over this short interval.
Maximum wing actuator force was below 82% of its limit; no channel saturated.
This demonstrates an aerodynamic response, **not sufficient lift or stable
flight**. The approximate pattern fails to support body weight. Longer integration,
stronger/learned wing control, realistic demonstrations, takeoff, flight stability
and landing remain required. The recorded controller is explicitly ineligible for
student acceptance, and will never be presented as learned flight.

Two tests verify unchanged anatomy/contact flags and walking observations,
valid 20 kHz ground contact, no applied root forces, and dissipative aerodynamic
force at identical physical state. The latter toggles the fluid model while
holding state fixed, separating force response from trajectory divergence.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.flight_probe \
  --output outputs/embodied_fly/flight_physics_probe_02
```

The first run is preserved separately. New captures must use unused output names.

## Inherited flight teacher for demonstrations

The supplied `wing_pattern_fmech.npy` and learned flight SavedModel are now
available with [attribution](../../assets/embodied_fly/teachers/FLIGHT_NOTICE.json).
The converted flight teacher's outputs match 32 TensorFlow reference probes to
maximum absolute error 8.6427e-7. Its smaller 256-wide layout is handled explicitly;
the existing walking importer remains numerically checked.

The supplied pattern alone approximately doubles upward force compared with the
synthetic probe, but still cannot support weight. `FlightTeacherOracle` adds the
inherited policy's wing/body corrections, observes the current complete body, and
supplies retracted-leg servo targets without removing joints or applying root
forces. A first 0.2-second hover remains near the target altitude. This is inherited
teacher performance, **not the learned MaleCNS actor flying**. Neither this adapter
nor its wingbeat generator belongs in the deployed student.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.flight_teacher \
  --teacher assets/embodied_fly/teachers/flight.npz \
  --wing-pattern assets/embodied_fly/teachers/wing_pattern_fmech.npy \
  --output outputs/embodied_fly/flight_teacher_hover_reproduction --seconds 1
```

The recorder now uses actual capture timestamps and validates their spacing
against the reported action clock. Legacy walking captures retain their 500 Hz
fallback. Flight recordings are shown at 1× with 20 kHz physics / 5 kHz control
labels; the inherited teacher is explicitly identified. The one-second hover
clip fully decodes to 50 frames at 1600 × 900, 50 fps.

Clean-source extended checks now confirm one-second inherited hover (0.174 mm
root-position RMSE) and half-second forward flight at 20 cm/s (0.319 mm RMSE),
with full anatomy, zero ground loading and zero solver warnings. Exact reports:
[hover](runs/flight_teacher_hover_01/report.json),
[forward flight](runs/flight_teacher_forward_01/report.json).
The shared student still needs direct wing-control learning, preserved recurrent
timing, terrestrial rehearsal, takeoff and landing. No inherited expert result
is counted as student success.

## First shared-actor wing-learning pilot

The unchanged online01 walking actor remained upright in all six original-clock
flight-physics cases, but tracking deteriorated. Increasing the actor clock to
5 kHz, even with mathematically scaled cell relaxation, introduced prohibited
support and toppled the normal walking case. The faster clock is not a validated
drop-in replacement. Reports retain both probes and their original failed gates.

Eight complete-body flight demonstrations supplied 12,000 individual-actuator
targets across 2.4 simulated seconds, speeds 0/5/10/20 cm/s and seeded wing phases.
All eight stayed inside the declared demonstration envelope with no warnings or
ground support. Setup took 4.194998 seconds and collection 20.218079 seconds.

The first mixed-clock pilot continued online01 for **60.168454 seconds** on the
RTX 4090, with 32 parallel *offline neural sequences*, 207 updates and 105,984
supervised examples. It used one shared actor, 105 walking batches at 500 Hz and
102 flight batches at 5 kHz. The recurrent cell relaxation was scaled to each
dataset's clock; all internal cell parameters and interfaces remained trainable.
Setup took 3.645980 seconds; validation 0.597049 seconds; peak CUDA allocation
was 5,176,925,696 bytes. The fixed graph remained unchanged.

Flight validation MSE fell 0.463187 → 0.019442; walking MSE stayed near 0.017.
**Physical evaluation rejected the checkpoint.** Both 0.3-second teacher-free
flight probes fell. Three of six fixed walking cases lost stability, and the
continuous stop/resume toppled. Low error on expert states did not preserve
closed-loop behavior. This diagnostic checkpoint is archived separately; it does
not replace online01 or establish learned flight.

Reproduce collection and the declared pilot with unused output directories:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.flight_collect \
  --teacher assets/embodied_fly/teachers/flight.npz \
  --wing-pattern assets/embodied_fly/teachers/wing_pattern_fmech.npy \
  --output outputs/embodied_fly/flight_demonstrations_reproduction
uv run --project experiments/embodied_fly --locked python -m embodied_fly.train \
  --graph outputs/fly_survival/malecns \
  --data outputs/embodied_fly/demonstrations_02 \
  --additional-data outputs/embodied_fly/flight_demonstrations_reproduction \
  --resume assets/embodied_fly/diagnostics/motor_online_01.pt \
  --output outputs/embodied_fly/motor_flight_probe_reproduction \
  --seconds 60 --worlds 32 --sequence 16 --burnin 16 \
  --reset-fraction 0.25 --lr 0.0001 --seed 43001 --wing-loss-weight 1
```

The training command uses CUDA; the original walking dataset must first exist
locally as documented in the main guide. Neither dataset is a live physics batch.
The student evaluator uses a declared captured airborne state only for reset,
then drives all 78 actuators directly. No teacher or oscillator runs in its loop.
Takeoff, landing, altitude/perception inputs, control-clock transitions and
survival utility remain future curriculum work.
