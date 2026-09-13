# Motor learning: stand, walk and hover

Current scope: one MaleCNS actor, one physical fly, three motor commands. Utility
selection and needs-driven behavior are deferred. Takeoff, landing and transitions
between ground and air remain later requirements; the first pilot starts each
primitive from its declared ground or airborne state.

Every world uses `wing_motion`: 5 kHz native MuJoCo physics and 500 Hz control.
The checkpoint records a fingerprint of masses, inertias, joints, contacts,
actuation, solver settings and flight-force parameters. Native and batched
evaluation must match. Wing bodies have zero mass/spatial inertia, no collisions
and no aerodynamic terms. Independent diagonal angular armature permits wing
motion; the tested custom flight law is the only wing-to-body coupling.

## Declared first pilot

- Parent: preserved `motor_flight_angle_probe_01.pt` walking diagnostic.
- 397 inputs, 78 outputs, four internal graph updates, the same fixed measured
  connections and trainable cell dynamics. Two neutral altitude inputs extend
  the parent's 395-input encoder. No raw-sensor-to-motor bypass.
- Utility selection is disabled. Utility-head and intention-encoder weights are
  frozen. The existing context projection uses fixed column 1 in all tasks;
  it is no longer a selected behavior. Task requests enter through sensory inputs.
- **32 physical worlds**: 11 stand, 11 walk, 10 hover. Straight walking request
  1 cm/s; ground altitude request zero, hover 1.8–2.2 cm. Initial yaw ±0.15 rad.
- **180-second wall cap**, seed 71001, 16 CPU physics threads, RTX 4090 graph and
  optimization. Fresh Adam 1e-5, recurrent chunks of 32 actions (64 ms).
- Online imitation from the current physical state: execute **80% reference /
  20% student** commands and train on the reference's bounded motor targets.
  Equal mean loss per task; hover adds twice the wing-channel MSE. No utility loss.
- Reset only worlds leaving the height/upright training envelope or reaching
  two seconds. Save causal failure traces. Assistance and timeouts are not proof
  of autonomous success. Record setup, training, collection, optimization and
  actual transitions separately.
- Independently evaluate the saved checkpoint without teachers, mixtures,
  utility intervention or an external wing oscillator. Retain all three cases,
  including failures. Do not promote merely because imitation error decreases.

This is motor imitation, not PPO. The inherited walking reference and the
wing-motion hover reference generate training commands; only the student actor
controls acceptance evaluation. Physical reward learning can follow once these
primitives work. The legacy utility-PPO CLI explicitly rejects motor-only
checkpoints to prevent silently applying the wrong objective.

The two-second implementation reference review kept all three tasks upright with
valid support and no numerical warnings. Hover root RMSE was 0.733 mm. Ground
raw heading gates still failed (stand/walk yaw RMSE 1.77/3.35 rad/s); these gates
remain unchanged. Reference control is not a learned result.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
  --graph outputs/fly_survival/malecns \
  --resume assets/embodied_fly/diagnostics/motor_flight_angle_probe_01.pt \
  --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/motor_focus_01 \
  --device cuda --seconds 180 --worlds 32 --threads 16 --sequence 32 \
  --teacher-mix 0.8 --episode-seconds 2 --lr 0.00001 --seed 71001

uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus evaluate \
  --graph outputs/fly_survival/malecns \
  --resume outputs/embodied_fly/motor_focus_01/actor.pt \
  --output outputs/embodied_fly/motor_focus_01_evaluation \
  --device cuda --seconds 2 --seed 72001 --neural-view
```

Use unused output names when reproducing. Exit 2 means physical gates failed;
the complete evidence remains saved. New training videos are checked and opened
automatically for local review.

The fingerprint hashes compiled values exactly. Mac and Linux compilation differ
by rounding (maximum observed absolute difference2.60e-14, with all physical
arrays agreeing at1e-12). Each host's native/batched bodies match exactly; replay
uses the captured training MJB. Cross-host live recompilation must not bypass the
strict fingerprint check silently. This is not a different body preset.

## Measured outcome

The first [three-minute pilot](runs/motor_focus_01/SUMMARY.md) and its
[lower-assistance continuation](runs/motor_focus_02/SUMMARY.md) both failed
all three unassisted cases. Their combined training time was360.924648 s,
with324,608 real physical transitions in32 worlds. Assisted training survival
improved, but this has not become reliable stand/walk/hover control.

- [First complete review](../../previews/embodied_fly/motor_focus_01_all_tasks_v1.mp4).
- [Lower-assistance review](../../previews/embodied_fly/motor_focus_02_all_tasks_v1.mp4).

Both films include every case and its failed gate, actual simulated neural
activity and observer eye cameras. Earlier reviewed walking remains unchanged.

The [ground-wing correction](runs/motor_focus_03/SUMMARY.md) then adds the same
explicit wing-accuracy term to ground tasks. After181.184887 s of training,
unassisted standing and walking remain upright with valid foot support for two
seconds. A native five-second test confirms upright standing, with0.801 mm drift;
walking falls at3.28 s. Raw tracking gates and hover still fail. No runtime wing
override is used. These are motor-imitation results, not PPO results.

- [Updated complete three-task review](../../previews/embodied_fly/motor_focus_03_all_tasks_v1.mp4).
- [Full five-second ground cases](../../previews/embodied_fly/motor_focus_03_ground5s_v1.mp4).

## Initial form on the ground

`--ground-posture` teaches the initial pose during idle and restoring wing
commands during both idle and walking. The wing target depends on measured
angle and speed; it is no longer simply zero torque. Walking legs and hover
retain their task targets. All78 outputs remain learned at runtime.

Evaluation records every hinge's initial angle, separate leg/wing/body errors,
wing speed and body sag. These measured scores supplement the existing physical
gates. Training currently optimizes corrective imitation, not a physical PPO
reward; passive joints are measured but not given invented actuators.
