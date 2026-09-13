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
