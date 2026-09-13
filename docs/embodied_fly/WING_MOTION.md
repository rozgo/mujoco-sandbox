# Flight dynamics driven by wing motion

User-approved course correction, 2026-09-13 03:45:16 UTC. Preserve the same
connectome-based actor, all six independent wing command channels and actual
wing angle/velocity feedback. Replace aerodynamic and inertial wing-to-body
coupling with a documented force model driven by the resulting wing motion.
The fly must still learn wing coordination. This is not a direct lift/velocity
action interface or a body-motion animation. Existing aerodynamic trials and
the selected walking film remain separate and reproducible.

## Mechanism

`sensors → encoder → fixed MaleCNS connections → decoder → six wing joint commands`

`measured wing motion → causal flight force law → free MuJoCo body → sensors`

The `wing_motion` preset retains 108 generalized velocities, 78 actuator commands
and the 395-input actor schema. The wing bodies have zero mass and spatial inertia,
with no wing collision or aerodynamic contribution. Positive **diagonal joint
armature** supplies an independent angular response for the six bounded actuators;
it is not physical wing mass and creates no wing/non-wing mass-matrix coupling.
MuJoCo integrates both the controlled wing coordinates and the free body. No live
body or wing pose is prescribed. Body/leg contacts remain enabled.

The compiler's positive moving-body-inertia requirement is handled by applying
`configure_model` after compilation and calling `mj_setConst`. Saved MJBs contain
the final parameters. Recompiling source XML must apply this configuration again,
including the batch model with added sensors. Tests check positive definiteness,
zero coupling and saved-model reload. Walking and aerodynamic presets are intact.

The force law receives only actual wing angles/velocities, body orientation and
body velocities. Rectified sweep speed is filtered independently for each wing
over 12 ms. Wing pitch modulates stroke effectiveness. Combined activity generates
lift at whole-fly center of mass; differences generate roll/yaw torque; mean stroke orientation modulates
forward thrust. Filter state is per world and resets with that world's episode.
It must be captured alongside physical state for exact live resumption.

The body response deliberately includes activity-dependent drag, angular damping,
and an upright restoring torque. Lift is biased toward world up. These are
engineered dynamics, not measured insect parameters. There is no target-position
or altitude servo in the force law and no force input from commands or a teacher.
When wing motion stops, activity and lift decay. Static wing poses cannot sustain
lift. No runtime pattern generator is added to the student.

## Initial declared parameters

FlyBody CGS units (cm, g, s) are retained. See `WingMotionConfig` for exact values.

| Quantity | Initial value |
| --- | --- |
| Physics / policy | 5,000 / 500 Hz; ten steps per action |
| Per-axis diagonal armature | 2e-6 g cm² |
| Per-axis torque limit | ±0.03 g cm²/s² |
| Wing joint damping / stiffness | 0.0002 g cm²/s / 0.001 g cm²/s² |
| Activity reference speed | 50 rad/s sweep |
| Maximum lift | 2.24 body weights |
| Horizontal / vertical drag time | 120 / 80 ms, scaled by activity |
| Angular damping time | 25 ms, scaled by activity |
| Restoring / steering acceleration scale | 100 / 60 rad/s² |
| Maximum modeled angular acceleration scale | 250 rad/s² |

The angular torque uses body mass × (0.06 cm)² as its declared inertia scale;
actual angular acceleration still follows MuJoCo's full articulated mass matrix.
Drag contributes separately from the lift cap. Existing walking mass is not
silently reassigned: this flight preset removes the two 8 μg wing body masses.

## Validation and first learning pilot

Before training: verify isolated wing movement, absence of aerodynamic forces,
motion-dependent lift/steering, shutdown decay, damping dissipation, native/batch
agreement and independent resets. Establish a physical hover reference using a
clearly labeled training-only wing controller. Evaluate the preserved actor on
the new body; changed physics is not a newly learned skill.

Then run a bounded initial warm start on new physical wing demonstrations with
walking retention, followed by independent student evaluation. Keep graph wiring,
actor count and sensory/motor routing intact. Record actual optimization time,
unique physical experience, failures, force-model configuration and hashes.
Learning success is sustained student control, not fitting loss or reference
controller performance. No speedup claim precedes measurement.

User language: describe **flight dynamics driven by wing motion** and state that
the force law is custom and non-aerodynamic. Do not claim recovered biological
flight mechanics or hide provided stabilization.

The initial reference test exposed a pitching moment from applying the force
at thorax COM. The final law applies its resultant at whole-fly COM, adding
the corresponding lever moment to MuJoCo's thorax force array. The first
failed reference is preserved. With this correction, the second reference
passes a two-second hover; this is a mechanism check, not student learning.

All five new force-model tests and the existing suite pass (61 total). Single
and sensor-augmented batch agreement is checked at a declared small numerical
tolerance; it is not bitwise equality. The independent mass-matrix check gives
exactly zero wing-to-body inertial coupling.

## Measured outcome and review

The [working reference](../../previews/embodied_fly/wing_motion_reference_v1.mp4) is a two-second training demonstration. The [student diagnostic](../../previews/embodied_fly/wing_motion_student_diagnostic_v1.mp4) retains the full rise and fall, actual simulated neural activity and observer cameras. Both are 1600 × 900, 50 fps, 1×; both were fully decoded and visually inspected.

Three RTX 4090 learning pilots took **423.149011 seconds** combined: 60.000525 seconds of decoder fitting, 180.274561 seconds of complete-actor imitation, and 182.873925 seconds of physical-reward PPO. Setup, reference collection, replay, evaluation and rendering are separate. The best imitation candidate briefly generates lift but fails sustained flight. PPO and four sampled-policy evaluations do not resolve it. All candidates remain diagnostic; takeoff, landing and unified walking/flight have not passed.

```sh
open previews/embodied_fly/wing_motion_reference_v1.mp4
open previews/embodied_fly/wing_motion_student_diagnostic_v1.mp4
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motion_flight \
  --controller reference --output outputs/embodied_fly/reference_reproduction --seconds 2
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motion_flight \
  --controller student --checkpoint assets/embodied_fly/diagnostics/wing_motion_brain_01.pt \
  --graph outputs/fly_survival/malecns --device cpu --seconds 2 --neural-view \
  --output outputs/embodied_fly/student_reproduction
```

CPU full-graph inference can be slow; use CUDA on the GPU host for evaluation. A failed physical gate exits with status 2 and still saves its report/capture. Output directories must be unused. Rendering a saved capture uses `python -m embodied_fly.record CAPTURE_DIR flight NEW_VIDEO.mp4`.


## Height-feedback curriculum

The opt-in `--height-inputs` actor extends the same encoder from 395 to 397 inputs. Two extra columns encode current root altitude above the z=0 floor and the explicitly requested altitude, each divided by 2 cm. These are ideal simulator measurements/commands, not biological sensors or learned vision. Ground examples request zero altitude; flight examples carry their declared reference target. Old corpus targets are read from the corresponding reference report and its hash is saved; no future trajectory is queried. A flat floor is assumed for this altitude measurement.

New columns start at zero and leave all old normalization and weights intact. Full-graph numerical migration and native/batch/recorded-observation tests precede learning. The force model receives no height command, so the brain must coordinate its wings to respond. The first longer-context pilot is declared in the learning journal; it is not yet a flight result.
