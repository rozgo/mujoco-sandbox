# Per-tick wing forces and hover reference

User correction, September 13: derive flight forces directly from current wing
angles and speeds at every physics tick. Remove the extra wing-activity average;
MuJoCo integrates the wing joints and free body. Preserve previous demos.

`wing_motion_instant_v2` retains the same complete `wing_position` body,
78 actuators, torque limits, massless/collision-free wings, body drag, angular
damping, attitude assistance and force bounds. It removes the 12 ms temporal
average only. Instantaneous normalized sweep speed replaces filtered activity.
The law reads actual qpos/qvel and body state before every MuJoCo step. It receives
no command, desired altitude, actor action or clock phase. No body poses or
velocities are overwritten in live simulation. Stopping wing motion immediately
removes its flight-force contribution; existing body momentum still integrates.

Physics remains 5 kHz and actions 500 Hz in this comparison. The response change
and the proposed timestep experiment are separate. No optimizer or reward
change and no new RL run is part of this physical-model check.

The explicit `wing_force_response` numeric marker survives XML/MJB and batch
compilation. Physical fingerprints include the selected force-law parameters.
Old models without the marker keep the original filtered law and fingerprint.
`instant_migrate` explicitly transfers the old checkpoint to the new contract,
preserving all actor/critic/optimizer/exploration tensors. This is a physical
transfer for evaluation, not training or evidence of successful transfer.
PPO and motor review reconstruct the response recorded in the checkpoint.

## Reference validation requested before more learning

Run the existing measured-state hover reference for ten seconds from the
declared nominal seed97013, using the unfiltered law. It commands bounded wing
actuators; the resulting measured wing motion supplies the flight forces. The
reference uses altitude and vertical-speed feedback to regulate wing amplitude.
It is a controller used to check the plant, not a trained MaleCNS actor.
Standing and walking use the same physical body in the simultaneous check.

Keep the existing gates: minimum hover height 5 mm, upright >0.5, root tracking
RMSE <5 mm, forbidden support <0.1 body weights and no numerical warnings.
Additionally report height peak-to-peak and RMS vertical speed, including the
initial transient and a separate settled window. Preserve failures if the
unchanged reference does not transfer. Do not adjust gates to claim success.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus reference \
  --preset wing_position --wing-response instant \
  --teacher assets/embodied_fly/teachers/walking.npz \
  --ground-posture --stand-initial-form --hover-reference state \
  --walking-reference anchored --device cpu --seconds 10 --seed 97013 \
  --output outputs/embodied_fly/instant_reference_probe_01
```

Video must identify the reference controller, show continuous physical hover
at 1x, and retain a stable observer view. Decode, visually inspect and open it.
This establishes a controllable reference case only; autonomous learned hover,
takeoff/landing, disturbances and survival remain separate requirements.

## Reference correction after the first probe

The original state reference sustains ten seconds but drifts horizontally;
its 8.60 mm root RMSE fails the unchanged 5 mm gate. Keep that failed probe.
The explicit `--hover-reference state-position` reference adds fore-aft position
feedback through wing-stroke angle, with speed gain0.6 and position gain0.3 in
the inherited CGS convention. It changes reference actuator commands only;
the force model, torque limits and complete body remain identical. The first
corrected probe passes hover with 2.88 mm root RMSE and a settled height span
of0.726 mm. Small lateral drift remains; this is not arbitrary 3D station-keeping.

The final reproducible capture uses the same nominal seed and ten-second duration
with `--hover-reference state-position`, after committing source. Render its
`hover` case with `embodied_fly.record`. This is a reference/controller plant
demonstration, never an RL success video. Preserve the simultaneous stand/walk
reports, including the walking yaw failure, without claiming every command passes.
