# One continuous velocity and heading exercise

The requested reset is a fresh student with no inherited policy, optimizer,
normalizer or neural memory. Keep the measured MaleCNS graph. Motor commands are
forward/left/up velocity and yaw rate; zero requests active braking and hovering.
No position or heading target is given to the teacher or student. PID state and
its wing clock remain teacher-only. Verify this teacher before student training.

Every episode contains all exercises, continuously, without resets between them:
six isolated translation directions, two turns in place, all twelve pairings of
translation direction with left/right turning, two planar diagonals, and two
diagonal climbing/descending turns. Brake/hover separates movements; begin and
end with hover. This covers isolated channels and mixed controls; it is not a
claim to enumerate every possible continuous command vector. Later episode
orders may vary while retaining the complete exercise in every world.

## Heading authority and physical version

The previous force law ties roll and yaw to left/right sweep-effort difference.
It cannot independently command lateral movement and heading. The explicit
`heading_control=True` plant adds yaw torque from the sine of the **measured**
left/right wing pitch difference. Its strength is 120 rad/s² before the existing
wing-activity engagement and inertia mapping. Symmetric pitch deviations around
-1 rad retain equal lift efficiency. The teacher uses at most ±0.25 rad pitch
offset to command/cancel yaw, within the unchanged mechanical ranges.

This is `wing_motion_heading_v3`, encoded in the compiled model and its physical
fingerprint. Prior v1/v2 models keep their original force law and fingerprint.
Body masses, contacts, joint limits, actuator torque limits and the 1,000 Hz
physics /500 Hz control clocks are unchanged and checked against the earlier
plant. The model reads actual wing angles/speeds every physical tick; it cannot
read desired velocities, heading, target positions or PID state. There are no
live root-pose writes, translational support servos or prescribed body paths.

The velocity PID uses PI feedback: translational Kp [20,20,50] s⁻¹ and
Ki [20,20,900] s⁻², with anti-windup and ±300 cm/s² acceleration limits.
Yaw Kp 20 s⁻¹, Ki 30 s⁻², acceleration limited to ±30 rad/s².
The roll loop uses Kp 300 s⁻² and Kd 35 s⁻¹ to reject turn/braking transients.
Its acceleration-to-wing mapping retains the accepted 30 Hz stroke reference,
joint feedback, force limits and physics-tick force integration. The yaw mapping
cancels the roll channel's yaw contribution while supplying requested turning.

Commands reach 1.5 mm/s on each requested translation axis and 0.45 rad/s yaw.
Ramps last 0.25 s. The complete reference is 71.2 s with continuous physical and
controller state. Per-stage gates are recorded before capture: final 0.4 s,
100 ms mean vector velocity error below 0.5 mm/s and yaw-rate error below
0.12 rad/s; upright >0.85, height >5 mm, forbidden contact <0.1 bodyweights.
Raw 500 Hz velocities are retained and displayed separately from the observer
mean; that mean never feeds the controller. Returning to the initial position
is not required by a velocity command.

The fresh student schema has 391 inputs, no needs or absolute/target-position
inputs, and 78 joint/adhesion outputs. It retains the fixed graph and decoder
widths. The encoder is initialized directly for the new schema, with no legacy
sensor-extension branch or fitted feature statistics. Every Linear layer starts
with new seeded random parameters; layer normalization starts with unit scales
and zero biases, and trainable cell dynamics start from neutral defaults. The
initializer has no parent-checkpoint input or weight-loading path.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_exercise \
  --contract-report docs/embodied_fly/runs/pid_movement_imitation_01/final_evaluation.json \
  --output outputs/embodied_fly/pid_velocity_teacher_01
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_video \
  --source outputs/embodied_fly/pid_velocity_teacher_01 \
  --output previews/embodied_fly/pid_velocity_full_exercise_v1.mp4
```

Run physical capture on the Linux host matching the previous fingerprint. The
Mac can render its transferred MJB/capture without recompiling the model.

Later utility and navigation learning should remain connected through the same
MaleCNS core. Velocity commands are motor-teaching inputs, not a decision to
install permanent external utility/navigation networks above the connectome.
No student training starts during this teacher-reference review.

## Faster physical flight follow-up

The user likes the full teacher video but requests approximately ten times faster
physical motion, with playback still 1x. `--speed-scale 10` commands 15 mm/s per
translation axis and 4.5 rad/s (258 degrees/s) yaw on the same 71.2-second
continuous schedule, including its 0.25-second ramps. No simulation clock or
replay speed is multiplied. Preserve the previous video and captures.

`--fast-flight` explicitly selects `wing_motion_fast_heading_v4`. Only body-axis
yaw damping changes: its time constant is 0.25 s instead of 0.025 s. Roll/pitch
damping, wing-to-force gains, body/joint mechanics, actuator bounds and clocks
remain unchanged. The old yaw damping would require 180 rad/s² just to sustain
4.5 rad/s, beyond the pitch channel's 120 rad/s² peak; reducing yaw resistance
provides the needed turning range. The PID's inverse wing mapping uses the new
declared damping. Forces still depend only on measured body/wing state.

Predeclared faster-motion review uses the same final 0.4-second stage window and
100 ms mean. Vector velocity error must be below max(0.5 mm/s, 10% of requested
vector speed); yaw-rate error below max(0.12 rad/s, 10% of requested yaw rate).
Braking and hover retain the original absolute tolerances. Physical upright,
altitude and contact gates are unchanged. Retain all failed development runs.
The faster body's controller must be validated before fresh student training.

The v4 development runs remain stable but miss tight sideways turns. The
additional `--lateral-control` flag selects `wing_motion_agile_v5`: measured
right-minus-left wing stroke orientation produces lateral thrust, bounded by
25% of instantaneous lift through a tanh response. Mean stroke orientation
still controls forward thrust; pitch difference controls yaw; sweep motion
provides lift. All forces remain zero without wing activity. This is an
explicit extension of the force law; old v3/v4 models remain reproducible.

The faster teacher adds causal command-acceleration and drag feedforward,
through the existing bounded wing targets. It uses current/previous commands
and measured heading, without future commands or root-pose changes. Feedback
gains are Kp [40,40,50], Ki [200,200,900], roll Kp 600/Kd 40 and yaw Kp 60/Ki 30.
Its desired yaw acceleration cap is 60 rad/s²; actual actuator torque limits
are unchanged. The slow teacher retains its original feedback settings.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_exercise \
  --contract-report docs/embodied_fly/runs/pid_movement_imitation_01/final_evaluation.json \
  --speed-scale 10 --fast-flight --lateral-control \
  --output outputs/embodied_fly/pid_velocity_fast_review
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_video \
  --source outputs/embodied_fly/pid_velocity_fast_review \
  --output previews/embodied_fly/pid_velocity_fast_review.mp4
```
