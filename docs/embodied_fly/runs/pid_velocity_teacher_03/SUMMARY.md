# Continuous velocity and heading PID reference

[Watch the complete exercise](../../../../previews/embodied_fly/pid_velocity_full_exercise_v1.mp4).
This is the PID teacher controlling the physical fly, not a trained MaleCNS actor.
One 71.2-second episode contains every exercise, with no resets or body-pose
assignments after initialization. All 50 declared stage checks pass.

The sequence includes hover, six translation directions without turning, left
and right turns in place, all twelve translation-direction/turn-direction pairs,
planar diagonals, and diagonal climbing/descending turns. Braking/hover separates
movements. Commands are forward/left/up velocity and yaw rate. No position or
heading targets are supplied, and return to the initial position is not required.

Left/right strafing changes heading by at most 0.123 degrees in the isolated
segments. The worst settled-window vector velocity error is 0.4611 mm/s and
worst yaw-rate error is 0.0417 rad/s. These are trailing 100 ms mean measurements
within each stage's final 0.4 seconds; raw 500 Hz velocities are retained and
shown separately in the video. Both cold-start transients and wingbeat motion
remain in the capture. Stage thresholds were specified before the first run;
this is a development reference, not a broad robustness benchmark.

## Fixed physical plant, tuned teacher

The old force law coupled roll and yaw through sweep imbalance. Versioned
wing_motion_heading_v3 adds yaw authority from the measured left/right wing
pitch difference, enabling sideways movement with an independently controlled
turn rate. It has no access to commands, targets or controller state. Wing/body
forces are recomputed every 1 ms physical tick. All previous models remain
available with their original fingerprints.

The body, masses, joints, contacts, actuator limits and clocks exactly match
the previous body. Three PID development captures use the same compiled model
hash. Teacher feedback tuning alone improves the declared stage count from
29/50 to 49/50 to 50/50. Earlier failures and exact measurements are preserved
in the two preceding run directories; see their reports for authoritative counts.

The final feedback uses translational Kp [20,20,50], Ki [20,20,900], roll
Kp 300/Kd 35, and yaw Kp 20/Ki 30. Commands reach 1.5 mm/s per translation axis
and 0.45 rad/s yaw. All use bounded wing-angle targets, the accepted 30 Hz
teacher stroke, and unchanged actuator torque caps. The neuron network is not
invoked in this capture.

## Review and next stage

The video includes a damped close view, fixed overview, body heading, and four
velocity charts. The raw-velocity strips and mean-velocity plots have separately
labeled scales; averaging is display/measurement only and never controls physics.
Playback is 1x and the episode remains continuous.

The encoded video passes full decoding: 3,560 frames at 50 fps, 1600x1000,
71.2 seconds. Ten sampled times and a full-resolution turn frame were visually
reviewed; labels, framing and raw/mean distinctions are readable. It was opened
for user review at 2026-09-14 04:32:33 UTC. User acceptance remains pending.
Rendering took 154.286 seconds. The final physical capture took 63.989 seconds;
all three captures together took 193.223 seconds, with zero learning updates.

The user requested this teacher review before training a fresh student. No old
policy, optimizer, or normalization will be resumed. The new initialization path
uses the same measured MaleCNS graph and fresh trainable parameters. Utility and
navigation are intended to learn through that same core later, rather than
through permanent external decision networks. No student training has started.

See [the interface and reproduction guide](../../VELOCITY_TEACHER.md),
report.json, MEASURED_STATS.json, and validation.json for exact evidence.
