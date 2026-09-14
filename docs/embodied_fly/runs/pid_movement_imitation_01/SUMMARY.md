# PID movement imitation: independent flight result

The final checkpoint has lower aggregate position error, but **fails the user’s target-following review and is not accepted as a flight controller**. Both new checkpoints pass the predeclared numeric improvement gate; neither completes a single precise tracking case. The preserved parent remains available. See REVIEW_DECISION.json.

[Intermediate student video](../../../../previews/embodied_fly/pid_movement_imitation_midpoint_v1.mp4)
and [final student video](../../../../previews/embodied_fly/pid_movement_imitation_final_v1.mp4)
show every declared case at 1x, with the accepted PID hover separately labeled.

## What was taught

The same preferred imitation parent learns the accepted PID's commands during
stationary hover and six opposite-direction round trips: accelerate, brake,
reverse, return and settle. Half the 64 worlds remain stationary; half move.
Distances 1.2–1.8 mm and timing scales 0.85–1.15 vary between 12-second episodes.
The previous small vertical reset perturbations are retained. All 78 motor
outputs remain learned, including the six wing joints.

Teacher actions execute throughout training; the student predicts actions and
receives supervised wing MSE plus 0.1-weighted other-joint posture MSE. No PPO,
critic updates, physical reward edits, automatic handoff or extra deployed
controller. The exact 399-input brain and graph routing, body and 1,000/500 Hz
physical/action clocks remain unchanged. Only current targets enter the actor,
not routes, future positions or the teacher clock/integral. See [the command schema](../../COMMAND_SCHEMA.md).

The intermediate checkpoint is saved after 304.510329 s,
67 updates and 274,432 transitions.
It does not inherit later updates. The final checkpoint has the full training
ancestry. Neither independent evaluation receives teacher actions or resets.

## Frozen 12-second development evaluation

| Actor | Six-route position RMS, mm | Stationary-hover RMS after 1 s, mm | Airborne | Complete tracking |
|---|---:|---:|---:|---:|
| Preferred parent | 74.103 | 77.357 | 7/7 | 0/7 |
| Midpoint | 23.388 | 24.293 | 7/7 | 0/7 |
| Final | 20.808 | 21.586 | 7/7 | 0/7 |

The same compiled physical model, starts, targets and measurement windows match
the preserved parent capture. Routes require reaching both targets, returning
and settling; staying airborne alone is not a pass. All cases and failures are
retained for the full 12 seconds. Selection details are in selection.json.
These are fixed development cases, not a generalization benchmark or a causal
proof that movement demonstrations improve every training setup.

## Measured training

- Actual training **601.387001 s**, plus 7.431540 s setup.
- 64 CPU MuJoCo/mjbatch worlds, 16 physics threads, RTX 4090 neural learning.
  Physics is not Warp in this fly experiment.
- 544,768 world/action transitions and supervised examples;
  1089.536 aggregate simulated seconds.
  905.853 transitions/s including learning and logging.
- 133 Adam updates, LR 1e-4, 64-step recurrent sequences. Fresh optimizer.
- Collection including student neural forward passes: 302.583184 s;
  gradient optimization: 174.061198 s;
  trace writing and loop overhead: 124.742620 s.
- 64 completed teacher-driven episodes,
  0 physical failures,
  64 complete teacher tracking sequences.
  These measure the PID's executed behavior, not student success.
- Teacher action share 100%; independent student-only training transitions zero.
  Student-only physical experience is collected separately during evaluation.
- Actor 2,516,402 parameters. Peak PyTorch CUDA allocation
  4,196,587,520 bytes (3.91 GiB), not total device memory.
- First/last training-batch wing MSE: 0.00061879 / 0.00001154.
  These are different collected states and are not held-out accuracy or a flight gate.

The 196-test regression suite passes in 158.46 s; focused tests 10 pass in 8.82 s.
Artifact verification checks every training trace, PID action execution, fixed
brain/body interfaces, and both checkpoints' exact Adam update counts.

The next stage depends on independent behavior. Passing teacher-command fitting
alone does not authorize an automatic handoff or establish learned correction.


## User review and command-path investigation

The user correctly points out that the fly still drifts to a different location.
A 72% reduction in error must not be described as following the requested path.
The existing opposite-direction captures expose the distinction: a 3 mm horizontal
separation request produces at most 0.0361 mm separation in the final student.
The vertical response is reversed: +3 mm requested pair separation produces
-1.0385 mm actual separation at the first hold; -3 mm requested produces
+1.2479 mm at the opposite hold. The reference PID produces the correct signs.
This is descriptive analysis of existing matched trajectories, not a new success
criterion or a new physical rollout. See direction_response.json.

The read-only command audit checks target encoding, centimeter scaling, a
90-degree world-to-body rotation fixture, and PID desired-acceleration signs and
gains at matched states. All pass. No live body fields change during PID queries;
zero physical steps or training updates occur. This rules out those specific
errors in the tested cases, not every possible implementation bug.

Across all 544,768 saved teaching observations, 95% have horizontal error below
0.9464 mm and altitude error below 0.1508 mm. Median errors are 0.1369 mm and
0.0392 mm; current/target altitude correlation is 0.9841. Moving the target while
the expert tracks it accurately still supplies mostly near-target examples.
The student later visits states far outside that error distribution. This is a
concrete weakness in the teaching design. It does not prove the only cause;
teacher phase/integral compatibility and other learning-path issues remain open.

Do not resume another long run on this basis. First demonstrate correct
command-to-wing correction on matched positive/negative position and velocity
errors, with sufficient history, then validate independent physical recovery.
Keep the current graph/body and preserve all failed results. The two audit JSON
files and reproduce_command_audit.py retain the evidence behind this diagnosis.


## Frozen learning-path implementation audit

The follow-up audits the actual saved MaleCNS checkpoints on the RTX 4090,
without optimizer updates or new physical rollouts. See learning_path_audit.json
and reproduce_learning_audit.py. The compiled model's named wing actuator IDs
are 14–19, exactly the decoder residual's six output channels; position-target
normalization round-trips within 5.96e-8. Replaying all 64 frames ×64 worlds of
the original first teaching batch through the preserved parent's deployment
forward reproduces its saved training predictions within 4.92e-7 per action.

On a 16-step, four-world sequence with warmed recurrent state, direct deployment
forward and the activation-recomputed training path agree within 2.69e-7 per
action. Maximum parameter-gradient difference is 4.66e-9. Gradients are finite
and reach the current height, target height, and both horizontal-error input
columns. All model parameters and buffers remain bitwise unchanged after the
audit. This rules out the tested wrong-channel, normalization, disconnected-input,
checkpoint-loading, and training-forward/backward mismatches. It does not prove
that the learned mapping is stabilizing or exclude every possible bug.

Twenty targeted existing tests also pass in 20.50 seconds: wing inertia and
force coupling, position actuator limits, native/batched PID equivalence, and
absence of a raw sensor-to-motor bypass. Those tests exercise physics separately
from the zero-physical-step frozen network audit. The network audit itself takes
11.499506 seconds. No further learning was launched.

The root cause remains unresolved. Data coverage and a weak corrective signal
are possible contributors, not a demonstrated sole explanation. The PID's
successful control of this same plant is evidence that the task is physically
controllable, not evidence that the student has learned the needed feedback.
Do not change the dynamics to address this learning failure, prescribe more time
without a diagnostic, or call reduced drift successful target following.

The exercises run together through one shared actor: 32 stationary-hover worlds
and 32 worlds divided among six directional out-and-back orders. A movement world
executes one 12-second directional exercise, including reversal, return and
settling; all six orders are not concatenated into one episode. The physical
plant is fixed. PID actions control training in this run; evaluation uses only
the saved learned actor. The user's aim is consistent, learnable flight dynamics,
not aerodynamic fidelity; the latter is not a missing requirement.
