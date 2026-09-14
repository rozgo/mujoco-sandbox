# Faster physical flight, real-time playback

[Watch the complete exercise](../../../../previews/embodied_fly/pid_velocity_fast_exercise_v3.mp4).
Commands are ten times the preserved reference: 15 mm/s per translation axis
and 4.5 rad/s (258 degrees/s) yaw. MuJoCo advances at the original physical rate;
the video plays at 1x. This is a PID teacher controlling the physical fly.
No fresh MaleCNS student has been initialized or trained in this follow-up.

One continuous 71.2-second episode includes six isolated translation directions,
left/right turns in place, all twelve direction/turn pairings, planar diagonals,
climbing/descending turns, braking and hover. There are no resets or live
root-pose assignments after initialization. All 50 declared stage checks pass.

## Measured behavior

- Isolated translation: 15.000 mm/s along each requested axis.
- Isolated turning: approximately 259.5 degrees/s in either direction.
- Worst settled vector-velocity error: 0.5643 mm/s.
- Worst settled yaw-rate error: 0.0591 rad/s (3.39 degrees/s).
- Minimum upright cosine: 0.99557; minimum altitude: 16.381 mm.
- No forbidden ground contact; body stays airborne throughout the exercise.

Settled measurements use a trailing 100 ms mean during each stage's final
0.4 seconds. The raw 500 Hz velocities remain in the recording and separate
video strips. Faster-motion error limits are max(0.5 mm/s, 10% requested vector
speed) and max(0.12 rad/s, 10% requested yaw rate). Stops and hover keep the
original absolute limits. These gates were declared before the faster captures.
This single development episode is not a broad robustness benchmark.

## Physical and controller changes

`wing_motion_agile_v5` preserves body masses/inertias, contacts, joint ranges,
actuator torque limits, 1,000 Hz physics and 500 Hz controls. Wings remain
mechanically decoupled; measured wing angles/speeds generate flight forces at
every physical tick. The explicit force-law changes are:

- Yaw resistance time constant increases from 0.025 to 0.25 seconds, retaining
  the original roll/pitch damping and measured pitch-difference yaw authority.
- Differential wing-stroke orientation supplies bounded sideways thrust;
  mean orientation supplies forward thrust and sweep activity supplies lift.
  Side thrust is at most 25% of instantaneous lift.

No target, command, teacher state or trajectory enters the force model.
The teacher adds causal command-acceleration and drag feedforward through its
wing targets. Translation Kp [40,40,50], Ki [200,200,900]; roll Kp 600/Kd 40;
yaw Kp 60/Ki 30. Previous teacher and physical variants remain preserved.

Four complete faster captures improve 14/50 -> 25/50 -> 34/50 -> 50/50 stage
checks. The last two share the identical v5 physical model; only yaw feedback
changes for the final correction. Earlier failure reports remain adjacent.

## Resources and reproduction

One CPU MuJoCo/mjbatch world and one physics thread. Final capture takes
65.779003 seconds, setup 3.638934 seconds; complete process 75.703225 seconds.
It records 35,600 actions and 71,200 physics steps. No optimizer updates or
GPU neural training occur. All four captures total 284.8 simulated seconds;
the first capture's timer was lost during report serialization and is explicitly
unknown. Known capture time for runs 02–04 totals 195.151927 seconds.

Relevant tests pass: 32 velocity/PID/wing tests, including native/batched
agreement, version persistence, force directions, no forces without wing
activity, and unchanged physical body arrays. Final capture/model hashes are
verified. The report and MEASURED_STATS.json retain exact values and provenance.

See [the interface and reproduction guide](../../VELOCITY_TEACHER.md).
The preceding [slow reference](../pid_velocity_teacher_03/SUMMARY.md) and video
remain available. Review the faster teacher before fresh student learning.

The final video uses a damped close camera and a fixed-scale world XY path map,
with a heading marker and 5 mm grid. The marker denotes measured fly position;
its icon is not a body-scale rendering. Gray and green trails show actual past
motion, with green covering the most recent two seconds. This replaces the tiny
3D overview in preserved v2 using the same video frames and physical recording.
Height, heading and four velocity plots remain synchronized. Playback stays 1x.

Both versions pass full decoding and checksum checks: 3,560 frames, 50 fps,
1600x1000, 71.2 seconds. Ten sampled times and full-resolution frames were
inspected. The map version was opened for user review at 2026-09-14 05:03:41 UTC.
Base rendering took 158.471 seconds; the map overlay took 15.791 seconds with
no additional simulation. User acceptance of the faster result remains pending.
