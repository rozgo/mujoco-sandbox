# PID hover reference: 1,000 Hz physics, no wing averaging

The user requested a nearly perfect reference video before any more RL.
The new PID reference sustains ten seconds from stationary wings in an airborne
start. It controls bounded wing actuators; measured wing angles/speeds produce
forces before every MuJoCo physics tick. Body motion is integrated normally.
There is no body pose/velocity override, direct PID body wrench or activity filter.
This is a reference controller, not a learned MaleCNS result.

Committed capture source: `e755247`. The fly uses 1,000 Hz physics and 500 Hz
reference actions. The physical timestep is recorded in the contract, and both
native and batch construction support it. Anatomy, actuator caps and the
instantaneous force law remain unchanged from the preceding physical version.

## Physical result

| Measurement | Entire ten seconds | After the first second |
| --- | ---: | ---: |
| Height peak-to-peak | 0.651 mm | 0.131 mm |
| Altitude RMS error | 0.070 mm | 0.036 mm |
| 3D position RMS error | 0.117 mm | 0.098 mm |
| Peak 3D position error | 0.597 mm | 0.160 mm |
| Vertical-speed RMS | 13.575 mm/s | 13.537 mm/s |

Minimum upright measure 0.995675; no ground support or numerical warnings.
Maximum measured wing actuator torque reaches the existing 0.03 CGS cap.
The startup transient remains in the video. The revised settled gates are
height span <0.2 mm, peak position error <0.5 mm and vertical-speed RMS <15 mm/s,
with upright >0.99 and no support or solver failure. All pass this nominal case.
The user accepted the video on September 13: "hover looks great." No RL was
performed in this reference stage. Preserve this configuration for motor learning.

The PID uses gains 900/1600/50 for altitude, bounded integral and requested
acceleration, gravity feedforward, and position/velocity feedback for horizontal
correction through the wings. A 30 Hz sinusoidal wing reference replaces the
old 12 Hz energy-based reference. Joint feedforward and wing-velocity feedback
0.0008 improve startup/tracking. This phase clock is explicit reference machinery;
it is neither inside the force law nor part of a trained actor. The force law
still couples roll/yaw steering, so this is not proof of unrestricted flight.

Eleven bounded development probes are retained in `probes/`, including settings
that missed the tighter gates. Probe 11 supplied the selected parameters.
The final run reproduces those results from committed source. Improvements
cannot be attributed to the timestep alone: the reference control also changed.

## Cost and verification

- **Zero training or optimizer updates.** One Apple Silicon CPU physics world;
  the RTX GPU is unnecessary for this controller reference.
- Final setup 1.370181 s; final capture 3.906184 s; 5,000 action transitions and
  10,000 physics steps over ten simulated seconds.
- Eleven probes: 80 aggregate simulated seconds, 31.921763 s stepping in total.
  Some short probes overlap in wall time; do not add their durations as elapsed
  development time or claim a matched throughput benchmark.
- Full relevant suite: 168 passed, 45 dependency warnings, 140.61 s. Focused physics,
  PID and replay checks: 27 passed in 15.70 s. Source lint passes. A separate native/batch
  trajectory check at 1,000 Hz also passes (one test, 2.37 s).
- Tests show bounded commands/torques, no controller writes to physical state,
  exact force response without temporal history, retained mass decoupling,
  binary force-law identity and correct physics tick counts. With wing forces
  cut, the same controller cannot prevent a fall.

[Review video](../../../../previews/embodied_fly/pid_hover_reference_v1.mp4):
ten seconds, 500 frames, 50 fps, 1600x900, 1x. A fixed camera has exactly zero
vertical tracking motion; eye cameras and numeric position/altitude errors are
synchronized to the capture. Rendering takes 51.023093 s. Fully decoded and
eight frames visually inspected, including startup. Opened automatically on Mac.

See [recipe and reproduction commands](../../PID_HOVER_REFERENCE.md),
[statistics](MEASURED_STATS.json) and [video verification](video_verification.json).
