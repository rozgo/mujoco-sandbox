# PID hover reference at 1,000 Hz

The user rejected the first unfiltered-force video as too jerky and requested
a PID reference controller, 1,000 Hz physics, and no further learning until the
reference video is nearly perfect. The earlier reference remains preserved.
The user accepted `pid_hover_reference_v1.mp4` on September 13: "hover looks
great." This configuration is now the accepted hover plant reference.

This stage trains no policy. `pid_hover.HoverPID` is an explicitly labeled
reference controller used to establish a controllable physical plant. It runs
at 500 Hz, with two MuJoCo steps per command. The new timestep is part of the
physical fingerprint; old checkpoints keep their original rate. Native and
batched environments construct the same body at the requested rate.

The complete fly, wing position actuators, torque caps and instantaneous force
law remain. Wing angles and speeds are read before each physics step. There is
no wing-activity average, kinematic body support or live pose/velocity override.
Wing mass/inertial coupling, collisions and aerodynamic terms remain zero.
The existing joint damping, velocity drag and attitude assistance remain declared
physical forces. Contact softness can change with timestep through MuJoCo's
default safety bound; 1 kHz is a versioned physical change, not a claim of identical
numerical trajectories to 5 kHz.

The reference PID regulates position through wing stroke amplitude, orientation
and left/right differences. It uses current position/velocity and a bounded
integral, with conditional integration at the requested-acceleration cap.
Gravity feedforward supplies the nominal support requirement. It does not cancel
the plant's passive drag. A 30 Hz sinusoidal wing target replaces the earlier
12 Hz energy-based reference motion; joint-inertia/spring/damping feedforward
and measured wing-velocity feedback improve tracking. MuJoCo's existing servos
and torque limits still determine actual wing motion.

The reference contains a trajectory phase clock. This is allowed in the plant
controller; it is not inserted into the MaleCNS actor or supplied as a learned
policy input. The law has coupled roll/yaw steering, not independent yaw control.
No general flight or takeoff capability is claimed.

Before video generation, require ten seconds of free hover, no solver warnings
or forbidden support, upright >0.99, and (after the first second) height span
<0.2 mm, peak position error <0.5 mm, vertical-speed RMS <15 mm/s. Keep the whole
cold start in the video and report its separate error. A fixed observer camera
cannot smooth away physical motion. These numerical gates support visual review;
they do not replace the user's review.

The bounded probes and rejected settings are retained. Probe 11 is selected for
the final committed-source capture: altitude gains 900/1600/50, 30 Hz wing target,
wing velocity gain 0.0008 in the inherited CGS torque convention. It produces a
0.13 mm settled height band, 0.16 mm settled peak position error and a 0.60 mm
cold-start peak error, reproduced in the final ten-second capture from `e755247`.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.pid_hover \
  --output outputs/embodied_fly/pid_hover_reference_01 --seconds 10 \
  --physics-hz 1000 --height 1.86651647 \
  --frequency 30 --kp 900 --ki 1600 --kd 50 --wing-kd 0.0008
uv run --project experiments/embodied_fly --locked python -m embodied_fly.record \
  outputs/embodied_fly/pid_hover_reference_01 hover \
  previews/embodied_fly/pid_hover_reference_v1.mp4 --camera-profile fixed
```

Both capture and rendering run on Apple Silicon CPU/native MuJoCo. The RTX GPU
is unnecessary for this small controller reference. No RL was performed during
this reference stage. The next motor-learning stage must explicitly transfer to
the accepted physical contract; utility work remains deferred.
