# Wing-stop mechanics diagnostic

The compiled inherited wing joints already have limits. Their solver reference
is `(0.001, 1)`: a 1 ms time constant and critical damping. Angular ranges are
unchanged from FlyBody: yaw −1.5…1.5 rad, roll −1…1.5 rad, pitch −1.27…2.92 rad.
These are compliant constraints, not hard clipping. Failed student trajectories
overshoot them substantially; a valid inherited expert also slightly exceeds
the left yaw bound. This does not by itself identify the cause of failed flight.

The opt-in `firm` pilot changes **only the six wing limit reference parameters**
to `(0.0002, 1)`. Masses, damping, spring stiffness, angular ranges, actuator
gains/torque limits, contacts and aerodynamics stay identical. The 0.2 ms time
constant is four 50 μs physics steps, above MuJoCo's documented two-step minimum.
It is an illustrative mechanical-stop choice, not a measured fly-joint calibration.
The original model remains the default. [MuJoCo solver parameters](https://mujoco.readthedocs.io/en/latest/modeling.html#solver-parameters)

Limits act through the MuJoCo constraint solver. No pose clipping, body force,
wing oscillator or learned-controller correction is introduced. Tests compare
all unchanged physical parameters and show less overshoot under an identical
bounded actuator input. Native evaluation now measures maximum wing-limit
violation at every physical substep. Batched training still uses the original
flight preset; adoption of firmer stops is conditional on physical evidence.

Before any retraining, run the inherited flight expert with the new stops for
one second of hover and 0.3 seconds of forward flight. Then run the preserved
wing-readout student from the same declared airborne initial captures. Record
all outcomes, limit violations, warnings and models. Keep the existing height,
orientation and tracking gates. A better-looking constrained trajectory would
not establish learned flight.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.flight_teacher \
  --teacher assets/embodied_fly/teachers/flight.npz \
  --wing-pattern assets/embodied_fly/teachers/wing_pattern_fmech.npy \
  --wing-limits firm --seconds 1 --speed 0 \
  --output outputs/embodied_fly/firm_stops_teacher_hover_01
uv run --project experiments/embodied_fly --locked python -m embodied_fly.flight_student \
  --graph outputs/fly_survival/malecns \
  --checkpoint assets/embodied_fly/diagnostics/motor_wing_readout_01.pt \
  --initial outputs/embodied_fly/flight_demonstrations_01/episode_004.npz \
  --wing-limits firm --seconds 0.3 --speed 0 \
  --output outputs/embodied_fly/firm_stops_student_hover_01
```
