# Motor command inputs

The existing Python interface uses arrays on `FlyBatch`, one row per world.
This is the current motor-learning schema, not a JSON/RPC service or proof that
all represented commands have been learned.

| Field | Shape | Units / meaning |
|---|---|---|
| `command` | `(N, 3)` float32 | Forward speed, lateral speed, yaw rate; cm/s, cm/s, rad/s |
| `requested_xy_cm` | `(N, 2)` float32 | Current target world X/Y position, centimeters |
| `requested_height_cm` | `(N,)` float32 | Current target world Z height, centimeters |

Current flight trials set `command` to zero and move the position target:

```python
env.command[i] = [0.0, 0.0, 0.0]
env.requested_xy_cm[i] = [0.15, 0.0]  # Absolute X target of 1.5 mm
env.requested_height_cm[i] = 1.8665  # Approximately 18.7 mm above z=0
```

The actor sees `command / 10`; current altitude and requested altitude each
divided by 2 cm; and horizontal target displacement transformed into anatomical
body axes, scaled by 1 cm. Horizontal displacement inputs are enabled only when
requested height is positive. Targets are ideal simulator/task inputs, not
inferred visual measurements. World +X/+Y/+Z denote forward/left/up only at the
canonical starting heading. The target remains in world coordinates as the fly turns.

These channels join measured joint positions/velocities, actuator state, body
angular/linear velocity and orientation, foot contacts, previous actions, needs,
and explicit wing angle/speed feedback in a 399-element observation vector.
The actor also receives its persistent recurrent neural state. Needs are zero
and utility selection is disabled in this motor stage.

Standing/walking/hover task IDs and round-trip route names are curriculum
metadata. No categorical route label, future target, episode timer, PID phase
clock or PID integral enters the deployed actor. The inherited motor context is
fixed; it is not a newly learned utility decision. Only the current target is
provided at each action tick, not the complete sequence or its target velocity.
The latter may be used by a separately documented reward calculation.

Output: 78 normalized motor commands, including six wing joints. The environment
maps them through the same bounded physical actuators at 500 Hz; instantaneous
measured wing states drive the declared flight forces at 1,000 Hz. A target
update never assigns the fly's physical pose or directly applies corrective force.

Implementation: `batch.py:FlyBatch.observation`, `observations.py:append_height`
and `append_horizontal_error`; sequences live in `round_trip_tasks.py`.
