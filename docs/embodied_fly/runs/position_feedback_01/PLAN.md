# Wing position actuation pilot

Started September 13, 2026, 13:23:09 UTC. Prior motor trials failed autonomous
hover and ground wing accuracy. The selected torque-model checkpoint remains
`state_hover_retention_02`; this pilot does not overwrite it.

The new `wing_position` preset changes six wing actuators to bounded absolute
position targets. It preserves the full FlyBody, zero wing mass/spatial inertia,
zero wing contacts/aerodynamics, independent angular armature, unchanged physical
joint limits, and the same measured-wing flight force law. All three commands
use this one model at 5 kHz physics / 500 Hz actions.

The illustrative servo uses kp=0.1 and kv=0.0002 in CGS torque/radian units and the
previous +/-0.03 g cm²/s² torque cap. Passive joint damping remains 0.0002. Targets
use the physical yaw [-1.5,1.5], roll [-1,1.5], pitch [-1.27,2.92] radian ranges.
This follows [MuJoCo position actuator semantics](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position).
These are engineering actuator choices, not recovered fly muscle parameters.

The actor still supplies all 78 commands. MuJoCo's actuator feedback acts every
physics substep; no task-dependent masks, wing oscillator or sensor-to-motor
bypass is added to the deployed network. The existing graph remains fixed.

Migration verifies the old physical fingerprint, initializes the six wing output
rows for position units, adds the existing 815 -> 128 -> 6 residual readout with
a zero final layer, and records the new fingerprint. Every other original tensor
is preserved. This initialization is neither training nor a successful policy.
Old checkpoints cannot silently resume against the new body.

Training: one bounded 180 s pilot; 32 worlds (11 stand, 11 walk, 10 hover), 16 CPU
physics threads, RTX GPU neural work, 32-action sequences, 2 s training episodes.
Train the wing sensor extension (lr .003) and wing residual (.0003); freeze the
original connectome dynamics and other decoder parameters. Keep task weights
4:4:1, ground wing weight 10, hover wing weight 2, ground non-wing retention 4.
Ground actions are wholly student-driven. Hover collection uses 80% teaching
commands to acquire sustained flight examples; evaluation uses 100% student.
Seed 95003. No training episode timeout establishes success.

Ground wing labels are constant resting positions. Hover labels convert the
measured-state reference torque to a servo target with a half-control-interval
velocity lead. This lead is training-only. The first reference omitted the lead:
standing passed; hover stayed upright but failed tracking (5.147 mm RMSE). With
the lead, hover passed (1.376 mm). The legacy walking reference falls in both
cases; this trial uses retained learned ground outputs for its body labels.

Review: predeclared seed 95013, complete 5 s per command, no resets/teachers, all
original physical/posture gates retained, warning and torque-limit measurements,
full trace and a 15 s 1x review video. Preserve failures. Do not promote without
independent physical success. Utility, takeoff and landing remain later work.
