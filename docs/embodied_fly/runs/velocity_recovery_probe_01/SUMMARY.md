# PID correction from saved student states

All ten two-second starts pass the predeclared recovery gates on the unchanged
v5 plant. Two starts have resting wings. Eight start from complete body/joint
states recorded during the prior student evaluation, spanning 0.25–4.0 seconds
of nominal flight and two held-out start times. The teacher aligns its own
oscillator from measured wing angle/speed; this phase is never supplied to the
actor. Prior physical state and prior joint command are restored only at reset.

Every case has final-window 100 ms mean speed below 0.003 mm/s and yaw below
0.004 rad/s; upright, height and forbidden-contact gates pass throughout. The
PID stops motion near the starting altitude; it has no position-return target.
This verifies usable corrective demonstrations, not learned student recovery.
Ten CPU physics worlds, 10,000 new actions, 20,000 physics steps; capture
6.562624 seconds, setup 2.865707 seconds, total process 10.805237 seconds.
No neural optimization. Probe-only data cannot be passed to the trainer.
