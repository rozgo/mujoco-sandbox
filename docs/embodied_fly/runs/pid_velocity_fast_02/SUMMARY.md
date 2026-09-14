# Faster teacher with causal motion feedforward

Same v4 physical model and 10x speed commands as fast_01. Add causal
command-acceleration/drag feedforward through bounded wing targets, stronger
translation PI feedback, and a faster roll loop. No neural learning.

All 71.2 seconds remain airborne and upright with no forbidden contact, but
only 25/50 stage gates pass. Independent translations improve; tight sideways
turns and subsequent brakes remain inaccurate. More forceful feedback does
not solve the coupling between banking and lateral thrust during fast yaw.
The next physical variant tests independent lateral thrust from measured wing
stroke difference. This failed candidate remains preserved.

Capture: 64.595432 seconds, 35,600 action transitions. See report.json for
all timing, gates, errors and the exact physical fingerprint.
