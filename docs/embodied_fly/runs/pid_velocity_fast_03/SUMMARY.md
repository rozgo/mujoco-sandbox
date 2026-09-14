# Independent lateral thrust solves faster translation

The v5 model adds bounded lateral thrust from measured wing-stroke angle
difference. Body mechanics, actual wing torque limits and all clocks remain
unchanged. The same 10x-speed exercise stays airborne, with no forbidden contact.
Every translation-error check passes, including all simultaneous sideways turns.

34/50 combined stage gates pass. Remaining failures are yaw braking: measured
turn rate overshoots by about 0.30 rad/s in the early settled window. The next
teacher increases yaw proportional feedback and reduces integral accumulation;
no physical model or acceptance gate changes are needed for that correction.

Capture took 64.777492 seconds. The complete 71.2-second trajectory, physical
fingerprint, source commit and all results remain recorded. No neural training.
