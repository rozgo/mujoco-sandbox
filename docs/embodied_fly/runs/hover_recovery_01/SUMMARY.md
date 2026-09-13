# Measured-state reference can recover from high altitude

No learning. Twelve native CPU MuJoCo worlds use the same wing_position body,
unchanged flight forces and unchanged state-hover reference. Initial heights
are 1, 2, 4 and 10 cm; initial world vertical velocities are -20, 0 and 20 cm/s.
Every world targets 2 cm and runs five seconds without resets or pose overrides.

Eleven cases stay above the 0.5 cm safety floor, remain upright, and finish with
about 0.109 cm final-second altitude RMSE. The 1 cm / -20 cm/s case crosses the
safety floor at 22 ms, touches down, and later recovers. That case remains failed.
The recovery-specific gate was selected in the probe before execution: remain
airborne/upright and final-second altitude RMSE below 0.3 cm. This is not the
unchanged motor release gate, a learned actor, or evidence of general robustness.

The exact executed source is probe.py and its hash is in report.json. Its warning
summary selected one category, but FlyBatch.step checks every category after
each action and raises on any warning. Every step completed; all categories
therefore remained zero. The reusable probe now sums every category directly.
No body or teacher change was needed. Proceed with longer student episodes to
teach corrections from its own sustained-flight states.
