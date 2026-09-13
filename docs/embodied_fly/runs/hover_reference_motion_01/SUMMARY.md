# Quieter reference flight is possible with the existing model

Analysis of preserved five-second traces, with no new learning or physics.
The original learned hover has a 1.204 cm full height span, 0.728 cm final-second
span, and a dominant 5.25 Hz height oscillation. Its wing sweep is near 10.5 Hz.
The reference from a 2 cm zero-speed start has a final-second height span of
0.0361 cm (0.361 mm) and a wing sweep near 11 Hz. Its settled height still has
an offset from the requested height; small variation does not mean zero error.

These use the same declared wing_position model/force recipe on different
native hosts and different starting heights/headings. They are not a paired
performance comparison. The contrast supports targeting learned feedback before
changing the body or flight law. It does not prove the exact mechanism of the
learner's oscillation. The reference is training-only, not a learned fly.

Run analyze.py through the isolated embodied_fly uv project to reproduce the
measurements from the preserved ignored captures. Both input trajectory/model
hashes are verified against their original reports.
