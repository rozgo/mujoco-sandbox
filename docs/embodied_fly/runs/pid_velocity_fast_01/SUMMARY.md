# First faster physical flight capture

Commands are ten times the preserved reference: 15 mm/s per translation axis
and 4.5 rad/s yaw, with the same 71.2-second exercise and real-time clock.
The explicit v4 plant reduces yaw resistance, preserving roll/pitch damping,
body mechanics, wing-force gains and actuation limits.

The complete physical capture remains upright and airborne, with no forbidden
contact. Only 14/50 declared stage checks pass. Braking carries residual motion
and mixed translation/turns lag; this is development evidence, not the final
fast teacher. The original slow reference remains preserved.

The report writer failed after saving the complete trajectory because a NumPy
boolean was not JSON serializable. The report was recovered from the saved
capture without repeating or changing physics. Original capture/setup timing
was not persisted and is reported as unknown, rather than reconstructed from
file timestamps. The serialization bug is fixed in source.
