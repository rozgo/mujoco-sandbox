# Frozen-history hover response diagnosis

The actor is replayed on all three original command histories, reproducing its
recorded actions within 5.22e-7. At six hover times, the preceding recurrent
memory is cloned for nominal, height +/-0.2 cm, and vertical-speed +/-2 cm/s
sensor probes. No physical world advances and no weights are trained. Setup
6.372643 s; diagnosis 11.665725 s.

The immediate yaw-wing target response to height is much weaker than the
reference at every sampled time. At 0.5 s it is approximately -0.0001 normalized
action per cm versus -0.0638 for the reference; at 2 s its sign differs. Several
vertical-speed responses also have the wrong sign. The sign of a correct wing
position correction depends on the wing's current phase, so a universally
positive/negative wing command is not the desired response.

These interventions are local neural sensitivity measurements, not a complete
closed-loop stability test. They motivate direct response supervision, not a
claim that all oscillation is explained. Held-input 5/25-update responses are
included in the report and do not represent real future flight states.

Implementation note: this initial diagnostic rotates inertial-frame velocity
into the anatomical frame. A subsequent teacher-equivalence test exposed the
additional COM-to-body-origin point-velocity correction (about 1.2e-5 in a
tested nominal roll target); the training implementation includes it. This
original report is preserved with its executed source hash. Recheck response
with the corrected implementation before claiming a measured improvement.
