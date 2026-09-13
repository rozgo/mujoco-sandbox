# Explicit position-controller initialization

Source da0997c. Migration took 9.059585 s; zero training transitions. Parent
state_hover_retention02 retains its old torque-control fingerprint and checkpoint.
The six final wing rows were initialized to resting position outputs; a zero-final
815 -> 128 -> 6 readout was added. All other original tensors and graph identities
are independently verified. Saved old/new GPU model arrays differ only in the
four declared actuator arrays. Body mechanics, contacts and force law remain equal.

This artifact is an initialization for all three commands, not a trained result
or a compatible resume of the old body. See report.json and verification.json.
