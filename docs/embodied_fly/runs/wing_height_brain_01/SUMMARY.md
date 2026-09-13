# Height feedback and longer recurrent supervision

The actor gained two declared altitude inputs through its existing sensory
encoder (395 → 397 inputs), with neutral weight migration. The first strict CUDA
migration check failed before training; the retained repeat calibrated the tiny
change against unchanged-parent CUDA variation. Both reports remain available.
No physical acceptance threshold changed.

The RTX 4090 pilot used eight offline recurrent sequences, 128 supervised steps
and 64 burn-in steps at 500 Hz. Actual optimization took **181.722557 s**, setup
5.930013 s and evaluation 1.698960 s. There were **106 updates / 108,544 reused
examples / 14,000 distinct training frames**, zero live physics worlds during
fitting, and 10,337,956,352 bytes peak CUDA allocation. The same graph, motor
routing, full output space and flight law were retained.

Validation MSE improved 0.0207893 → 0.0179357, but both two-second flights failed
(hover/forward root RMSE 18.818/48.691 mm). All six original-model ground cases
fell. Forcing the same checkpoint's activity context to explore also failed hover
(21.249 mm RMSE). This intervention is explicitly ineligible for policy acceptance.
Neither added altitude feedback nor this activity intervention rescued flight.

The checkpoint, original reports, migration evidence and causal capture
verification are preserved. The user subsequently requested motor learning only
and the same physical body across stand/walk/hover; see [the current
curriculum](../../MOTOR_FOCUS.md). This failed candidate is not the new parent.
