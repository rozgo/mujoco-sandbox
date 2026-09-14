# Continue the improving wing-readout PPO checkpoint

Run 04 achieves four ten-second flights and reduces common first-two-second
velocity RMS from about 25.96 to 20.28 mm/s, approximately 22%. The two paired
complete cases also improve full velocity RMS from about 31.77 to 24.4 mm/s.
Peak displacement increases to approximately 217 mm because climb remains;
this is physical progress but is not yet satisfactory hovering.

Continue run 04's final checkpoint for 600 measured training seconds. Preserve
its wing-readout Adam state, critic weights/Adam, fixed exploration and actor
sampling RNG. Same trainable scope, 32 worlds, 16 physics threads, all 78 deployed
outputs, graph/body/force law, reward, anchor, clocks, PPO clip/KL limit, LR and
physical starts. Physical episodes restart from the declared dataset starts.
The independent critic minibatch shuffle restarts from the recorded seed.
There is no fresh critic warmup because the previous critic is retained.

A discarded-weight smoke verifies continuation before the measured run. The
comparison's before-PPO panel uses run 04's final capture, which matches the
actual parent checkpoint; reuse its PID capture. Preserve earlier original-parent
comparisons. Midpoint/final physical checks remain mandatory. A further block
must justify itself by actual flight; do not extrapolate from loss or throughput.
