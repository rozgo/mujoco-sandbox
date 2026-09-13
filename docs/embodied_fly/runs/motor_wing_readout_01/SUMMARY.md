# Wing-output calibration — useful diagnostic, flight still fails

The same angle01 actor changed only the six existing wing rows/biases in its final
motor layer: **1,542 parameters**. Every other parameter, including the encoder,
intrinsic core, utility head, motor hidden layer and non-wing output rows, remains
bitwise unchanged. No module, sensor bypass or oscillator was added. Runtime
commands still pass through the complete graph actor.

The second frozen-feature diagnostic took **12.223347 s**, including 8.674879 s
graph replay and 0.923476 s cache writing. The existing 256-unit motor hidden layer
already contains linearly decodable wing information (bounded probe MSE 0.006600).
Calibration used **10.000287 s** of RTX 4090 optimization after **1.266937 s** setup,
20,126 minibatch updates and 0.057808 s evaluation/save. Minibatches sampled 1,024
cached frames from **9,000 distinct training frames**. The 20,609,024 sampled
examples are repeated optimization samples, not additional simulated experience.
There were no live physics worlds during this fit. Peak CUDA allocation was
30,985,728 bytes; the earlier feature-replay workload is separate.

Whole-episode held-out wing MSE fell **0.093234→0.006163**. Both 0.3-second physical
flight tests still failed (hover/forward root RMSE 8.659/16.284 mm). Over the first
30 ms, sampled upward passive force was **0.2306 body weight**, compared with
0.0547 in angle01 and 0.9630 in the inherited expert. Wing amplitudes also became
excessive; stronger motion did not establish a stable wingbeat. These are sampled
control-boundary forces, not physical-substep averages.

Five of six fixed walking cases remained stable; slow walking fell. The continuous
walk/stop/resume was stable with valid support, but stop/resume tracking gates
failed. Existing ground action parameters are unchanged and the stage masks wing
outputs, so different trajectory outcomes cannot be claimed as learned ground
improvement or degradation. Known sparse CUDA variability and fragile contact
trajectories remain relevant; all observed outcomes are retained without
attributing this particular difference to an untested cause.

Keep the selected angle01 video and online01 reference. Next collect short
teacher-assisted corrections on states visited with the new wing outputs, then
calibrate those same existing rows with original and corrective histories. The
teacher remains training-only; final evaluation must run without assistance.
