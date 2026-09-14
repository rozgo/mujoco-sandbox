# Existing wing readout PPO stage

The first three PPO pilots did not produce controlled hover. Run 02 fixed
oversized optimizer steps; run 03 widened the horizontal reward, with essentially
unchanged early drift and two final failures. Preserve all three.

Train the existing nonlinear wing readout (815 -> 128 -> 6, 105,222 parameters)
from the same approved first imitation checkpoint. Freeze the sensory encoder,
recurrent cell parameters, base decoder and exploration variance for this stage.
The full measured connectome still runs on current observations at every actor
step. All 78 outputs still come from that actor; there is no PID control, action
mask, added controller, or physics change. This is a temporary optimization
scope inside the same architecture, not an additional brain.

Frozen upstream weights allow exact reuse of recorded motor-neuron features for
PPO and the light original PID-label anchor. Verify cached actions/log probabilities
against the full actor before updates, including resets. Test after a readout
update that cached and full recurrent replay still agree. Retain actual neural
memory after updates; it has not become stale because its parameters did not
change. The deployed actor does not use these training caches.

Keep 32 worlds, 16 CPU physics threads, 1 kHz physics, 500 Hz actor, 512-step
rollouts, two PPO epochs, original hover rewards with run 03 horizontal width
2 cm/s, fixed .003 latent exploration, clip .2, analytic KL ceiling .02, fresh
critic, and the same four fixed evaluation starts. Initial readout Adam LR 3e-6,
with post-step backtracking. This higher proposed LR is constrained by the same
measured KL safeguard; prior full-actor LR was 3e-7. No encoder/core gradients
are claimed during this stage.

Run a short infrastructure smoke first, discard its weights, then 600 measured
training seconds from the original parent. Evaluate at halfway and completion.
Milestone remains all four ten-second flights with at least 20% lower common
first-two-second velocity RMS than the parent. Also report full drift, climb,
rotation and failures. Continue evidence-led improvement if this fails, as the
user requested; do not count optimizer throughput as physical success.
