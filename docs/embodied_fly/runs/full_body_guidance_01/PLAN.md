# Bounded learning through the shared decoder

Started September 15, 2026 at 00:19:56 UTC. User approved model-guided learning
and a results video. Parent is the consolidated full-body checkpoint
`f2af7deb5987499f32430a841dd995c68f5ab6ca7d6a84ed4fa99b29adbc2266`.

Keep the canonical body, agile-v5 force law, 1 kHz physics, 500 Hz commands,
391 inputs, full MaleCNS core and 78 outputs. All shared decoder parameters are
eligible; there are no joint-group masks or new output branches. Upstream actor
weights and the world model remain frozen for this first local experiment.

Collect ten seconds of the acting parent's physical and raw motor-neuron
histories in eight worlds. Preserve training episodes 0/2/3/4/5 and validation
1/6/7; episodes 8/9 are reserved for complete physical checks. Cached histories
come from actual recurrent execution. They are held fixed during local gradient
updates, so predicted improvement is not a closed-loop guarantee.

Compare physics plus the learned acceleration residual against analytical-only
guidance from identical starting weights, batches, learning rate and update count.
Use 200 ms predicted trajectories and a velocity objective averaged over the
last 100 ms, avoiding phase-aligned PID angle penalties. Penalize unwanted
rotation, tilt, inadequate sustained lift, floor proximity and deviations from
the retained actor's complete 78-output command vector. No rapid-action penalty.

Initial recipe: batch 16, Adam at 3e-6, 100 updates per arm, physical checks at
50 and 100. Limit changes to 0.015 normalized action units on a fixed calibration
bank by interpolating decoder weights toward their preserved parent; this is
training-only, not a runtime output limiter. Maximum 180 s optimization per arm;
record any truncated arm rather than silently describing it as matched.

Promotion requires four complete ten-second physical flights, at least 5% lower
total velocity RMS and net climb, and no more than 5% worse horizontal RMS than
the parent. Keep parent, midpoint and final evidence regardless of outcome. If
only model loss improves, preserve the parent and diagnose model-to-loop transfer
before spending more training. Render PID, parent and the actual candidate,
include inconvenient failures, decode/check the video, and open it on Mac.
