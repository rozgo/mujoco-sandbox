# Final bounded hover push

Started September 15, 2026, 02:11:53 UTC. The user authorizes one more hover-flight
attempt, then stopping if it cannot learn. No open-ended automatic extensions.
Preserve the current shared-decoder checkpoint and every earlier video.

First test one explicit v6 force-law variant, with the same body, contacts,
joint dynamics, torque limits, 1 kHz physics and 500 Hz actor. Measured sweep
activity alone controls lift; measured pitch controls yaw without changing lift
efficiency. Horizontal thrust scales with body weight times wing engagement,
instead of instantaneous lift. Unequal sweeps still produce roll, but no
incidental yaw. Damping, upright response and maximum lift are unchanged.
Stationary wings produce no flight force. No target, desired velocity, phase
clock or teacher state enters the force law.

Validate native/batch agreement and force directions/decoupling. Evaluate PID
and the unchanged actor on four ten-second starts. Adopt v6 only if PID completes
all four with mean velocity RMS below 1 mm/s and the unchanged actor completes
all four. If either fails, preserve those results and use the original v5 for
the single training block. Do not tune multiple physical variants in this push.

Train **one shared 78-output decoder** with PPO using actual full-MaleCNS/body
rollouts. Upstream encoder and modeled neural parameters stay frozen. All decoder
weights, normalization and output rows are eligible; no wing branch, output masks,
teacher execution or direct sensor-to-action path. Raw motor-cell caching is
conditional likelihood replay of collected experience, not prediction of future
neural activity. Fresh experience follows every policy update.

Hard training cap: 1,200 seconds, finishing at most the active rollout. Evaluate
at 300, 600 and 1,200 seconds, excluding evaluation/serialization from training
time. Use 32 worlds, 16 native physics threads, RTX 4090 neural work, 512-step
rollouts, 64-step minibatches, two actor/four critic epochs. Fixed tanh-latent
noise 0.0015, Adam LR 3e-6, PPO clip 0.2, post-step KL limit 0.005 with exact
Adam rollback, gradient cap 1. Discount/GAE timescales remain 2 s/0.25 s.
All worlds start cold from training episodes 0–7, with actual full neural memory
retained between rollouts and zeroed only at resets. No recovery-bank mix.

Use the already-audited coordinated full-vector velocity reward, 5 mm/s scale,
100 ms causal mean, positive survival/upright/angular terms and one terminal
penalty. No phase-matched action loss or rapid joint-motion penalty. Fresh actor
Adam is required by the consolidated topology; fit a fresh independent critic
for the chosen plant/reward, with one critic-only rollout before actor updates.
No old imitation labels are applied to changed physics.

Stop training early if an evaluation has zero complete flights or mean velocity
error exceeds the same-plant untrained transfer by more than 25%. Retain all
snapshots. Promotion requires all four flights, at least 5% better velocity RMS
and absolute net climb, and at most 5% sideways regression versus both the old
v5 baseline and the same-plant unchanged-weight baseline. Stationary hover is
separate: all four flights with velocity RMS <=5 mm/s and absolute climb <=10 mm.

Render original v5, same-weight physical transfer, trained policy and the
appropriate PID reference at 1x. Label different physics explicitly; do not
attribute transfer gains to learning. Preserve failures and show matched scales.
Decode and inspect the video, open it locally, and report a clear verdict. If
the bounded trial fails, recommend pausing this learning effort rather than
automatically launching another experiment.
