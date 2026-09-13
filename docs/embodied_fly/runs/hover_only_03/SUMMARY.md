# Bounded reward: corrected incentive, hover still failed

The opt-in bounded physical reward removes the identified early-termination
incentive without changing the body, wing force law, action clock, actor or
reference PID. It keeps pilot 01's actor/Adam/exploration and starts a fresh
critic for the changed returns. The failed pilot 02 is outside its ancestry.

Measured training: **306.305 s**, **720,896 transitions**, **18 actor updates**,
**176 critic updates**, with four critic-only warmup rollouts included. All 64
worlds train hover. Source `b37a82e`. The relevant suite passes **174 tests**,
45 dependency warnings, in 150.97 s; the five focused reward/transfer/PPO checks
also pass. The new test verifies reward ordering even for large finite errors,
exact failure penalty and absence of physical-state writes by reward code.

All three deterministic starts still fail: **1.426, 1.518 and 1.842 s**. PID's
statistics remain identical to the preceding matched captures and pass its tight
gate. New training returns are not comparable to the old reward. Fixing the
incentive did not by itself establish usable hover; this actor is not promoted.

The optimization audit shows one accepted actor step per non-warmup rollout,
then KL rejection. Median detected approximate KL is **1.334**, range
**0.801–2.516**, against a **0.03** early-stop threshold. The guard observes the
effect of a prior update and cannot undo that step. This motivates the paired
smaller-learning-rate test; it is not a proof that update size explains all failure.

[Video](../../../../previews/embodied_fly/hover_only_pid_comparison_v3.mp4):
complete ten-second capture, including failure and subsequent motion, 500 frames,
50 fps, 1600x900, 1x. Fully decoded, inspected and automatically opened.
See [plan](PLAN.md), [statistics](MEASURED_STATS.json), [evaluation](evaluation.json)
and [video verification](video_verification.json).
