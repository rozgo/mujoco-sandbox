# User-authorized PPO continuation

During timing01 the user asked to double down on hover PPO with more time or
resources. Continue the completed timing01 checkpoint for **another 600 seconds**.
Retain its actor, critic, both Adam optimizers and exploration state. Preserve
timing01 independently and evaluate/show its nominal three-command video.

Keep the same physical body, graph, observations/actions, rewards, 512-step
rollouts, 128-step recurrent gradients, .999 discount, .995 GAE lambda, two
epochs, actor LR 1e-6, critic LR 3e-4 and KL limit .03. No imitation or teacher.
The trained critic does not repeat the initial warmup (`--critic-warmup-rollouts 0`).

Increase to **64 worlds** if timing01's measured peak CUDA allocation permits:
16 stand, 16 walk, 32 hover; still 16 CPU MuJoCo/mjbatch physics threads and RTX
4090 neural training. Collection and optimization remain full-world batches
with activation recomputation. This is not a MuJoCo Warp run. If allocation
fails, preserve the failed attempt and resume timing01 with 32 worlds instead;
never hide or charge such an attempt to successful training ancestry.

Seed 99603. New physical episodes and recurrent states start from reset.
The same declared perturbation curriculum widens again over the first ten
simulated seconds per world of this continuation. Command proportions remain
unchanged. Optimizer/critic preservation does not imply resuming live episodes.

Capture timing01's nominal review before launching timing02 to avoid GPU
evaluation contention; render/inspect/open it on the Mac during continuation.
Evaluate timing02 with the same nominal five-second review and the three
predetermined ten-second starts from timing01's plan, paired with the parent.
Include every failed attempt. The first checkpoint remains available even if
the continuation regresses. Decide on further work from the physical results.
