# Smaller physical-reward updates from the airborne parent

The first all-command PPO pilot preserves standing/walking but loses hover.
It records 37 updates over 37 rollouts: the approximate-KL stopping condition
cuts every batch short after one update. Early maximum KL is 1.669 against a
0.03 stopping threshold. The threshold stops subsequent updates; it is not a
rollback or a guarantee that an accepted update stays below that divergence.

Return to position_sustain_retention01, not the failed PPO checkpoint. Repeat
the same 180-second, 32-world, five-second-episode recipe and seed98003 with
actor learning rate 3e-7 instead of 3e-6. Fresh critic and optimizers. Keep
128-step rollouts, 32-step recurrent chunks, two epochs, initial sigma0.01,
ground retention weight4, reward rates, physical thresholds, model and neural
architecture unchanged. This is a declared learning-rate comparison on the
same initial-state distribution, with measured throughput/update counts.

Evaluate mean actions for five complete seconds per command with seed97013,
no teacher, critic, sampling, resets or action masks. Existing acceptance
gates remain. Archive every failure, verify physical-return gradients separately,
and decode/inspect/open the complete video. Keep the parent preferred if this
also regresses. Learned utility and the full survival simulator remain unfinished.
