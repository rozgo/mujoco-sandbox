# Thirty-minute velocity imitation continuation

User authorization: "do the full train and report back." First observed work
timestamp 2026-09-14 06:18:37 UTC. One continuous 1,800-second learning allowance;
finish the active update. Setup, validation, checkpoint writes, final physical
evaluation and rendering have separate measured timers.

Continue `velocity_imitation_01.pt`, SHA256
`f60f58bfa346113967ab09903c3ebd9eb6c24351dca8e8cd8dc961045d2760ab`.
Retain its Adam moments, sampler state and all learned parameters. No fresh
random restart. Fixed MaleCNS graph and exact accepted `wing_motion_agile_v5`
physical contract. Same cached dataset: eight complete training episodes,
two held-out episodes, all 50 movement/braking/hover stages in every episode.

Only intentional learning change: 8/64 replay sequences start at physical time
zero, with resting wings and zero neural memory, one from every training
episode. The other 56 sequences cover all 50 stage IDs every update. No resets
are inserted into the recorded physical trajectories. Padded context before
time zero is masked and cannot alter recurrent state. This is explicitly
recorded as a sampler transition, with no permission to change other settings.

Unchanged: 64 parallel sequences, 128 supervised steps, 64 context steps,
four internal recurrent updates, Float32, activation recomputation, Adam LR
1e-4, gradient norm cap 1, loss = six-wing-command MSE + 0.1 times other-joint
MSE. No PPO, critic or inherited reward functions. The simulator uses 1,000 Hz
physics and 500 Hz commands, with wing forces at each physics tick.

Save complete continuation checkpoints every 300 learning seconds without
restarting the optimizer. Complete the full allowance unless numerical failure
or an external interruption makes it impossible. Do not extend the budget or
change learning settings based on intermediate loss. The final checkpoint is
the primary result, not the retrospectively best-looking intermediate.

At the end, compare fixed held-out windows with the first-minute checkpoint,
including the constant mean-wing-pose baseline and wing-motion correlation.
Run student-only physical episodes on the exact teacher plant and matched
starts, preserving failures and unreached stages. Produce and open a 1x
PID/student comparison; retain all previous videos. Intermediate checkpoints
remain available for diagnosing progress without interrupting this training run.
