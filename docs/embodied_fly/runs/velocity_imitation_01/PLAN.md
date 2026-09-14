# Fresh velocity imitation: first one-minute burst

Authorized September 14, 2026. First observed work timestamp 05:53:27 UTC.
Preserve previous policies and the accepted `pid_velocity_fast_04` reference.

- Exact `wing_motion_agile_v5` physical contract, 1,000 Hz physics and 500 Hz
  action/observation rate; four recurrent core updates per action.
- Ten complete continuous 71.2-second teacher episodes: eight training and two
  held-out validation episodes, all containing all 50 stages. Small initial
  heading and teacher wing-phase variations. No physical resets between stages.
- Collect with ten CPU MuJoCo/mjbatch worlds/threads. Cache actual observations
  and PID joint targets; use only the eight training episodes for gradients.
- Fresh random encoder/readouts, neutral cell excitability/leak/bias, seed
  121101; measured MaleCNS graph unchanged. No earlier policy, normalization,
  optimizer or recurrent state is inherited. Utility heads remain inactive.
- RTX 4090, Float32, activation recomputation, 64 sequences per optimizer update,
  128 supervised steps and up to 64 preceding recurrent warm-up steps. Windows
  remain contiguous within a recording; each batch covers all 50 stage IDs.
  Warm-up reconstructs state from recorded observations with current weights;
  truncated backpropagation does not replay a complete 71.2-second brain history.
- Adam LR 1e-4, gradient norm cap 1. Loss is six-wing-command mean squared error
  plus 0.1 times the other 72 joint/adhesion command mean squared error. No PPO,
  critic, automatic teacher/student blending, or motor masks in this burst.
- Train for 60 seconds of measured loop wall time, finishing the active update.
  Save full optimizer and sampler states to continue the same run later.
  Time collection, initialization, validation, evaluation and rendering separately.
- Report fixed held-out sequence losses before/after; physical student evaluation
  uses a complete exercise unless the student fails. Count unreached stages as
  unreached, not successful. No live pose/velocity corrections or PID assistance.
- Produce a synchronized 1x teacher/student video with the same model, initial
  state and commands. Show failures, retain traces, and automatically open it.

The 60-second budget is a diagnostic allowance, not a promise of learned flight.
Continue only after inspecting the first result; do not silently extend training.
