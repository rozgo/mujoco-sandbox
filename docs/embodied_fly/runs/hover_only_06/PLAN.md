# One tighter vertical-motion reward continuation

Conditional on the phase probe not exposing a blocking timing/feedback bug,
continue pilot05 for 600 requested training seconds on 64 hover worlds, 16 CPU
physics threads, RTX4090 neural work. Same instantaneous 1 kHz body, 500 Hz
actions, 399 inputs, 78 outputs, fixed graph, one recurrent actor, LR1e-7,
critic LR1e-4, inherited .001 exploration and actor Adam, horizon512, sequence128,
two epochs, gamma .999, GAE .995, KL .03, entropy0, five-second episodes.

Only physical reward change: vertical-speed scale 5 ->2 cm/s (50 ->20 mm/s),
same weight1. This makes body bobbing less rewarding without rewarding any wing
frequency or penalizing wing movement. All other reward terms and gates remain.
Because the return objective changes, explicitly reset the separate critic and
its Adam and fit it for four rollouts with actor/exploration frozen. This fitting
is included in the measured training budget, not hidden pretraining. Actor,
actor optimizer and exploration remain inherited. Seed120306.

Evaluate final frozen actor on the same three ten-second starts and accepted
PID. Compare old/new under both reward scales offline, plus independent physical
metrics. Render/open a new 1x PID/PPO video and a matched before/after comparison.
Do not promote survival or mean altitude improvement as smooth-hover success.
No PID imitation, oscillator, action mask or new motor stage.
