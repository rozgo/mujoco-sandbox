# Goal-directed reward PPO comparison 02 — predeclared

Run the implemented, audited `--flight-tracking-reward` variant to produce a
learned result and a new video. Preserve the preferred imitation parent and the
unsuccessful `round_trip_ppo_01` child. Do not promote either child without the
existing physical tracking gates. This is a single-seed development comparison.

Use the exact `round_trip_ppo_01` command, changing only the output name and
adding `--flight-tracking-reward`. Start from the same preferred
`pid_imitation_01_teacher_stage.pt` parent. The critic and optimizer start fresh
as in that prior comparison; no critic architecture change, teacher, imitation,
extra sensory input or new physical controller is introduced.

The two velocity reward scores now prefer target motion plus a bounded
correction toward the target, tapering to zero at a held target. All other
reward terms, weights, scales, termination rules and the physical plant remain
unchanged. See `../flight_reward_01/SUMMARY.md` for the precise reward and audit.

Keep 64 worlds (32 stationary hover, 32 ordered routes), 16 CPU physics threads,
RTX 4090 learning, 1,000 Hz physics and 500 Hz actions. Keep the same seed,
12-second episodes, randomized path amplitudes/timing, 2,048-step horizon,
128-step sequences, two actor epochs, 16 critic epochs, two critic warmup
rollouts, actor LR 1e-7, critic LR 1e-4 and exploration standard deviation 0.003.
Request 300 seconds and record the actual completed-cycle training time.

Reuse the unchanged frozen parent and previous child evaluation captures.
Capture the new actor alone on all seven matching 12-second nominal cases,
including failures. Retain the exact existing completion and progress gates:
at least 10% lower six-route position RMS than the preferred parent, no more
than 5% stationary-hover regression, and no new physical failure. Full
acceptance still requires all seven tracking/settling gates. The old reward
child is an additional matched comparison, not a replacement baseline.

Render all seven new learned cases with the accepted PID hover reference,
inspect and fully decode the video, and open it automatically. Archive measured
training, evaluation, checkpoint and video evidence with actual separate times.
The existing reward suite passed 195 tests before this run; no source behavior
has changed since that validation, so do not repeat the full suite for these
run documentation additions.
