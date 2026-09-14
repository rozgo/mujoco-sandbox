# Goal-directed reward: measured PPO result

The new reward PPO trial **does not pass the declared progress gate**. Accurate route tracking and settling remain unsolved; preserve the preferred parent.
This is a measured learned-policy result, separate from the preceding reward
preference audit. Both checkpoints and all seven review cases are preserved.

[Watch the new learned actor and PID hover reference](../../../../previews/embodied_fly/round_trip_reward_v2.mp4).

## What changed

From the same preferred imitation parent, repeat the preceding PPO command with
only `--flight-tracking-reward` added. The two velocity scores reward movement
toward the requested target and slowing near it, incorporating target movement.
All other reward weights, physical dynamics, actor/critic architecture, training
settings and seed are preserved. Both trials initialize a fresh critic and
optimizer. No PID commands, imitation updates or external oscillator run in PPO
or learned evaluation. The PID is separately labeled in the eighth video panel.

One shared actor controls all seven cases. Its encoder, motor decoder and cell
dynamics are trained; measured graph routing, utility/intention layers and fixed
observation statistics remain unchanged. Exact tensor/config checks are recorded
in artifact_verification.json. Same canonical wing_position body, 1,000 Hz
physics and 500 Hz actions, with wing state producing forces every physics tick.

## Matched frozen evaluation

| Metric | Preferred parent | Previous reward PPO | New reward PPO |
|---|---:|---:|---:|
| Six-route mean position RMS, mm | 74.103 | 75.870 | 75.762 |
| Stationary-hover position RMS after 1 s, mm | 77.357 | 78.734 | 78.707 |
| Complete waypoint/return/hold gates | 0/7 | 0/7 | 0/7 |

Route RMS improvement relative to the preferred parent: **-2.238%**
(negative means worse). Relative to the preceding reward PPO: 0.143%.
The new actor stays airborne in 7/7 complete
12-second captures. The same body, starts, target paths and measurement windows
are verified; unchanged parent captures are reused. No evaluation resets or
exploration, and no cases removed. These are fixed development cases from one
training seed, not generalization or statistically significant superiority.

## Actual training and resources

- Training: **399.676774 seconds**, plus 7.961510 s setup.
  The requested 300 s finishes its current full rollout/update cycle.
- 64 CPU MuJoCo/mjbatch worlds, 16 physics threads;
  RTX 4090 neural training. This fly trial does not use Warp physics.
- 524,288 world/action transitions; 1048.576 aggregate
  simulated seconds; 1311.780 transitions/s including learning.
- 4 rollouts, 64 actor updates, 1024 critic updates;
  0 KL stops. Two critic warmup rollouts are included in training time.
- Collection 138.963871 s; optimization 260.711175 s.
- 65 completed training episodes,
  5 physical failures,
  0 complete training tracking successes.
- Actor 2,516,402 parameters; critic 236,033.
  Peak PyTorch CUDA allocation 9,073,805,312 bytes
  (8.45 GiB); not total device memory usage.
- New frozen evaluation capture 45.589914 s, setup 7.185971 s.
  Training, evaluation, video rendering and development time are recorded separately.

Critic explained variance is measured after fitting the collected training
rollout, not on held-out outcomes. A better fit or preference-audit score alone
is not a better fly; the frozen physical result determines progress.

## Reproduction and evidence

Use the exact training command in command.json, source commit in training.json,
and preserved parent checkpoint. The sole reward implementation was validated
by the preceding 195-test suite; this run adds no source behavior changes.
Evaluate the new actor with `embodied_fly.round_trip_evaluate`, then render its
capture with `embodied_fly.round_trip_video --reference` pointing at the preserved
round_trip_reference_02 capture. Compare reports using
`../round_trip_ppo_01/reproduce_comparison.py` with explicit parent/child/output
arguments. Video metadata and video_qa.json record rendering and full decode.
