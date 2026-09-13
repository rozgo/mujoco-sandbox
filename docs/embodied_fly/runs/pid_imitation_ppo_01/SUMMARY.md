# PPO retains the learned wingbeat but does not improve position holding

The user found the end-of-teaching video promising. This continuation starts
from exactly that actor, with no PID, teacher labels or imitation loss. It
retains three independent ten-second airborne starts and the new 29.75 Hz,
30-degree wing rhythm. Five minutes of PPO does not meaningfully improve the
altitude loss or drift. Keep the earlier imitation checkpoint as the preferred
development reference for this round; neither passes accurate-hover gates.

| Nominal measurement | Before PPO | After PPO | Accepted PID |
| --- | ---: | ---: | ---: |
| Airborne review starts | 3/3 | 3/3 | 1/1 |
| Accurate-hover review passes | 0/3 | 0/3 | 1/1 |
| Altitude RMS, after first second | 8.724 mm | 8.695 mm | 0.036 mm |
| Peak position error, after first second | 108.999 mm | 109.575 mm | 0.197 mm |
| Vertical-speed RMS, after first second | 12.817 mm/s | 12.816 mm/s | 13.537 mm/s |
| Mean height error, seconds 6–10 | -9.861 mm | -9.850 mm | approximately 0 |
| Height span, seconds 6–10 | 0.532 mm | 0.545 mm | 0.104 mm |
| Mean repeated height ripple, seconds 6–10 | 0.095 mm | 0.095 mm | 0.101 mm |
| Dominant sweep frequency, seconds 6–10 | 29.75 Hz | 29.75 Hz | 30.00 Hz |

Small repeated ripple is now comparable to the PID, but it occurs around the
wrong altitude while the fly drifts horizontally. Matching rhythm is distinct
from regulating position. Tiny differences in this single training run do not
establish improvement. These are development starts used for checkpoint
selection, not an independent generalization benchmark. See the complete
[matched measurements](comparison.json) and [evaluation](evaluation.json).

## What ran

**312.290 seconds** of training plus **7.835 seconds** setup on 64 hover worlds,
16 CPU MuJoCo/mjbatch physics threads and RTX 4090 neural work. This is not
MuJoCo Warp. The 300-second budget finishes the current rollout/update, producing
14 rollouts, 458,752 actions and 917.504 aggregate simulated seconds. Collection
takes 118.908 seconds and optimization 193.374 seconds. All 128 completed
five-second episodes survive; harder reset widening remains locked.

The fixed MaleCNS graph, 399-input/78-output actor, canonical body, instantaneous
wing forces, 1,000 Hz physics and 500 Hz actions are preserved. PPO uses the same
bounded physical reward as pilot06, including the 20 mm/s vertical-speed scale.
Actor LR is 1e-7, critic LR 1e-4, exploration standard deviation/floor .001,
rollout 512 actions, recurrent sequence 128 and two epochs. No phase target,
wing oscillator or direct body force controller is added.

Because the parent is an imitation checkpoint, PPO and critic Adam start fresh.
Four critic-only fitting rollouts take 34.215 seconds **inside** the training
time, with the actor fixed. There are 46 actor updates and 112 critic updates.
Five KL stops limit further actor updates; median observed maximum KL is .0290,
with a maximum .1356 against the .03 limit. This check acts after a step and does
not roll it back. The final critic has low RMSE on its bootstrap targets but
explained variance -0.0114; this does not establish useful long-horizon credit.
Peak PyTorch CUDA allocation is 12,553,766,912 bytes.

Actor and critic contain 2,516,402 and 236,033 parameters respectively. Recorded
nonzero physical-reward gradients reach cell excitability, leak and bias.
Graph/body/routing/normalization/fixed-utility invariants, finite tensors,
checkpoint ancestry and fresh Adam step counts were verified. No training
failure windows were generated. The same package had already passed 179 tests;
this run changes no source or physical configuration.

## What this round established

Imitation taught the existing actor to generate the PID-like wing motion
independently at the current clock rate. Slowing the actor is therefore not
justified as a fix for an inability to generate 30 Hz motion. Its effective
feedback bandwidth and recurrent credit assignment still need separate study;
anatomical wiring does not determine these modeled neural time constants.

The next useful question is how to learn corrective changes to an already good
rhythm. More identical PPO time is not supported by this run. A focused follow-up
could test correction direction/gain around this actor's own wing phases, and
whether PID labels remain phase-consistent on student-generated trajectories.
The failed imitation handoff may involve hidden teacher phase or state-distribution
shift; neither cause has been isolated. No additional training or motor stage is
started in this completed comparison round.

- [Before/after PPO, same 1x physical capture setup](../../../../previews/embodied_fly/pid_imitation_ppo_before_after_v1.mp4)
- [PPO beside PID](../../../../previews/embodied_fly/pid_imitation_ppo_pid_comparison_v1.mp4)
- [Promising imitation reference](../../../../previews/embodied_fly/pid_imitation_teacher_pid_comparison_v1.mp4)
- [Exact training report](training.json), [measured statistics](MEASURED_STATS.json),
  [checkpoint selection](selection.json), [artifact verification](artifact_verification.json)

Source 67b251b trains and evaluates this continuation. The parent is
`315ef3a259ed0e74757163bf58ab22feabbc2049d24992a5028d6802805df1d3`;
the resulting actor is
`721cc7e574075f36497eba2c9bd7c4c3d91ffee99a6d6293d31bfdc7996ea560`.

Both final videos were fully decoded (500 frames, 50 fps, ten seconds, 1600x900,
1x), sampled frames visually inspected, and opened on macOS at 23:57:19 UTC on
September 13. Capturing four independent ten-second worlds took 36.643 seconds;
PID/actor and before/after renders took 67.044 and 73.779 seconds concurrently.
The review milestone is 37 minutes 15 seconds after the observed effort start;
engineering, training, transfers, evaluation and review are included in that
elapsed time. Total actual training this round is 614.356 seconds across the
full imitation run and PPO; setup is separate. The selected earlier teacher
checkpoint contains 58 imitation updates and 118,784 actions, ending at a
recorded 151.924 seconds, and does not inherit the later failed handoff updates.

```sh
open previews/embodied_fly/pid_imitation_ppo_before_after_v1.mp4
```
