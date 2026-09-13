# PID imitation learns the wing rhythm; handoff regresses flight

The end-of-full-teaching checkpoint flies independently for all three ten-second
review starts. Its settled wing motion is approximately 29.75 Hz with 30-degree
sweeps, close to the accepted PID's 30 Hz and 30 degrees. The prior PPO actor
used approximately 9.25 Hz and 96-degree sweeps. The same actor, physical body,
1,000 Hz physics and 500 Hz action clock can therefore produce the faster pattern.
This is measured physical motion; no PID, phase clock or oscillator runs in the
student's evaluation worlds.

It has not learned accurate hover. Nominal full-capture position RMS is 60.44 mm,
peak error 109.00 mm and altitude RMS 8.29 mm. In the last four seconds it flies
about 9.86 mm below the target. Its 0.53 mm height span in that late window is
smaller than the former bobbing, but the large drift and altitude bias remain.
All three accurate-hover gates fail. Settled vertical-speed RMS after the first
second is 12.82 mm/s, versus 57.97 mm/s for the prior actor; this does not make
the new checkpoint better at position holding.

The user reviewed this video as promising: "this looks great," while explicitly
noting the altitude loss and drift. Preserve this checkpoint/video as the wing
motion reference for the next comparison; this is not acceptance of accurate hover.

The later teacher-removal portion damages this behavior. The final checkpoint
fails at 0.090, 0.092 and 0.090 seconds on the same three starts. Teacher-command
agreement during assisted collection did not establish a stable independent
feedback controller. The end-of-full-teaching checkpoint was selected for the
predeclared five-minute physical PPO continuation. See the
[selection record](../pid_imitation_ppo_01/selection.json).

## What was trained

The existing PID generates targets from each world's current physical state.
All 78 actor outputs remain learned. Wing command MSE receives weight 1 and
the other joint-command MSE weight 0.1, keeping the PID's initial body posture
as the non-wing target. The same full MaleCNS recurrent actor is updated through
its sensory/motor interfaces and cell gain, leak and bias; measured graph wiring,
routing, observation normalization and fixed utility context remain unchanged.
There is no new deployed controller or sensory bypass.

Teacher actions execute for the first half of the wall-time budget, then their
share decreases linearly to zero by 80%; the final fifth executes only student
actions. Teacher labels continue throughout, including when its executed share
is zero. This entire stage is imitation, with no PPO or critic updates. Teacher
phase and integral are hidden from the actor; physical wing state is observed.
This hidden teacher state and the changing student state distribution are
possible contributors to the handoff failure, not established diagnoses.

The run uses 32 worlds, 16 CPU MuJoCo/mjbatch threads and RTX 4090 neural training;
64-step recurrent sequences and Adam at 1e-4. It takes **302.066 seconds** plus
7.871 seconds setup: 112 updates, 229,376 physical action transitions, 458.752
aggregate simulated seconds. Of these, 45,056 actions execute with zero teacher
share. There are 666 completed episodes and 570 failures across the complete
curriculum. Full traces retain the teacher, student and executed commands and
physical states. All 112 trace hashes and both checkpoint hashes were verified.

The 302.066 seconds includes 148.192 seconds collection, 95.683 optimization and
58.191 trace-writing/loop overhead. It is not 302 seconds of pure gradient work,
and the earlier selected checkpoint does not inherit the later failed updates.
The separate PPO stage reports its own measured time and fresh optimizer/critic.

## Review and reproducibility

- [Independent end-of-teaching actor beside PID](../../../../previews/embodied_fly/pid_imitation_teacher_pid_comparison_v1.mp4)
- [Failed handoff actor beside PID](../../../../previews/embodied_fly/pid_imitation_handoff_pid_comparison_v1.mp4)
- [Measured training statistics](MEASURED_STATS.json), [full training report](training.json),
  [teacher-stage evaluation](teacher_stage_evaluation.json),
  [handoff evaluation](handoff_evaluation.json), [wing diagnosis](teacher_stage_wing_diagnosis.json)

Both videos retain full ten-second captures at 1x, including drift and failures.
They were fully decoded, sampled frames inspected and opened on macOS at
23:48:57 UTC on September 13. Fixed matched overviews fit the entire trajectory;
same-magnification body-following detail insets are labeled. This is development
evidence, not a promoted hover release or a multi-command validation.

Source 4321a87 trains the actor; 89f02ac captures the comparisons; 67b251b renders
them. The full package suite passes 179 tests in 157.91 seconds; the ten focused
checks pass in 8.37 seconds. Test-development failures and their corrections are
recorded in [validation](validation.json). The body and force law were unchanged.

```sh
open previews/embodied_fly/pid_imitation_teacher_pid_comparison_v1.mp4
```
