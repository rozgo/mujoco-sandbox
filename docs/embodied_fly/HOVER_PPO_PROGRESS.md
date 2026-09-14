# Learning to regulate flight with PPO

The retained **run-11 midpoint** completes four ten-second flights. Relative to
run 05, total velocity RMS is **14.5% lower**, climb **42.7% lower**, and peak
displacement **35.3% lower**, with slightly lower horizontal RMS. It still climbs
about **61 mm**, above the 50 mm goal; stationary hover remains unfinished.
The first two seconds have slightly higher velocity error than the parent.

[Watch the retained comparison](../../previews/embodied_fly/velocity_hover_ppo_11_comparison_v1.mp4).
PID reference, run 05 and selected five-minute midpoint; same body, forces, starts,
commands, camera settings and chart scales. Four cases, 43 seconds at 1x.
The ten-minute final continuation regressed relative to midpoint and remains
archived. [Selection and measured costs](runs/velocity_hover_ppo_11/SUMMARY.md).

Five later trials remain preserved: unchanged continuation (run 06) worsened
sideways control; a vertical reward increase (run 07) reduced climb to about
67 mm but raised horizontal RMS to 14.1 mm/s. A combined velocity reward (run 08)
lowered horizontal RMS to 7.73 mm/s but increased climb to 135 mm and total RMS
to 16.43 mm/s. Recovery practice with small training gusts (run 09) increased
climb to 161 mm and total RMS to 18.69 mm/s. Removing the imitation gradient
(run 10) preserves the 30 Hz wingbeat but still yields 133 mm climb and
16.82 mm/s total RMS. None of runs 06–10 replaces run 05.

A subsequent frozen-noise diagnostic finds 32/32 ten-second survivors without
noise, 30/32 with half noise, and 18/32 with original noise. Run 11 therefore
changes only fixed exploration amplitude .003 -> .0015, preserving the original
reward and imitation. Its midpoint gives 12.76 mm/s total RMS, 9.19 mm/s
horizontal RMS and 60.73 mm climb. This is the new preferred development
checkpoint. [Frozen comparison](runs/hover_exploration_01/SUMMARY.md).

Run12 resumes that midpoint with a tighter PPO KL cap (.02 -> .005). Its
quarter/midpoint/final all survive4/4, but none improves combined hover. The
midpoint reduces climb to43.74mm while increasing horizontal RMS to11.76mm/s
and total RMS to13.61mm/s. Startup dips less and later climb is slower, so the
vertical gain is real; sideways drift prevents promotion. Run11 remains
selected. [Results and trial video](runs/velocity_hover_ppo_12/SUMMARY.md).

Sensor timing accounts for only about 0.23 mm of climb. A frozen-weight neural
probe confirms all three measured velocity signals reach the motor features.
The completed imitation ablation retains the original run-05/06 reward and
experience budget. Its first teacher gradient is only about .025% of the PPO
gradient norm before clipping/Adam. This single-run comparison does not establish
phase-sensitive imitation as the main cause. See the [phase discussion](PHASE_INVARIANT_GUIDANCE.md).
Stationary hover remains unfinished.

## The learning sequence, including failures

| Trial | Change | Training seconds | Transitions | Ten-second flights | Common early velocity RMS |
|---|---|---:|---:|---:|---:|
| 01 | Full actor PPO + light imitation | 604.368 | 1,081,344 | 0/4 | Incomplete window |
| 02 | Post-step KL guard and lower LR | 603.390 | 704,512 | 2/4 | 26.54 mm/s |
| 03 | Wider horizontal reward | 603.910 | 720,896 | 2/4 | 25.88 mm/s |
| 04 | PPO on existing wing readout | 602.166 | 1,753,088 | 4/4 | 20.28 mm/s |
| 05 | Continue same weights/optimizer/critic; retained parent | 604.488 | 1,769,472 | 4/4 | 15.59 mm/s |
| 06 | Unchanged continuation; sideways regression | 603.991 | 1,769,472 | 4/4 | 15.08 mm/s |
| 07 | Vertical reward rate 2 -> 3; mixed outcome | 603.681 | 1,769,472 | 4/4 | 13.19 mm/s |
| 08 | Combined velocity reward; climb regression | 605.275 | 1,769,472 | 4/4 | 17.58 mm/s |
| 09 | Small training gusts; climb regression | 604.224 | 1,753,088 | 4/4 | 17.76 mm/s |
| 10 | Imitation gradient off; rhythm retained, climb regresses | 603.871 | 1,769,472 | 4/4 | 16.36 mm/s |
| 11 | Half exploration; retain midpoint over final | 606.861 | 1,769,472 | 4/4 | Midpoint 16.79 / final 14.88 mm/s |
| 12 | Tighter KL limit; vertical gain, sideways regression | 604.080 | 1,769,472 | 4/4 | Midpoint 15.45 / final 14.57 mm/s |

Original imitation: 2/4 complete, common early RMS approximately 25.96 mm/s.
Early RMS uses 0–2 s across all four starts. The earlier run-05 improvement over
original imitation was 53% total and 70% horizontal RMS, using 0.2–10 s on
starts 0 and 8, which complete both before and after. The new run-11 comparison
uses all four complete flights against run 05. All velocity metrics use the declared 100 ms average. These
are four fixed development starts, not a broad robustness benchmark.

The optimizer guard fixed oversized updates but did not by itself solve flight.
Progress began when PPO focused on the existing 815 -> 128 -> 6 wing readout
(105,222 parameters), leaving the learned encoder, recurrent cell parameters and
base decoder steady. This preserves the same actor and measured connectome.
All 166,700 neurons still process the live input each actor step, and all 78
controls still come from the same actor. No PID control, new network or physics
change is inserted during evaluation. The original PID supplies a light training
anchor. The separate 1712 -> 128 -> 128 -> 1 critic predicts return for learning.

Frozen upstream weights also permit exact reuse of recorded motor-neuron features
during PPO optimization. It avoids repeated full-graph backpropagation while the
live actor continues running the full graph. Tests check replay before and after
readout updates and across resets. The physical improvement and speed improvement
are separately measured; lower training loss alone is never used as success.

## Training cost and backend

- Twelve completed PPO trials: **7,250.304584 s = 120 min 50 s**, **18,399,232 transitions**,
  **36,798,464 physics steps**, **10.222 hours of aggregate simulated experience**.
- Productive checkpoint lineage: **25 min 8 s PPO**, following **31 min 6 s
  imitation**, for **56 min 14 s selected training ancestry**. Failed pilots
  remain part of trial cost even though their weights are not ancestors.
- All pilots: **32 worlds**, **16 CPU physics threads**, RTX 4090 neural work.
  MuJoCo/mjbatch CPU physics; **not MuJoCo Warp**. **1,000 Hz physics**, **500 Hz
  actor**, four internal neural updates per actor step. No clock changes.
- Run-11 full trial: approximately **2,916 world/action transitions/s**, including
  learning, and **1.44 GiB peak PyTorch CUDA allocation**. Physics collection and
  full-brain inference dominate; cached actor optimization takes 2.69 seconds
  within the 606.86-second stage. The selected snapshot is at 301.33 seconds.
- Setup, replay checks, evaluation, rendering, transfer, tests and discarded
  smoke runs are measured separately in the per-run reports and TIME_LOG.
  An early audit-only aborted collection has no retained compute timer.
- Full fly suite after the tighter-update implementation: **256 passed**.
  The refinement also passes **33 focused sampling/replay/optimizer tests**.
  GPU replay, frozen-parameter and matched recipe checks pass. Comparison videos
  are fully decoded, inspected and opened locally.

The remaining weaknesses are residual climb and cold-start regulation. Preserve
the selected run-11 midpoint, its run-05 parent and the later final checkpoint;
the latest update is not automatically the best controller. Survival alone is
not stationary hover.

Next: a single roughly ten-minute recovery-start PPO block, with16 normal-start
worlds and16 initialized from the policy's own valid climbing/drifting states.
Restore physical state and full neural/reward histories together; retain the
same actor, reward, physics and evaluation. Promote only on joint improvement
of vertical and horizontal control without worse startup. If it fails, pause
further extensions for phase-conditioned control/interface diagnostics.
[Concrete implementation and gate](runs/velocity_hover_ppo_12/NEXT_RECOVERY_PLAN.md).

[Retained run details](runs/velocity_hover_ppo_11/SUMMARY.md) and
[checkpoint manifest](PREFERRED_HOVER.json).

```sh
open previews/embodied_fly/velocity_hover_ppo_11_comparison_v1.mp4
```
