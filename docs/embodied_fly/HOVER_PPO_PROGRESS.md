# Learning to regulate flight with PPO

The retained run-05 checkpoint completes four ten-second flights. Velocity error is
approximately **53% lower**, horizontal velocity error **70% lower**, and peak
displacement **21% lower** than the original imitation checkpoint on the two
matched complete flights. It still climbs about 106 mm in ten seconds. The
initial sustained-flight/reduced-error milestone is reached; stationary hover
remains unfinished.

[Watch the full comparison](../../previews/embodied_fly/velocity_hover_ppo_progress_v1.mp4).
PID reference, original imitation and final PPO; same body, forces, starts,
commands, camera settings and chart scales. Four cases, 43 seconds at 1x.

Four later trials remain preserved: unchanged continuation (run 06) worsened
sideways control; a vertical reward increase (run 07) reduced climb to about
67 mm but raised horizontal RMS to 14.1 mm/s. A combined velocity reward (run 08)
lowered horizontal RMS to 7.73 mm/s but increased climb to 135 mm and total RMS
to 16.43 mm/s. Recovery practice with small training gusts (run 09) increased
climb to 161 mm and total RMS to 18.69 mm/s. None replaces run 05.
[Latest completed comparison](runs/velocity_hover_ppo_09/SUMMARY.md).

Sensor timing accounts for only about 0.23 mm of climb. A frozen-weight neural
probe confirms all three measured velocity signals reach the motor features.
The next approved test removes the PID imitation gradient while retaining the
original run-05/06 reward and experience budget. See the [run-10 plan](runs/velocity_hover_ppo_10/PLAN.md)
and [phase discussion](PHASE_INVARIANT_GUIDANCE.md). Flight improvement remains
unfinished; the completed gust trial is not a success claim.

## The learning sequence, including failures

| Trial | Change | Training seconds | Transitions | Ten-second flights | Common early velocity RMS |
|---|---|---:|---:|---:|---:|
| 01 | Full actor PPO + light imitation | 604.368 | 1,081,344 | 0/4 | Incomplete window |
| 02 | Post-step KL guard and lower LR | 603.390 | 704,512 | 2/4 | 26.54 mm/s |
| 03 | Wider horizontal reward | 603.910 | 720,896 | 2/4 | 25.88 mm/s |
| 04 | PPO on existing wing readout | 602.166 | 1,753,088 | 4/4 | 20.28 mm/s |
| 05 | Continue same weights/optimizer/critic; retained | 604.488 | 1,769,472 | 4/4 | 15.59 mm/s |
| 06 | Unchanged continuation; sideways regression | 603.991 | 1,769,472 | 4/4 | 15.08 mm/s |
| 07 | Vertical reward rate 2 -> 3; mixed outcome | 603.681 | 1,769,472 | 4/4 | 13.19 mm/s |
| 08 | Combined velocity reward; climb regression | 605.275 | 1,769,472 | 4/4 | 17.58 mm/s |
| 09 | Small training gusts; climb regression | 604.224 | 1,753,088 | 4/4 | 17.76 mm/s |

Original imitation: 2/4 complete, common early RMS approximately 25.96 mm/s.
Early RMS uses 0–2 s across all four starts. The quoted 53% overall velocity and
70% horizontal reductions use 0.2–10 s on starts 0 and 8, which complete both
before and after. All velocity metrics use the declared 100 ms average. These
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

- Nine completed PPO trials: **5,435.492479 s = 90 min 35 s**, **13,090,816 transitions**,
  **26,181,632 physics steps**, **7.273 hours of aggregate simulated experience**.
- Productive checkpoint lineage: **20 min 7 s PPO**, following **31 min 6 s
  imitation**, for **51 min 13 s selected training ancestry**. Failed pilots
  remain part of trial cost even though their weights are not ancestors.
- All pilots: **32 worlds**, **16 CPU physics threads**, RTX 4090 neural work.
  MuJoCo/mjbatch CPU physics; **not MuJoCo Warp**. **1,000 Hz physics**, **500 Hz
  actor**, four internal neural updates per actor step. No clock changes.
- Retained run-05 stage: approximately **2,927 world/action transitions/s**, including
  learning, and **1.44 GiB peak PyTorch CUDA allocation**. Physics collection and
  full-brain inference dominate; cached readout optimization takes 2.65 seconds
  within the 604.49-second stage.
- Setup, replay checks, evaluation, rendering, transfer, tests and discarded
  smoke runs are measured separately in the per-run reports and TIME_LOG.
  An early audit-only aborted collection has no retained compute timer.
- Full fly suite after the imitation-ablation implementation: **241 passed**. GPU replay, frozen-parameter and resume checks
  pass. Every comparison video was fully decoded, inspected and opened locally.

The remaining weakness is joint regulation of vertical and horizontal motion. The retained checkpoint reduces the
extra climb of run 04, but still climbs more than the original imitation parent.
Preserve this result as the starting point for that work; avoid restarting from
an unrelated checkpoint or treating survival alone as hover.

[Retained run details and reproduction](runs/velocity_hover_ppo_05/SUMMARY.md).

```sh
open previews/embodied_fly/velocity_hover_ppo_progress_v1.mp4
```
