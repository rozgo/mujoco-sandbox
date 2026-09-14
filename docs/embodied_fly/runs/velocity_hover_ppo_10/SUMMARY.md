# Removing imitation preserves the wingbeat but does not improve our best hover

The requested ablation is complete. Run 10 sets the PID imitation weight to zero
and uses the same run-05 parent, original reward and 108-rollout experience
budget as imitation-enabled run 06. **Run 05 remains the preferred checkpoint.**
Both run-10 snapshots complete all four ten-second flights but miss the combined
hover criteria. This completes the comparison, not the broader flight objective.

| Checkpoint | Total velocity RMS | Horizontal RMS | Net climb |
|---|---:|---:|---:|
| Retained run 05 | 14.92 mm/s | 9.30 mm/s | 105.99 mm |
| Continuation with imitation, run 06 | 18.65 mm/s | 15.42 mm/s | 101.04 mm |
| Continuation without imitation, run 10 | 16.82 mm/s | 9.12 mm/s | 133.49 mm |

These are means over four fixed development starts, with 100 ms averaged velocity
and the unchanged 0.2–10 s RMS interval. Run 10 improves total RMS relative to the
matching continuation but worsens it relative to the retained parent. Its
midpoint measures 18.69 mm/s total RMS, 15.72 mm/s horizontal RMS and 96.12 mm
climb. No case falls during either evaluation; survival alone is not hover.

## What the experiment tells us

Removing the teacher gradient does not destroy the learned wing rhythm. The
dominant sweep frequency remains 30 Hz on both wings in all evaluated cases.
For start 0, peak-to-peak sweep angles are about .549/.548 rad, versus
.547/.545 rad in run 05. A separate 1–10 s motion diagnostic finds similar
rapid body-speed variation: 12.52 versus 12.51 mm/s after subtracting the 100 ms
causal mean. Linear-detrended height RMS is .220 versus .239 mm. These extra
diagnostics are not new rewards or substitutes for flight acceptance.

A frozen-weight reconstruction of the first teacher replay batch measures
anchor gradient norm **.03906**, versus **155.21** for run 06's physical PPO
gradient: about **.0252%**. Its loss reproduces the recorded .000109774 value.
This covers the first update before clipping and Adam; it is not a bound on
optimizer updates or evidence about every later gradient. No weights change
during the 6.962223-second probe.

Together these results do not establish phase-sensitive imitation as the main
cause of poor regulation. Framewise imitation remains a design concern when
teacher and student phases differ, but this ablation does not justify assuming
that phase-aligned imitation will fix flight. It also does not prove imitation
is always beneficial or irrelevant: this is one training seed.

The initial rollout is not bitwise reproducible across separate executions.
The off-smoke's largest reward-rate difference from run 06 is .00013823; an
on-smoke also differs by .00010820. Both smokes are discarded and preserved.
Functional replay and unchanged upstream weights pass; the recorded core recipe
fields, parent/graph/physical hashes and total experiences match the control.
All 424 accepted updates satisfy KL .02 (maximum .01994650). There are zero
supervised imitation presentations and 55,296 diagnostic-only teacher targets.

One useful next diagnostic is frozen-policy flight with and without exploration
noise. During run 10's changing-policy training, 256 of 520 completed episodes
fail; deterministic final evaluation completes all four. Changing weights and
sampling both differ, so this is motivation for that diagnostic, not evidence
that noise alone causes the gap. Preserve the parent and separate those effects
before adding more phase or PID penalties.

## Measured work and video

- **603.871220 s training (10 min 4 s)**, **32 worlds**, **108 rollouts**,
  **1,769,472 transitions**, **3,538,944 physics steps** and
  **3,538.944 aggregate simulated seconds**. 424 actor updates, 3,456 critic
  minibatches; no changes to the existing 105,222-parameter training scope.
- CPU MuJoCo/mjbatch on 16 physics threads; RTX 4090 neural work. Same FlyBody,
  fixed MaleCNS graph, 391 observations, 78 actions, 1 kHz physics / 500 Hz actor.
- Collection 597.147245 s; actor optimization 2.519554 s, including .079860 s
  diagnostic teacher loss computation; critic 4.177458 s. No imitation gradient.
- Separate setup 14.831243 s; replay audit 3.371294 s; checkpoint IO .073152 s;
  physical evaluation 90.271678 s. Peak CUDA allocation 1,550,359,040 bytes.
- Start **2026-09-14T18:02:48.495156+00:00**;
  report **2026-09-14T18:14:26.112156+00:00**. Training source `cb80318`.
  Final SHA256 `14d93e8c82062826885d38679b7d5daeb35b93bdaef2b602fad6e2e07b81f54c`.
- Full fly tests: **241 passed**, 45 upstream warnings, **169.44 s**.
  Focused tests: 16 passed in 1.51 s. Renderer and diagnostic lint checks pass.
- Comparison render **68.552991 s**, complete decode **4.449667 s**.
  **43 s / 2,150 frames / 50 fps / 1920x1080 / 1x**. Inspected and opened
  **18:18:32 UTC**, 24 min 18 s after the approved comparison began.
  Transfer, discussion, setup, tests and rendering are not included in training.

[Watch PID / imitation on / imitation off](../../../../previews/embodied_fly/velocity_hover_ppo_10_ablation_v1.mp4).

```sh
open previews/embodied_fly/velocity_hover_ppo_10_ablation_v1.mp4
```
