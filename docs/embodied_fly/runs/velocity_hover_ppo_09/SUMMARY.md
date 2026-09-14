# Recovery gusts did not improve calm hover

Run 09 remains airborne for all four ten-second tests, but is not promoted.
Compared with run 05, mean total velocity RMS worsens **14.92 -> 18.69 mm/s**
and climb **105.99 -> 160.75 mm**. Horizontal RMS improves **9.30 -> 8.32 mm/s**,
while vertical RMS worsens **11.67 -> 16.73 mm/s**. Midpoint total RMS is
18.15 mm/s with 151.25 mm climb. These are the existing four development starts,
using 100 ms velocity averaging and the 0.2–10 s measurement window.

The trial retained run 08's vector reward and returned to run 05's weights.
Eight training worlds remained calm and 24 received 0.35-second physical force
pulses. There were 1,248 recorded events, none in a calm world. Original calm
evaluation did not use gusts. The same body, force law, network and clocks were
preserved. All upstream weights are unchanged and recurrent replay passes.
All 505 accepted actor updates respect the .02 KL limit (maximum .01993147).

This curriculum did not establish better velocity regulation. The preferred
checkpoint is still run 05. Completing this trial does not complete the broader
flight-improvement goal. The next user-approved experiment removes the PID
imitation gradient, using the original run-05/06 reward and calm training to
isolate that change; it does not inherit these regressed weights.

## Measured costs and evidence

- **604.224114 s training**, 107 rollouts, **1,753,088 transitions** and
  **3,506,176 physics steps**. 32 worlds; CPU MuJoCo/mjbatch on 16 threads;
  neural computation on RTX 4090. 1,000 Hz physics / 500 Hz actor.
- Collection 597.026739 s; actor optimization 2.798926 s, including 0.154134 s
  imitation; critic 4.371403 s. 3,424 critic minibatches.
- Setup 15.063301 s; replay audit 3.363404 s; checkpoint IO .072635 s;
  physical evaluation 90.579135 s. Peak CUDA allocation 1,548,214,272 bytes.
- Start **2026-09-14T16:29:08.248645+00:00**;
  report **2026-09-14T16:40:46.520727+00:00**. Source `b3c8f7c`.
- Final checkpoint SHA256
  `6bb16c8bdafce69b71849e9230ec1b7bc319fd8cc97eacf4feeca8c54dc8164a`.
- Render **69.229384 s**; full decode **4.409229 s**. Video is
  43 seconds / 2,150 frames / 50 fps / 1920x1080 / 1x, inspected and opened
  **2026-09-14T17:57:14Z**. Time between training and rendering is discussion
  and idle wall time, not additional training.
- Implementation tests: 235 passed, 45 warnings, 170.19 s. The discarded
  smoke and its measured counters are preserved in `smoke_training.json`.

[Watch PID / retained run 05 / run 09](../../../../previews/embodied_fly/velocity_hover_ppo_09_comparison_v1.mp4).

```sh
open previews/embodied_fly/velocity_hover_ppo_09_comparison_v1.mp4
```
