# Stronger vertical reward reduces climb, with a sideways tradeoff

All four fixed starts complete ten seconds. Increasing the vertical tracking
reward rate **2 -> 3** reduces mean climb **105.99 -> 67.41 mm (36%)**, and
vertical velocity RMS **11.67 -> 7.43 mm/s (36%)**. However, horizontal RMS
worsens **9.30 -> 14.07 mm/s (51%)** and total RMS **14.92 -> 15.91 mm/s**.
Peak displacement rises **117.10 -> 126.85 mm**. The combined hover criterion
is not met; **run 05 remains the preferred overall checkpoint**.

| Snapshot | Ten-second flights | Net climb | Vertical RMS | Horizontal RMS | Total RMS |
|---|---:|---:|---:|---:|---:|
| Parent, run 05 | 4/4 | 105.99 mm | 11.67 mm/s | 9.30 mm/s | 14.92 mm/s |
| Run 07 halfway | 4/4 | 119.30 mm | 13.13 mm/s | 7.21 mm/s | 14.98 mm/s |
| Run 07 final | 4/4 | 67.41 mm | 7.43 mm/s | 14.07 mm/s | 15.91 mm/s |

Metrics are means across the same four predetermined development starts
(0, 1, 8, 9), with the declared 100 ms average and common 0.2–10 s RMS window.
The 0–2 s velocity RMS improves **15.59 -> 13.19 mm/s**, but this shorter window
does not override the full-flight sideways regression. Every final case climbs
67.13–67.77 mm and has horizontal RMS 14.05–14.08 mm/s: none meets the requested
50 mm climb / 10 mm/s horizontal criterion. These are not broad robustness tests.

## One reward change; the same brain and flight plant

Return to run 05, not run 06. Restore actor Adam, critic/Adam and actor sampling
RNG. Increase only the vertical reward rate; keep its 0.5 cm/s width, all other
reward terms, force law, physical body, sensor/action interfaces and clocks.
Maximum live rate becomes 6 instead of 5. The first rollout updates only the
critic on the new reward, retaining its input calibration. The remaining
rollouts update the existing 105,222-parameter wing readout. All upstream actor
parameters remain identical. All 166,700 MaleCNS neurons still run in the live
control path and the same actor produces all 78 controls.

Recorded recipe comparison changes only the reward's vertical rate (plus its
version and derived maximum), explicit transition metadata and one critic
warmup rollout. No PID actuator help or new deployed network is added. A light
PID-label anchor remains part of training, unchanged. No position/height target
is supplied to the actor. The critic is training-only. Fixed .003 exploration,
PPO proposed LR 3e-6, KL ceiling .02 with rollback, 512-step rollouts and 64-step
training sequences remain unchanged. All 413 accepted updates satisfy
the analytic KL ceiling; maximum **0.01995413**.

## Measured work

- **603.681157 s training**, **32 worlds**, **16 CPU physics threads**;
  RTX 4090 neural inference/learning, CPU MuJoCo/mjbatch physics, not Warp.
  1,000 Hz physics, 500 Hz actor, four internal neural updates per actor step.
- **1,769,472 transitions**, **3,538,944 physics steps**,
  **3,538.944 aggregate simulated seconds**;
  approximately **2,931 transitions/s** including learning.
- 108 rollouts, 413 actor updates,
  3,456 critic minibatches, 54,784 imitation targets.
- Collection **597.045209 s**, actor optimization
  **2.369302 s** (includes **0.142231 s** imitation),
  critic **4.240253 s**.
- Separate setup **15.205988 s**, replay audit **3.379245 s**,
  checkpoint IO **0.070140 s**, physical evaluation **90.071512 s**.
- Training start **2026-09-14T15:16:25.988950+00:00**, report **2026-09-14T15:28:03.229708+00:00**.
  Peak PyTorch CUDA allocation **1,548,214,272 bytes**, about 1.44 GiB.
- This trial's ancestry **3676.484627 s** includes run 05
  and this new block, excluding run 06 and discarded smoke weights. The retained
  run-05 checkpoint's ancestry stays **3,072.803470 s**.
- Source `6f00631`. Final checkpoint SHA256
  `631810d364857a38e281bb54a32d172869e7a8daa93a971dfaee0b0e4478b959`.

The discarded transition smoke took **17.009556 s** and collected **49,152
transitions**. It verified critic-only warmup, preserved calibration, subsequent
actor updates, full/cached replay and unchanged upstream parameters. Full fly
suite: **231 passed**, 45 upstream warnings, **165.98 s**. The launch-path typo
aborted before graph loading completed, with zero collected transitions or
optimizer updates; corrected launch uses the original graph.

[Watch PID / run 05 / run 07](../../../../previews/embodied_fly/velocity_hover_ppo_07_comparison_v1.mp4).
The final comparison uses matched starts, chart scales and camera settings.

```sh
open previews/embodied_fly/velocity_hover_ppo_07_comparison_v1.mp4
```

The bounded continuation and agreed reward-adjustment trials are complete.
Stationary hover remains unfinished. The next design issue is keeping horizontal
and vertical control good together, rather than selecting a checkpoint solely
for survival or improvement along one axis. Preserve the current objective,
measure the tradeoffs and validate any next reward revision before training.

Video QA: full decode **4.147516 s**, render **70.045065 s**; 43 seconds, 2,150 frames, 50 fps, 1920x1080, 1x. Opening, wing-detail views, flight traces, case transitions and final results were inspected. Opened on the local Mac **2026-09-14 15:32:55 UTC**.
