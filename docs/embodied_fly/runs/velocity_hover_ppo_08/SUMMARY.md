# Vector reward correction did not improve overall flight

Run 08 completes all four ten-second flights, but **is not promoted over run 05**.
Final total velocity RMS worsens **14.92 -> 16.43 mm/s** and net climb
**105.99 -> 134.63 mm**. Horizontal RMS improves **9.30 -> 7.73 mm/s**, while
vertical RMS worsens **11.67 -> 14.49 mm/s**. Peak displacement rises
**117.10 -> 136.83 mm**. The midpoint likewise misses the combined goal:
total RMS 15.42 mm/s, horizontal 8.34 mm/s and climb 116.67 mm.

These are means over the four fixed development starts, with the declared
100 ms average and 0.2–10 s RMS interval. Survival alone and a single-axis
improvement do not complete the restored goal. Run 05 remains the retained
checkpoint and the goal remains active.

## What changed and what was learned

One vector-velocity reward replaced the separate axis terms. Its maximum rate
and all non-velocity terms remained unchanged. Offline scoring ranks the saved
run-05/06/07 flights in the intended total-error order, addressing a demonstrated
reward mismatch. That correction did not by itself produce better RL flight.
The same body, graph, observations, commands, actions, clocks, trainable wing
readout and PPO settings were preserved. One critic-only rollout adapted to
the changed reward; critic calibration remained fixed.

All 491 accepted actor updates satisfy analytic KL .02, maximum
0.01984776. Full/cached replay passes and all upstream
actor parameters remain unchanged. Full fly tests: 234 passed in 165.88 s.
The discarded transition smoke took 16.653117 s and produced ten actor updates;
its weights are not ancestors of the full run.

## Diagnostics after the unsuccessful trial

The evaluator refreshed MuJoCo derived values before reading the actor inputs,
whereas collection used values returned by the last physics step. MuJoCo's
[simulation-loop documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#simulation-loop)
explains why derived values can precede the integrated state. A frozen run-05
comparison using collection timing changes climb only **about 0.23 mm**, from
105.99 to 105.77 mm, with essentially unchanged total velocity RMS (14.92 versus
14.93 mm/s). This difference is not the main cause. The diagnostic took
**45.700021 s**; it changed no weights. Existing evaluation remains preserved.

A second frozen-weight probe perturbs only measured forward/left/up velocity
by +/-0.1 cm/s for 2–100 ms at 0.4, 2 and 5 seconds of a captured flight.
History replay reproduces recorded actions within **3.58e-7**. The three inputs
produce independent motor-feature responses; at 100 ms their derivative singular
values are approximately **0.26–0.39**, with identical duplicate baselines.
The signal is present. Wing output sensitivity to vertical input is smaller,
but different plant axis gains prevent treating that alone as a decoder defect.
This is a neural response probe, not a physical recovery demonstration or proof
that the encoder is sufficient for every flight skill. Probe time **19.524581 s**.

The next change supplies varied physical recovery experience while retaining
this network: eight calm worlds plus 24 with small force pulses during training.
The original evaluation remains disturbance-free. See the run-09 plan. No
completion claim is made for the overall flight-improvement goal.

## Measured work and video

- **605.275283 s training**, **32 worlds**, 16 CPU MuJoCo/mjbatch
  physics threads, RTX 4090 neural computation. 1 kHz physics / 500 Hz actor.
- **1,769,472 transitions**, 3,538,944 physics steps,
  3,538.944 aggregate simulated seconds.
  108 rollouts, 491 actor updates, 3,456 critic minibatches.
- Collection 598.256435 s, actor optimization 2.710560 s
  (includes 0.144033 s imitation), critic 4.281198 s.
- Separate setup 14.748966 s, audit 3.381892 s,
  checkpoint IO 0.072490 s, physical evaluation 91.117554 s.
- Start **2026-09-14T15:57:49.200201+00:00**, report **2026-09-14T16:09:29.079380+00:00**.
  Source `2eceb24`; checkpoint SHA256
  `ca676efb6b3ee4f48798f19dff1c1675b20b2532a4a6e5f8239cf64f6b6d07f6`.
- Video render 71.093333 s, full decode **4.332757 s**;
  43 seconds / 2,150 frames / 50 fps / 1920x1080 / 1x. Inspected and opened
  **2026-09-14 16:15:41 UTC**. Same starts and chart/camera settings in all panels.

[Watch PID / run 05 / run 08](../../../../previews/embodied_fly/velocity_hover_ppo_08_comparison_v1.mp4).

```sh
open previews/embodied_fly/velocity_hover_ppo_08_comparison_v1.mp4
```
