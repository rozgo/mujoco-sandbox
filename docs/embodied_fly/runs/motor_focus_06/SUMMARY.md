# Ground posture continuation (06)

**Progress, not accepted.** Both ground cases remain upright with permitted support for five seconds; strict posture/tracking gates and hover remain failed.

One command-conditioned MaleCNS actor, same canonical body and all 78 learned outputs. Zero reference actuation during collection: the student executes every action; references only label corrective targets. This is online imitation, not PPO.

- Training 120.113071 s; setup 8.004375 s.
- Collection/forward 93.947955 s; backward 26.155438 s.
- 107, 520 transitions/105 updates/215.040 aggregate simulated seconds.
- 32 worlds(11 stand/11 walk/10 hover), 16 native CPU physics threads; RTX 4090 graph/learning.
- PeakCUDA 9, 411, 467, 264 bytes; 5 kHzphysics/500 Hzcontrol.

| Case | Stable 5 s | Height loss | Wing max deviation | Wing speed RMS |
|---|---:|---:|---:|---:|
| stand | True | 1.01% | 15.48 deg | 0.481 rad/s |
| walk | True | 0.09% | 15.45 deg | 1.651 rad/s |
| hover | False | 66.27% | 172.04 deg | 7.018 rad/s |

Every state/model/checkpoint hash and causal previous-action observation was verified. All failed training traces remain preserved. Complete review includes hover failure:

[Watch all cases](../../../../previews/embodied_fly/motor_focus_06_all_tasks_v1.mp4).

15 s, 750 frames, 1600×900, 50 fps, 1×; full decode and visualQA, automatically opened. Render/encode 60.054140 s.
