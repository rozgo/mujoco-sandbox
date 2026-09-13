# Ground posture continuation (05)

**Progress, not accepted.** Both ground cases remain upright with permitted support for five seconds; strict posture/tracking gates and hover remain failed.

One command-conditioned MaleCNS actor, same canonical body and all 78 learned outputs. Zero reference actuation during collection: the student executes every action; references only label corrective targets. This is online imitation, not PPO.

- Training 181.025272 s; setup 7.638945 s.
- Collection/forward 143.035657 s; backward 37.972705 s.
- 156, 672 transitions/153 updates/313.344 aggregate simulated seconds.
- 32 worlds(11 stand/11 walk/10 hover), 16 native CPU physics threads; RTX 4090 graph/learning.
- PeakCUDA 9, 411, 467, 264 bytes; 5 kHzphysics/500 Hzcontrol.

| Case | Stable 5 s | Height loss | Wing max deviation | Wing speed RMS |
|---|---:|---:|---:|---:|
| stand | True | 3.48% | 17.07 deg | 0.336 rad/s |
| walk | True | 0.93% | 14.34 deg | 1.686 rad/s |
| hover | False | 66.54% | 131.13 deg | 7.525 rad/s |

Every state/model/checkpoint hash and causal previous-action observation was verified. All failed training traces remain preserved. Complete review includes hover failure:

[Watch all cases](../../../../previews/embodied_fly/motor_focus_05_all_tasks_v1.mp4).

15 s, 750 frames, 1600×900, 50 fps, 1×; full decode and visualQA, automatically opened. Render/encode 61.335856 s.

The first render command ran before artifact transfer had finished and produced no video. After transfer completed, the same retained capture was rendered successfully; no simulation or training was repeated.
