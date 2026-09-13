# Stronger ground-wing loss (07)

**Not selected.** Raising ground-wing imitation weight 10→100 does not improve
rest-angle accuracy. Keep 06 as the preferred development checkpoint.

- Same body, one actor, 32 worlds(11/11/10), zero executed reference assistance.
- 120.509577 s training; 8.126440 s setup.
- 105 updates/107, 520 transitions/215.040 aggregate simulated seconds.
- Collection/forward 94.308113 s; backward 26.191177 s.
- RTX 4090 graph/learning, native CPU MuJoCo physics 16 threads.
- Both ground cases remain upright for 5 s with zero prohibited support.
- Stand/walk wing RMS 0.1443/0.0830 rad, versus 06's 0.1140/0.0631 rad.
- Standing sag improves 1.01%→0.751%, but the requested wing result worsens.
- All posture/tracking gates remain failed; hover falls.

Every capture/model/checkpoint hash and causal action feedback was checked,
including 179 saved failure traces. Full 07 video is preserved with all 3 cases.
The learning target is now correct, but increasing its scalar weight is not
sufficient to learn accurate ground-wing feedback in this small run.

[Complete review](../../../../previews/embodied_fly/motor_focus_07_all_tasks_v1.mp4): 15 seconds, 750 frames, 1600×900, 50 fps, 1× playback. Full decode and visual QA passed; automatically opened. Render/encode: 59.753756 seconds.
