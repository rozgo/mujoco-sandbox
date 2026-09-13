# Bounded PPO follow-up and camera review — September 13, 2026

The parent position_sustain_retention_01 remains the development baseline.
Outcome04 is the first all-command position PPO candidate to retain standing,
walking and five-second airborne control in the original matched review.
Its hover root error decreases from 6.004 to 5.499 mm (8.4% in one case).
It still fails the unchanged accuracy gate, remains vertically oscillatory,
and fails all three additional deterministic starts. It is not promoted.

## What we learned

The frozen parent fails three other hover starts even without exploration.
On the original successful initial state it survives zero and .003 noise but
falls after 2.768 seconds with the previous PPO .01 noise. Smaller exploration
alone does not produce a usable PPO refinement. Adding eight critic-only
rollouts before actor updates preserves the original case, but does not repair
the broader starting-state fragility. This is evidence about these pilots, not
a general claim about PPO. The physical body, rewards, actor architecture,
measured graph and command set remain fixed throughout.

The two new pilots use 32 worlds (11 stand/11 walk/10 hover), 16 CPU MuJoCo
physics threads and RTX 4090 neural work, at 5 kHz physics / 500 Hz actions.
Both resume the original parent independently. Warmup counts inside training;
no teacher controls collected or evaluated actions. Utility remains disabled.

## Measured runs

Outcome01/02 preceded this new round and are now fully archived. Outcome03/04
are the two new bounded pilots. All four preserve ground stability; walking's
strict yaw gate remains failed.

| Run | Change | Training seconds | Transitions | Actor updates | Hover in original review |
| --- | --- | ---: | ---: | ---: | --- |
| position_outcome_01 | Initial all-command PPO | 180.961 | 151,552 | 37 | Falls |
| position_outcome_02 | Actor LR reduced to 3e-7 | 183.389 | 94,208 | 184 | Falls |
| position_outcome_03 | Initial/minimum noise .003 | 184.976 | 126,976 | 110 | Falls |
| position_outcome_04 | Eight critic-only rollouts first | 182.497 | 131,072 | 70 | Airborne, accuracy gate fails |

The two new runs use **367.474 seconds** of training and **258,048**
transitions (516.096 aggregate simulated seconds). Outcome04's 48.195-second
warmup and 64 initial critic updates are included. Its total is 70 actor and
134 critic updates. Both runs peak at 18,218,412,544 bytes of CUDA allocations.
Evaluation, video work and implementation are separate from these totals.

## What the camera revealed

The overview now has a wider 1.30 cm distance and damped target tracking
(0.12 seconds horizontally, 0.40 seconds vertically, with a bounded lag).
The successful original capture is re-rendered with identical state/model
hashes. No simulation pose, force, action or timing is changed.

The original body already oscillates around 5 Hz: height 0.902–2.106 cm,
vertical speed RMS 8.27 cm/s. Outcome04 spans 0.889–2.245 cm, with vertical
speed RMS 8.20 cm/s. It is sustained airborne control, not steady hovering.
The smoother camera makes this existing physical bobbing easier to see.

- [Original checkpoint with damped camera](../../previews/embodied_fly/position_sustain_retention_01_all_tasks_v2_damped.mp4)
- [First new PPO pilot, including its fall](../../previews/embodied_fly/position_outcome_03_all_tasks_v1.mp4)
- [Critic-warmup PPO pilot](../../previews/embodied_fly/position_outcome_04_all_tasks_v1.mp4)
- [Matched exploration diagnosis](runs/position_exploration_01/SUMMARY.md)
- [Additional-start results after PPO](runs/position_outcome_04_exploration/SUMMARY.md)

## Validation and next boundary

132 tests pass (33 existing dependency warnings); focused PPO/exploration tests
and changed-source lint checks pass. All four pilots' 386 failure traces,
12 full command captures, physical/graph identities, frozen normalization and
utility parameters, bounded actions, causal feedback and reward decomposition
are verified. All completed training videos are decoded end-to-end, visually
inspected and opened on the Mac. The exact source capture is verified for the
camera-only rerender. Full diagnostic trajectories stay in ignored outputs;
checksummed reports and authoritative checkpoints/videos are archived in Git/LFS.

No third PPO run is launched in this round. Keep both checkpoints available,
with the original as baseline. The next motor work should target consistent,
quieter hover across initial states explicitly. Command transitions, takeoff,
landing and learned needs/utility integration remain unproved and deferred.
