# Hover imitation with frozen ground targets: pilot does not pass

The three-minute pilot uses the declared recipe without physics or actor
architecture changes. The checkpoint remains diagnostic; ground_outcome_03 is
still the development parent. Training-only references are absent in evaluation.

| Five-second unassisted case | Outcome | Wing angle RMS / maximum |
| --- | --- | --- |
| Stand | Falls; first recorded envelope failure at 0.542 s | 1.648 / 3.004 rad |
| Walk | Upright, permitted support; wing-posture gate passes, tracking fails | 0.0405 / 0.1654 rad |
| Hover | Falls; position error 18.56 mm | Flight posture, ground gate inapplicable |

Standing and hovering are regressions/failures, not accepted primitives.
Walking wing RMS improves from the parent's 0.0528 to 0.0405 rad and its peak
deviation falls below the 0.2 rad gate. This is one development comparison,
not evidence of broad robustness. Initial qpos, qvel and all 397 observations
match the parent's evaluation exactly in every case. All cases retain the full
five seconds, including failure frames. There are no numerical warnings.

Training: **180.791216 seconds**, 148 updates, 151,552 physical action transitions,
303.104 aggregate simulated seconds. Setup 9.812909 seconds; collection and
forward inference 144.470024 seconds; backward/optimization 36.307506 seconds.
32 worlds (11 stand, 11 walk, 10 hover), 16 native MuJoCo CPU threads; RTX4090
graph inference/learning, peak allocated CUDA 9,441,672,192 bytes. Physics remains
5 kHz, control 500 Hz. Counting frozen-reference inference does not add physical
experiences. This is online imitation, not a PPO run.

Ground collection uses only student actions. Hover collection blends 80%
reference with 20% student, explicitly recorded per task and trace. The frozen
parent reference stays unchanged. Utility/intention weights remain frozen; all
three internal cell-parameter groups receive finite nonzero learning gradients.
All three saved training failure traces are verified against hashes, actual
executed mixtures and causal previous-action feedback. The 131 completed
training episodes contain three walking failures; the assisted training success
counts do not establish frozen-checkpoint success from a fresh neural reset.

Evaluation: 6.385944 seconds setup plus 31.079010 seconds capture. The full
15-second video includes all three commands at 1x, 750 frames, 1600×900, 50 fps.
Render/encode takes 59.781641 seconds. The entire file is decoded; opening,
ground behavior and final failure frames are visually reviewed, then opened.

Checkpoint SHA-256:
`7b4a0267178a7232b75a8e2deb990954f6d1911446a5f0f5675e98d43efb75de`.
Source 4ae7183 is committed and clean. Exact settings, graph/body hashes, losses,
gradient audit, comparisons and failure verification accompany this summary.

Next investigation: check fresh-reset control and recurrent-state behavior
before allocating another mixed-task run. The mismatch between training episode
counts and the final fresh-start standing result needs direct examination.
Lower learning rate or stronger startup retention are hypotheses, not fixes
established by this pilot. Preserve the successful teacher as a training tool;
it is not installed into the deployed actor. Hover, transitions and the eventual
utility/survival curriculum remain unfinished.
