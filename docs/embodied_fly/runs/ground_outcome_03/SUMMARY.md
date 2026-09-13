# Ground outcome 03: wing guidance with physical PPO

Wing posture improves in both ground commands. Standing wing-angle RMS is
0.05704rad versus motor_focus_06's0.113997rad; walking is0.05276rad versus
0.063092rad. Both commands remain upright with valid support. Peak wing errors
still exceed0.2rad; standing sag3.479% is worse than06's1.008%, and heading gates
fail. Hover falls. This is a promising development candidate, not a release.

One shared397-input/78-output MaleCNS actor and the canonical wing_motion body
are unchanged. PPO learns physical movement/support/posture. A training-only
auxiliary adds100 times mean squared normalized-command error over six wing
channels. Its bounded labels depend only on current measured wing angles and
speeds. No walking-leg teacher targets, legacy corpus, utility loss, output
mask or runtime joint correction are used. Every executed action comes from
the actor's tanh-normal distribution; all78 outputs remain active.

| Actual measurement | Result |
| --- | --- |
| Training wall time | 182.342745 s |
| Setup | 6.754555 s |
| Collection and inference | 112.291787 s |
| PPO plus auxiliary optimization | 70.038626 s |
| Physical transitions | 131,072 |
| Aggregate simulated time | 262.144 s |
| Parallel worlds | 32:16standing /16walking |
| Physics / control | CPU MuJoCo/mjbatch5kHz / actor500Hz |
| Graph and learning | RTX4090 CUDA |
| Rollouts / PPO updates | 32 / 283 |
| Wing-label presentations | 144,896, including replayed examples |
| Completed episodes / failures | 129 / 6 |
| Peak CUDA allocation | 9,644,402,176 bytes |

The first gradient audit separates physical-return gradients from the weighted
wing auxiliary. Core-bias norms are79.4092 and2.64913 respectively; both reach
the same intrinsic neuron dynamics. This is PPO with corrective supervision,
not pure RL. Fixed connections and inactive utility/intention weights remain
unchanged. Failed-trace labels are verified against pre-action measurements;
executed controls differ from those labels and preserve causal action feedback.

Five-second independent development evaluation uses seed72001, allthree cases,
zero teacher or resets, and no numerical warnings. The current candidate remains
preserved while an identical-recipe continuation is evaluated.

| Command | Stable | Speed RMSE cm/s | Yaw RMSE rad/s | Wing RMS rad | Wing max rad | Full task |
| --- | --- | --- | --- | --- | --- | --- |
| stand | yes | 0.258 | 1.285 | 0.057 | 0.242 | fail |
| walk | yes | 0.489 | 3.694 | 0.053 | 0.242 | fail |
| hover | no | 0.320 | 2.351 | 1.253 | 2.865 | fail |

Recorded2–5s standing wing maximum is0.1936rad; the complete five-second maximum
is0.2420rad. The whole trajectory remains part of acceptance. No startup frames
are discarded to claim a pass. Full15s review retains standing, walking and
failed hover at1x with eye cameras and the measured neural-latent projection.
