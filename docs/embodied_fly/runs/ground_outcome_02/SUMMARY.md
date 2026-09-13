# Ground outcome 02: smaller physical-PPO steps

Not selected over motor_focus_06. Both ground commands remain upright on valid
supports, but standing wing position and body sag are worse than06. Walking wing
error is slightly lower. Hover falls and all full task gates remain failed.

This trial restarts from06 with fresh PPO/critic state at actor learning rate
1e-6, ten times smaller than ground_outcome01. Reward, canonical physical body,
actor, all78 action channels, reset disturbances, rollout dimensions and180s
budget are unchanged. Training seed81002; evaluation seed72001. There is no
teacher, rehearsal, utility loss, output mask or runtime correction.

| Actual measurement | Result |
| --- | --- |
| Training wall time | 182.308462 s |
| Setup | 7.155512 s |
| Collection and inference | 111.604189 s |
| PPO optimization | 70.691535 s |
| Physical transitions | 131,072 |
| Aggregate simulated time | 262.144 s |
| Parallel worlds | 32:16standing /16walking |
| Physics / control | CPU MuJoCo/mjbatch5kHz / actor500Hz |
| Brain and learning | RTX4090 CUDA |
| Rollouts / PPO updates | 32 / 288 |
| Completed episodes / failures | 128 / 3 |
| Peak CUDA allocated memory | 9,644,186,624 bytes |

Smaller steps allow288 updates over32 rollouts, compared with47 updates over47
rollouts in01. This fixes the large-step optimization bottleneck without proving
physical improvement. Gradients from return reach all three intrinsic cell
parameter groups; fixed connectome and utility/intention weights are unchanged.

Five-second independent development evaluation runs allthree commands from the
same checkpoint and same body, with no resets or teacher. Zero numerical warnings.

| Command | Stable | Speed RMSE cm/s | Yaw RMSE rad/s | Wing RMS rad | Wing max rad | Full task |
| --- | --- | --- | --- | --- | --- | --- |
| stand | yes | 0.152 | 1.139 | 0.149 | 0.579 | fail |
| walk | yes | 0.504 | 3.507 | 0.060 | 0.253 | fail |
| hover | no | 0.205 | 1.659 | 1.455 | 3.002 | fail |

Recorded-state windows show standing wing error persists after two seconds
(RMS0.147rad, maximum0.362rad), so the remaining problem is not just startup.
Walking last-three-second wing maximum0.206rad is closer to the0.2rad gate,
but the complete trajectory still fails. Do not trim startup to claim success.
The complete15s review retains allthree cases at1x. Adjacent reports preserve
checkpoint/model/capture/failure hashes and verified causal action feedback.
