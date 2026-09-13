# Ground outcome 01: direct motor PPO

Not selected. Standing falls in the independent five-second test; walking
stays upright but misses command and wing-posture targets; hover falls.
Preferred motor_focus_06 remains preserved, with its existing limitations.

This trial starts from06 and trains one motor-only MaleCNS actor directly on
physical outcomes. No teacher, imitation rehearsal, output mask or runtime
joint correction is used. All78 action channels are learned. The fixed
connectome,397inputs,4internal updates and canonical wing_motion body remain
unchanged. Utility/intention weights are frozen and verified unchanged.

A separate training-only critic and tanh-normal exploration implement recurrent
PPO. Rewards are measured every2ms from movement, support, uprightness, wing
pose/speed and body posture. Standing additionally rewards initial leg pose.
Failure penalty is applied once before reset; timeouts use the terminal state
for value bootstrapping. Adjacent report records all weights and scales.

| Actual measurement | Result |
| --- | --- |
| Training wall time | 182.957318 s |
| Setup | 6.593570 s |
| Physical collection and inference | 166.133669 s |
| PPO optimization | 16.804158 s |
| Physical transitions | 192,512 |
| Aggregate simulated time | 385.024 s |
| Parallel worlds | 32:16standing /16walking |
| Physics / control | CPU MuJoCo/mjbatch5kHz / actor500Hz |
| Brain and learning | RTX4090 CUDA |
| Rollouts / PPO updates | 47 / 47 |
| Completed episodes / physical failures | 193 / 7 |
| Peak CUDA allocated memory | 9,644,186,624 bytes |

All47 rollouts permit only one PPO update before KL early stopping. The
1e-5 learning rate moves the narrow78-dimensional action distribution too far.
The declared next trial starts from06 at1e-6, preserving reward and physics.
This is an optimization diagnosis, not evidence that the reward solves the task.

Development evaluation uses seed72001, allthree complete5s cases, zero teacher,
no resets and no numerical warnings. Same checkpoint and body in every case.

| Command | Stable | Speed RMSE cm/s | Yaw RMSE rad/s | Wing RMS rad | Wing max rad | Full task |
| --- | --- | --- | --- | --- | --- | --- |
| stand | no | 0.428 | 2.994 | 0.528 | 2.656 | fail |
| walk | yes | 0.634 | 3.337 | 0.066 | 0.314 | fail |
| hover | no | 0.387 | 1.927 | 1.295 | 3.000 | fail |

The full15s video retains failed cases. Recorded actions, causal feedback and
checkpoint/model/state/failure hashes are verified in adjacent evidence.
