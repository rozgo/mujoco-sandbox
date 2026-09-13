# Ground outcome 04: unchanged hybrid continuation

Not selected over ground_outcome_03. Both ground commands stay upright with
permitted support, but peak wing errors and standing sag worsen. Hover falls.
The preceding03 checkpoint is retained as the better wing-posture development
candidate; earlier motor_focus_06 remains the preserved ground comparison.
No checkpoint is declared a completed motor release.

This is an identical-recipe continuation of03: same physical body, shared actor,
reward, corrective wing loss100, learning rate1e-6,32worlds, rollout dimensions,
noise and reset ranges. PPO/critic optimizers and exploration state are verified
resumed. Seed81004; no teacher commands are executed. Utility/intention weights
remain unchanged. Every action channel is still learned.

| Actual measurement | Result |
| --- | --- |
| Training wall time | 185.485408 s |
| Setup | 6.407774 s |
| Collection and inference | 111.287628 s |
| PPO plus auxiliary optimization | 74.183626 s |
| Physical transitions | 131,072 |
| Aggregate simulated time | 262.144 s |
| Worlds | 32:16standing /16walking |
| Physics / control | CPU MuJoCo/mjbatch5kHz / actor500Hz |
| Graph and learning | RTX4090 CUDA |
| Rollouts / PPO updates | 32 / 298 |
| Wing-label presentations | 152,576, including replayed examples |
| Completed episodes / failures | 128 / 4 |
| Peak CUDA allocation | 9,646,195,200 bytes |

Independent evaluation uses three complete5s cases atseed72001, one checkpoint,
no teacher/reset, and no numerical warnings. Source/body/checkpoint/capture
hashes and every failure trace are retained and verified. Wing labels match
pre-action measurements; they are not substituted into executed controls.

| Command | Stable | Speed RMSE cm/s | Yaw RMSE rad/s | Wing RMS rad | Wing max rad | Full task |
| --- | --- | --- | --- | --- | --- | --- |
| stand | yes | 0.234 | 1.308 | 0.061 | 0.319 | fail |
| walk | yes | 0.496 | 3.580 | 0.053 | 0.259 | fail |
| hover | no | 0.262 | 1.636 | 1.153 | 3.006 | fail |

The full15s review includes failed hover at1x. Do not extend this unchanged recipe
again based only on later-window improvement. Hover remains a larger missing
capability; inspect its training target before the next motor curriculum.
