# Direct local feedback imitation does not retain hover

The pilot is preserved as failed; position_sustain_retention01 remains the
development baseline. It retains standing and forward walking in the original
review, but hover crosses the 0.5 cm height floor at 0.58 seconds and eventually
settles on the ground. Full five-second hover root RMSE is 17.70 mm. Walking's
yaw gate remains failed. Do not promote it based on a response loss.

The run adds the declared paired height/vertical-speed response loss to the
existing motor imitation recipe. No PPO, body/force change, raw sensor bypass,
utility learning, motor masking or runtime teacher is added. The fixed graph,
normalization and utility/intentions remain unchanged; actual gradients reach
the modeled cell dynamics. One checkpoint controls all 78 outputs in every case.

| Measurement | Result |
| --- | ---: |
| Setup | 12.371462 s |
| Training | 180.734316 s |
| Collection/forward | 143.127583 s |
| Backward/optimization | 37.592718 s |
| Physical world/action transitions | 144,384 |
| Aggregate simulated experience | 288.768 s |
| Optimizer updates | 141 |
| Synthetic sensor inputs | 2,256, counted separately |
| Peak CUDA allocation | 9,591,054,848 bytes |

32 worlds (11 stand, 11 walk, 10 hover), 16 native CPU MuJoCo/mjbatch threads,
RTX 4090 neural work, 5 kHz physics and 500 Hz actions. Every physical action
comes from the student. The 382 failed training episodes and their complete
saved trace windows are verified; all are retained. The training and evaluation
physical fingerprints and graph identities match their parent.

| Command | Stable | Full gate | Root RMSE |
| --- | --- | --- | ---: |
| Stand | yes | pass | 0.148 mm |
| Walk | yes | fail: yaw | 0.812 mm |
| Hover | no | fail | 17.699 mm |

All three full captures have finite state, bounded actions and causal
previous-action observations. There are no numerical warnings or evaluation
resets. Since the original hover case fails, the optional extra-start evaluation
is not run. The unassisted video retains the failure.

[Complete three-command review](../../../../previews/embodied_fly/position_hover_response_01_all_tasks_v1.mp4).

All 750 encoded frames decode; 13 keyframes include the fall onset and impact.
The video was opened locally after review. Render/encode takes 59.370403 s.
