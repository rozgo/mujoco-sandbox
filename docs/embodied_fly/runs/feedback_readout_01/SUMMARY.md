# Staged readout refinement: ground stability, no accepted hover

This continuation freezes the sensor encoding learned in wing_feedback01 and updates only its 105,222 nonlinear wing-readout parameters. All other parent state remains bitwise unchanged, including sensor weights, cell parameters, base motor decoder and normalization. The same single actor controls all 78 actuators through fixed MaleCNS wiring on the same body.

It uses online corrective imitation with zero teacher mixing. References supply resting-wing and measured-state hover targets; every executed action comes from the actor. This is not PPO or a runtime reference controller.

| Measurement | Result |
| --- | --- |
| Setup | 10.294863 s |
| Training | 180.911450 s |
| Collection/forward | 174.575517 s |
| Backward/optimization | 6.324239 s |
| Worlds | 32: 11 stand, 11 walk, 10 hover |
| Physical transitions | 178,176 |
| Aggregate simulated experience | 356.352 s |
| Optimizer updates | 174 |
| Peak allocated CUDA memory | 897,061,376 bytes |

Physics uses native CPU MuJoCo/mjbatch at 5 kHz on 16 threads; the RTX 4090 runs neural inference and learning. Actions arrive at 500 Hz. Non-wing outputs match the frozen parent within 8.05e-7 on 178,176 identical histories.

Training preserves 224 completed episodes and 109 failure traces. Fourteen hover episodes reach the two-second training timeout, which only means they stayed within the loose height/upright envelope. It does not establish hover tracking or independent success.

The full unassisted seed72015 evaluation runs five seconds per command. Setup takes 5.984868 s and stepping/capture takes 32.375438 s, adding 7,500 transitions and 15 aggregate simulated seconds.

| Command | Physical outcome | Resting-wing result |
| --- | --- | --- |
| Stand | Upright, valid support | RMS 0.11793 rad; peak 0.39100 rad; fails |
| Walk | Upright, valid support | RMS 0.05758 rad; peak 0.26493 rad; fails |
| Hover | Falls; root tracking RMSE 17.030 mm | Not a ground posture task |

All strict task gates fail with zero numerical warnings. Evaluation seeds differ from prior pilots; these outcomes do not establish a matched improvement estimate. **Retention02 remains the preserved development checkpoint.**

The [full video](../../../../previews/embodied_fly/feedback_readout_01_all_tasks_v1.mp4) retains every command and failure at 1×: 15 s, 750 frames, 1600×900, 50 fps. Rendering/encoding took 59.647187 s. It was fully decoded, visually inspected and opened automatically.

All 109 training failure traces and complete evaluation captures pass independent hash, finite-state, bounded-action and causal-feedback checks. The selected parameter boundary and unchanged body/graph hashes are verified. The staged implementation works, but resting-wing control and hover remain unresolved. Further work needs a materially different motor-learning approach before claiming the full motor, utility or multi-fly survival goal is achieved.
