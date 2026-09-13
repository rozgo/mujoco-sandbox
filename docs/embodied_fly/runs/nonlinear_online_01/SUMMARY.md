# Online nonlinear wing readout: ground support retained, hover incomplete

The same nonlinear wing decoder now learns from physical states caused by its own actions. Reference controllers provide training labels only. Ground labels restore resting wing pose and speed; the measured-state hover reference supplies airborne corrections. This is online corrective imitation, not PPO. One checkpoint commands all 78 actuators through MaleCNS.

Only the 105,222 decoder parameters change. All original actor parameters, feature normalization, graph wiring, body mechanics and flight-force parameters remain frozen. Total actor size is 2,516,146 parameters, including the original 2,410,924.

| Measurement | Result |
| --- | --- |
| Training | 180.376794 s |
| Setup | 10.172478 s |
| Collection and forward computation | 174.544957 s |
| Backward computation and optimization | 5.818723 s |
| Parallel worlds | 32: 11 stand, 11 walk, 10 hover |
| Physical transitions | 174,080 |
| Aggregate simulated experience | 348.16 s |
| Optimizer updates | 170 |
| Peak allocated CUDA memory | 896,806,912 bytes |

Physics uses native CPU MuJoCo/mjbatch on 16 threads at 5 kHz. The RTX 4090 runs neural inference and learning; the actor acts at 500 Hz. Every executed training action comes from the actor, with zero teacher mixing.

Training preserves 569 completed episodes and 493 failure traces: 15 standing failures, 40 walking failures and 438 hover failures. Successful training timeouts do not establish independent acceptance. All 493 failure traces were verified against their hashes and recorded actor actions.

The unassisted evaluation uses seed 72011 and five seconds per command. Setup takes 6.297731 s; stepping and capture take 31.530508 s. It adds 7,500 physical transitions and 15 aggregate simulated seconds.

| Command | Physical outcome | Wing posture |
| --- | --- | --- |
| Stand | Upright, valid ground support | RMS 0.20895 rad; peak 0.62875 rad; fails |
| Walk | Upright, valid ground support | RMS 0.11189 rad; peak 0.55324 rad; fails |
| Hover | Falls; root tracking RMSE 19.756 mm | Not a ground posture task |

All strict task gates fail, with zero numerical warnings. **Retention02 remains selected; this candidate is not promoted.** Evaluation seeds differ from the preceding pilot, so these are individual outcomes, not a matched performance estimate.

Independent checks prove every original state scalar and normalization buffer unchanged, matching graph/body/model hashes, and finite causal bounded captures. Non-wing outputs match a frozen copy on 174,080 identical histories within 9.54e-7. The first evaluation launch had a mistyped graph path and failed before a rollout; its evidence is preserved separately. The corrected launch uses the declared seed and unchanged checkpoint.

The [complete video](../../../../previews/embodied_fly/nonlinear_online_01_all_tasks_v1.mp4) includes every command and failure at 1×: 15 seconds, 750 frames, 1600×900, 50 fps. Rendering and encoding took 60.252149 s. The video was fully decoded, visually inspected and opened automatically.

The decoder has not learned reliable resting-wing posture or hover. Further motor work must address feedback and physical recovery. Learned utility and the multi-agent survival goal remain incomplete. No physics mechanism or acceptance threshold was relaxed.
