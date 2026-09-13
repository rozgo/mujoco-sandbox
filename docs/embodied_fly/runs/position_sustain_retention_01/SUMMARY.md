# Learned walking retained while hover moves closer to its target

One unassisted checkpoint stands, advances 4.84 cm while walking, and remains
airborne/upright through the complete five-second review. Standing passes.
Walking yaw and hover position still fail their preserved gates. This is the
latest developmental motor candidate, not a completed release or survival brain.

Source 2b24e73; parent position_motion01. The unsuccessful position_sustain01
branch is preserved and is not an ancestor of this checkpoint. The same
wing_position body, measured-wing flight law, 397 live inputs, 78 outputs and
four recurrent graph updates are used for every command. Connectivity,
normalization and utility/intentions remain fixed; all motor encoder/decoder
and modeled neuron dynamics parameters learn.

The previous policy provides ground action targets during training, with its
own frozen recurrent state. The student executes every action in all worlds.
Standing retains complete initial-form supervision, ground wings retain rest
targets, and an additional weight-4 body-command loss discourages forgetting.
Hover learns from the unchanged measured-state reference. These are imitation
losses, not PPO rewards. The training-only reference is absent at evaluation.

| Measurement | Actual result |
| --- | ---: |
| Setup | 11.481816 s |
| Training | 181.159775 s |
| Collection / forward | 142.545569 s |
| Backward / optimization | 38.602594 s |
| World/action transitions | 150,528 |
| Aggregate simulated experience | 301.056 s |
| Optimizer updates | 147 |
| Peak CUDA allocation | 9,451,158,528 bytes |

32 worlds (11 stand, 11 walk, 10 hover), 16 native CPU MuJoCo/mjbatch threads,
RTX 4090 neural work, 5 kHz physics / 500 Hz control. Five-second episodes,
32-action recurrent chunks, fresh Adam 0.0001, seed 97003. Ground/hover task
weights 4:4:1; startup weight 1. 2,336,540 trainable parameters. The 63 failed
training episodes are all hover; every failure trace is verified. Eleven stand,
eleven walk and two hover episodes reach their five-second training timeout.

| Command | Stable | Full gate | Result |
| --- | --- | --- | --- |
| Stand | yes | PASS | initial form retained; root RMSE 0.149 mm |
| Walk | yes | FAIL | 4.84 cm forward; root RMSE 1.51 mm; yaw RMS 2.67 rad/s |
| Hover | yes | FAIL | root RMSE 6.00 mm, above the unchanged 5 mm gate |

The seed97013 evaluation starts hover at 1.8665 cm, drops no lower than 0.9017 cm,
peaks at 2.1061 cm, and ends at 1.7220 cm in the last recorded frame (4.998 s).
Walking has valid support and passes speed and resting-wing gates. Ground wing
maximum deviations are 0.126 degrees standing and 0.145 degrees walking.

The complete review uses no teacher, action blending, pose resets or output masks.
Setup 8.408059 s; capture 31.806296 s; 7,500 transitions / 15 aggregate simulated
seconds; zero numerical warnings. Capture/checkpoint/model hashes, bounded
actions, causal previous-action inputs and frozen parameter boundaries are verified.
One premature local archive check read the model while its transfer was still
running and rejected its hash. After the transfer handle completed, every model
and trajectory hash matched and the archive checks passed. No simulation was rerun.

The retained and non-retained continuations share evaluation seed97013; their
hover root RMSEs are 6.00 and 20.34 mm respectively. They also differ in training
distribution and ground losses, so this is a recipe comparison, not an isolated
effect of one loss or evidence of general robustness. The older parent's review
used another seed and is not presented as a matched comparison.

The hover-error breakdown identifies both early height loss and horizontal drift.
Across recorded pre-action samples, altitude RMS error is 4.68 mm and the final
horizontal displacement is 7.18 mm. This directs the next work toward lift
recovery and reducing velocity bias, while retaining the working ground commands.
Do not declare success by raising the gate or masking outputs. Motor transitions,
takeoff/landing, learned utility and multi-agent survival remain outstanding.

[Complete three-command review](../../../../previews/embodied_fly/position_sustain_retention_01_all_tasks_v1.mp4).
All 750 frames were decoded; opening, middle and ending frames of every command
were inspected and the video was opened. 1600x900, 50 fps, 15 seconds, 1x.
Render/encode 59.426399 s. Earlier videos and candidates remain intact.
