# Guided recovery retains ground switching but not unassisted hover

This continuation from position_commands01 keeps start/stop/resume behavior,
but it does not recover hover. It is not promoted. The first command-training
candidate remains the more useful ground-transition result; the earlier
position_sustain_retention01 remains the combined development reference. Each
video uses exactly one checkpoint for every case; no skills are stitched across
different actors.

The pilot reduces Adam learning rate to 1e-5 and uses an 80% state-reference /
20% student action blend only in training hover worlds. All ground actions remain
student actions. The same body, flight forces, graph, actor, varied starts,
one-second ground switching and supervision weights are retained. No teacher,
action blend, pose override, external oscillator or mask is used in evaluation.

| Measurement | Result |
| --- | ---: |
| Setup | 11.886595 s |
| Training | 180.597429 s |
| Collection/forward | 143.178561 s |
| Backward/optimization | 37.405665 s |
| Physical transitions | 146,432 |
| Aggregate simulated experience | 292.864 s |
| Optimizer updates | 143 |
| Within-episode command changes | 96 |
| Peak CUDA allocation | 9,451,158,528 bytes |

32 worlds: 12 changing ground commands, five fixed stand, five fixed walk and
ten hover; 16 native CPU MuJoCo/mjbatch threads and RTX 4090 neural work. Training
has 32 completed five-second episodes and zero failures. That includes assisted
hover and does not prove learned flight. No failed trace windows are generated
in this run; its declared mixture is recorded in source, configuration and episode
logs. Evaluation trajectories are fully captured and independently verified.

Without assistance, standing passes. Fresh-start walking stays upright but its
path/yaw tracking fails (23.681 mm root RMSE; 2.878 rad/s yaw RMS). Hover crosses
the 0.5 cm height floor at 0.288 s and fails (25.205 mm root RMSE).

The separate eight-second continuous check preserves every physical state and
neural memory across stand/walk/stop/resume. All worlds remain stable and stand/
stop pass. Moving forward speed is 0.922-0.944 cm/s, but mean yaw 0.529-0.600
rad/s still exceeds the 0.35 gate. There are no numerical warnings in either
evaluation. This retains useful switching while leaving the combined motor goal
unmet. Assisted training success did not transfer to autonomous hover here.

[All-command motor review](../../../../previews/embodied_fly/position_commands_recovery_01_all_tasks_v1.mp4).
[Continuous command review](../../../../previews/embodied_fly/position_commands_recovery_01_continuous_v1.mp4).
