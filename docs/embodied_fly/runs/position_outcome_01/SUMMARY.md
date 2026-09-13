# PPO preserves ground control but loses hover

Preserved failed pilot; position_sustain_retention01 remains preferred. The final
mean policy passes standing and keeps walking, but hover falls and tips over.
No teacher or exploration is present in the full five-second review.

Source452f97e; parent position_sustain_retention01; training seed98003 and review
seed97013. The --motor-all implementation selects one physical reward per world,
and adds ground-only MSE against the frozen parent. All sampled actions execute.
The actor architecture, measured graph and wing_position physical body remain unchanged.
No hover-action teacher labels or runtime control substitutions are used.

| Measurement | Actual result |
| --- | ---: |
| Setup | 8.924503 s |
| Training | 180.961423 s |
| Collection / inference | 153.113881 s |
| Optimization | 27.829988 s |
| Physical world/action transitions | 151,552 |
| Aggregate simulated experience | 303.104 s |
| Rollouts / PPO updates | 37 / 37 |
| Peak CUDA allocation | 18,218,412,544 bytes |

32 worlds:11stand/11walk/10hover;16 native CPU physics threads;RTX4090 neural
work. Physics5kHz, actions500Hz,128-step rollouts,32-step recurrent chunks,two
planned epochs. Actor learning rate3e-6,initial sigma0.01. Every rollout stops
after one update because subsequent approximate KL exceeds0.03. The early maximum
is1.669. This stopping rule is not an update rollback or a guaranteed KL bound.

Physical-return gradients reach modeled neuron dynamics before any retention
gradient is added: excitability L2=10.8191, leak=2.89336, bias=410.487; all finite,
with more than165,000 nonzero cell gradients in each group. Measured edges and
utility/normalization stay fixed. The separate1711→128→128→1 critic has235,777
parameters and is training-only. Parent supervision gradients are near zero in
the first audit because the student initially equals that parent.

All232 training failures are hover;11stand and11walk episodes time out. Every
failure trace verifies causal controls, selected reward rates times0.002s and
exactly one terminal penalty. The task-specific reward terms are not double-counted.

| Command | Stable | Full gate | Root RMSE |
| --- | --- | --- | --- |
| Stand | yes | PASS | 0.154 mm |
| Walk | yes | FAIL: yaw | 1.354 mm |
| Hover | no | FAIL: falls | 17.430 mm |

Evaluation setup7.352583s,capture31.486782s,7,500transitions/15aggregate seconds,
zero numerical warnings. All three complete captures and physical/parameter
identities are verified. Every gate is retained. Next: repeat from the airborne
parent with ten-times smaller actor updates, keeping the other settings fixed.

[Complete failed review](../../../../previews/embodied_fly/position_outcome_01_all_tasks_v1.mp4).
All750frames decoded; all cases visually inspected and the video opened.15s,
1600x900,50fps,1x;render/encode60.692612s. The implementation passes128tests
in91.79s,with33known dependency warnings; focused reward/PPO checks pass12tests.
