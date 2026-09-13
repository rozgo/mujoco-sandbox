# Reduced exploration: ground retained, hover fails

The actor starts from position_sustain_retention_01. Initial and minimum PPO exploration decrease from .01 to .003 following the frozen-actor diagnostic. All other training settings match outcome02; critic warmup is off. The unassisted hover still falls, so this candidate is not promoted.

Measured training: 184.976469 s, 126,976 transitions, 110 actor updates and 31 rollouts. 32 worlds, RTX 4090 neural work, native MuJoCo CPU physics. All 55 training failure traces and all command captures are verified.

| Task | Stable | Full gate | Root RMSE |
| --- | --- | --- | --- |
| stand | True | True | 0.150 mm |
| walk | True | False | 1.297 mm |
| hover | False | False | 21.758 mm |

Physical graph/actor identity, frozen utility and normalization, bounded actions, causal previous-action feedback, selected per-task rewards and one terminal penalty per failed episode are independently checked. The video includes all three cases and uses the requested damped observer camera.
