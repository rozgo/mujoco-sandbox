# Critic warmup retains airborne control

The second bounded pilot adds eight critic-only rollouts to outcome03's reduced-exploration recipe, starting again from the preserved sustained-hover parent. Warmup uses 48.194531 seconds within the training budget, with 64 critic updates and zero actor updates. The rest of the run applies ordinary PPO plus the existing ground-retention loss.

Training: 182.497163 s; 131,072 transitions; 70 actor updates and 134 critic updates; 32 worlds. Native CPU MuJoCo/mjbatch physics, RTX 4090 neural work.

| Task | Stable | Full gate | Root RMSE |
| --- | --- | --- | --- |
| stand | True | True | 0.128 mm |
| walk | True | False | 1.496 mm |
| hover | True | False | 5.499 mm |

This is the first all-command position PPO candidate to preserve five-second airborne control in the matched review. Hover root RMSE improves from 6.004 to 5.499 mm (about 8.4%) in that one case. It still misses the unchanged 5 mm gate, and walking yaw still fails. It does not establish general robustness.

Vertical bobbing is essentially unchanged: vertical speed RMS 8.20 cm/s versus 8.27 cm/s for the parent, and height spans 0.889–2.245 cm. A lower root-position RMSE does not establish calmer hover. The new damped camera exposes the existing movement more clearly.

All 46 training failure traces and all three unassisted captures are verified. Ground/hover reward selection, single terminal penalties, causal feedback, bounded actions, graph/physical identities and fixed utility/normalizers are independently checked. No teacher, critic or output override controls evaluation. The same predetermined exploration starts are tested next; preserve the parent until these are understood.

The [additional-start check](../position_outcome_04_exploration/SUMMARY.md) fails all nine hover cases, including all three deterministic starts. Ground cases remain stable. The PPO candidate is preserved but not promoted; retain the original baseline.
