# Smaller-step physical PPO: hover still fails

Continues the preserved sustained-hover parent with all-command PPO. The only recipe change from outcome01 is actor learning rate 3e-7 instead of 3e-6. All 23 rollout batches complete eight actor updates each; no KL stop occurs. Hover still falls during the unassisted review. This checkpoint is not promoted.

Training: 183.388904 s, 94,208 transitions, 184 updates, 32 worlds; native MuJoCo CPU physics and RTX 4090 neural work. All 53 training failure traces and all three full evaluation captures pass provenance, causality, bounds and finite-state checks.

| Task | Stable | Full gate | Root RMSE |
| --- | --- | --- | --- |
| stand | True | True | 0.144 mm |
| walk | True | False | 1.541 mm |
| hover | False | False | 15.154 mm |

The complete 15-second, 50 fps, 1x video is decoded, visually inspected and opened. Utility remains disabled; no teacher or critic controls evaluation. Graph wiring and physical identity match the parent.
