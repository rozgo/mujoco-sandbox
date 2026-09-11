# Adaptive dog learner backend comparison

Started **2026-09-11 17:02:51 UTC** after the user asked whether the adaptive dog
was trained on CPU or GPU and requested a GPU comparison. The accepted v1 used
**CPU MuJoCo/mjbatch physics, CPU actor inference and MPS learner updates** on an
Apple M3 Max. Earlier pilot CUDA work used different bodies/curricula and was not
a matched comparison of this limb-loss policy.

## Plan frozen before the new runs

Branch `experiment/adaptive-dog-gpu-comparison`; official v1 remains frozen on
main. Run **three 90-second continuations**, using the final v1 refinement recipe:

| Label | Machine | Physics | Actor | Learner |
| --- | --- | --- | --- | --- |
| mac_mps | Apple M3 Max, 16 CPU cores | CPU | CPU | Apple GPU / MPS |
| desktop_cpu | Ryzen 9 5950X, 16 cores / 32 threads | CPU | CPU | CPU |
| desktop_cuda | Same Ryzen desktop + RTX 4090 | CPU | RTX 4090 | RTX 4090 |

Use the same archived `limb_rear_overlap_iter100_seed2.pt` parent (SHA-256
`aefb52f219090ccdfe48cbe9b217106520669f7999cc25a6a251dffc4a346825`), healthy teacher,
training seed **2**, **512 environments**, **16 physics threads**, PPO settings,
rewards, body mixture and 2 ms physics / 20 ms control. Freeze the original
healthy-motion reference (SHA-256
`c8cc59cc4b61e6268aa8da94aa9c5f05b684cd6c53d19a0d1d27691b64b06733`) so different
backends do not regenerate different training targets. Default training behavior
is unchanged unless the new optional frozen-reference path is supplied.

Check forward actions, critic values and gradients on a single frozen bank of
1,080 observations before training on each device. Limits: action error <1e-4,
value error <1e-3, gradient error/global maximum CPU gradient <1e-4.

Use each run's **final checkpoint**, without selecting prettier intermediates.
Compare completed transitions/second, rollout/update times and identical
development evaluations: **8 trials per body**, seed **9217**, 0.5 ms physics,
all nine bodies, including support, visible feet and healthy gait. Demonstration
seed **9143**, cases healthy/lower-FR/whole-FR, matched viewpoints and 1× playback.
No new policy will be promoted through this benchmark.

The desktop CPU and CUDA runs are sequential. The Mac can run concurrently on
its independent hardware. This is one seed per configuration and an equal-time
engineering pilot, not a multi-seed learning-quality or pure GPU-speed claim.
The desktop already has another GPU workload (about 11.2 GiB allocated and 21%
utilization at the first probe); leave it running and record GPU/CPU load during
both desktop runs. Load variation and backend-specific numerical/RNG differences
limit causal interpretation.

This compares the existing **neural-network backends**. Physics remains CPU
MuJoCo/mjbatch on every configuration. It does not benchmark GPU-resident physics
or MuJoCo Warp. All attempts, failures, source commits and checkpoint identities
will be retained.
