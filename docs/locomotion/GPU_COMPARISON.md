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

## Results

All three starting parameter states, frozen motion references and source commits
match. All forward/gradient backend checks pass. Training ran from clean commit
`94b0dc0`; the final checkpoints were transferred through commits and Git LFS.

| Configuration | Measured training time | PPO updates | Environment transitions | Transitions/s | Complete tasks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mac M3 Max / MPS | 89.40 s | 149 | 1,830,912 | **20,480** | **72/72** |
| Ryzen 5950X / CPU | 89.41 s | 69 | 847,872 | **9,483** | **68/72** |
| Same desktop / RTX 4090 CUDA | 89.97 s | 87 | 1,069,056 | **11,882** | **56/72** |

CUDA delivers **25.3% more transitions per second than CPU-only on the same
desktop**. The existing Mac/MPS setup is **72.4% faster than this desktop CUDA
configuration** overall. These are the measured rates for this experiment under
the recorded load, not general hardware rankings.

The host-clock median rollout/update stages were **0.549 / 0.035 s** on Mac,
**0.963 / 0.302 s** for desktop CPU, and **0.955 / 0.046 s** for desktop CUDA.
The rollout loop includes CPU physics, observations, reference matching, rewards
and actor transfers. It dominates the CUDA run, explaining why accelerating
neural updates yields a modest total gain. These stage readings use existing
host timings and backend synchronization behavior; they are not isolated CUDA
kernel timings. Whole-run throughput is the primary timing result.

The desktop CPU policy misses four whole-FR goals. CUDA misses all eight lower-FR
and all eight whole-FR goals and fails several clearance/stride/speed checks.
All three retain the healthy walking gates; the MPS run retains every compared
v1 gait gate. **This does not establish that CUDA learns worse**: the runs have
different update counts and backend-dependent arithmetic/minibatch RNG paths,
with only one seed and no repeated learning trials. It establishes that this
90-second CUDA continuation did not reproduce v1's gait quality. No extra tuning
or checkpoint selection was performed after observing these results.

The RTX 4090 used PyTorch **2.14.0+cu130**, CUDA **13.0**, driver **580.173.02**;
both machines used MuJoCo **3.13.0**, Python **3.14.7** and the pinned uv lockfile.
Mean total GPU utilization was about **20.0%** during the desktop CPU run and
**20.4%** during CUDA. These include the preexisting workload and cannot be
attributed solely to the dog. The other workload was not stopped.

**85 tests passed**, including the new frozen-reference equivalence check.
No official v1 weights, runtime settings, media or main-branch files were changed.
The three runs used **268.776 seconds (4 min 29 s)** total new training and
**3,747,840 transitions**; they are short continuations, not training from scratch.

## Inspect or reproduce

[Watch the synchronized training-backend comparison](../../previews/locomotion/learner_backend_comparison.mp4).
Rows show healthy, lower-FR and whole-FR; columns show MPS, desktop CPU and CUDA.
Each column has its own final trained weights, shared across its rows. All panels
use the same CPU inference/physics setup, initial conditions and timestamps at
1×. The 12-second trial outcomes, including failures, remain visible. Cameras
are observer output. Full nine-body metrics are separate from these three
predetermined illustration cases. A complete task requires reaching 5 m and
remaining in the goal lane for one second, without falling or using disallowed
support; merely crossing 5 m near the end does not pass.

All 300 encoded frames decode; opening, middle and ending frames were inspected.
The native Mac viewer passes a five-second launch check, and recorded torques
stay within the original limits. The unchanged 8 mm sampled penetration target
is exceeded by lower-FR in MPS (8.889 mm) and CUDA (8.217 mm). These failed
checks remain recorded in [video QA](gpu_comparison/video_QA.json); no contact
parameters or thresholds were changed to pass the comparison.

From `experiments/adaptive_locomotion`, on the comparison branch:

```sh
# Use a new label/output directory for a repeat; preserve these attempts.
uv tool run --from uv==0.12.12 uv run --locked python scripts/benchmark_learner.py \
  --label my_cuda_repeat --device cuda

# View the actual CUDA-trained checkpoint on Mac or Linux.
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/learner_desktop_cuda_90s_seed2.pt \
  --case whole_fr --presentation damage --timestep 0.0005
```

The optional `--healthy-motion-path` loads an identical validated reference bank;
without it, the trainer still collects its reference as before. The benchmark
uses the current standard CPU actor for CPU/MPS and CUDA actor for CUDA, so it
tests the available execution paths rather than a new GPU-optimized trainer.

[Summary and checkpoint identities](gpu_comparison/summary.json) ·
[Per-policy gait checks](gpu_comparison/evaluation_summary.json) ·
[Original official v1](OFFICIAL_V1.md)

The practical next step for a larger acceleration would be to profile and move
more of the rollout work onto the GPU. This experiment does not justify
replacing the official policy, or claim that simply selecting CUDA makes our
current CPU physics faster.
