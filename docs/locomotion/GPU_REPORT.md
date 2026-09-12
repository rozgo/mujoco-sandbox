# Adaptive dog: measured RTX 4090 training from scratch

**One RTX 4090, 4,096 parallel worlds throughout, one final policy for walking, standing and moving supports.** Replaying the successful curriculum from random weights took **22 min 05 s of training**, collecting **71.76 million new experiences**. The final checkpoint completes **36/36 walking tasks**, passes **136/144 strict static-balance checks** and **24/24 moving-support checks**. All 204 task trials stay upright.

[Exact timing and provenance](GPU_REPORT.json) · [Recipe and reproduction](gpu_from_scratch/README.md) · [Final evaluation](gpu_from_scratch/evaluation/stage_30/summary.json) · [Earlier continuation experiments](GPU_CONTINUATION_REPORT.md)

## GPU learning time

Each row continues the previous row's weights. Healthy walking and damage adaptation together took **11 min 33 s**; that is the complete nine-body walking curriculum.

| Training phase | Learning time | Parallel worlds | New experiences | Aggregate simulation hours | Experiences/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Healthy walking from random weights | 1m 54s | 4,096 | 9.63 million | 53.52 | 84,658 |
| Adapt walking to missing limbs | 9m 39s | 4,096 | 29.79 million | 165.48 | 51,463 |
| Standing on static supports | 8m 23s | 4,096 | 26.84 million | 149.09 | 53,318 |
| Moving platforms | 2m 09s | 4,096 | 5.51 million | 30.58 | 42,643 |
| **Complete shared policy** | **22m 05s** | **4,096** | **71.76 million** | **398.68** | **54,159** |

Rounded rows need not add exactly; the ledger retains full precision. Learning-loop time is **1,325.012506 s**. It includes physical rollouts, CPU observation/reward work, transfers, PPO optimization, reference/rehearsal losses and ordinary in-loop checkpoint writes. The timer ends after CUDA synchronization. It excludes offline reference collection, setup, standalone evaluation, rendering and earlier abandoned attempts.

**Rehearsal was already part of the archived recipe; it was not added during this replay.** Early standing rounds retain the walking controller through a reference loss. Later standing and moving rounds also use saved walking examples, at the original weight of 15. Reusing those examples is actual training work, so its cost stays in the learning timer. It does not create extra worlds or inflate the new-experience count. The examples and reference policies were rebuilt from this run's own GPU-trained checkpoints.

The 30 processes span **26m 37s** from first launch to final completion. Their recorded setup totals **187.827 s**; remaining process overhead includes interpreter imports, hardware queries, final saves and teardown. Kernel caches were already available. The final combined evaluation takes **50.103 s on the Mac**, outside the GPU learning timer. Earlier walking-retention checks also run separately. These are elapsed measurements on a shared desktop, not isolated kernel timings.

## Scale and mechanics

| Statistic | Setting or measured value |
| --- | --- |
| New simulated experience | **398.68 aggregate hours**, or **16.61 days**, summed across worlds and resets |
| Training throughput | **54,159 new world/action transitions per second**, including learning |
| Aggregate experience rate | **1,083 simulated seconds per wall-clock second** across all worlds |
| Physics | **MuJoCo Warp / CUDA, 500 Hz**, 2 ms timestep |
| Control | **50 Hz**, one action every 20 ms, ten physics steps per action |
| Physics work represented by collected data | **717,619,200 world-physics steps** |
| PPO rollout | **24 steps × 4,096 worlds = 98,304 experiences** |
| Updates | Four epochs, minibatches of **3,072**; **128 optimizer steps per complete rollout** |
| Complete curriculum | **30 stages, 730 rollouts, 93,440 optimizer steps** |
| PPO settings | Clip 0.2, discount 0.99, GAE lambda 0.95; Adam with the archived per-stage learning rates |
| GPU work | Batched contact/rigid-body physics, actor inference and neural-network optimization |
| CPU work | NumPy observations/rewards, host coordination and reference-data collection |

The measured stack is **MuJoCo 3.13.0, MuJoCo Warp 3.13.0, NVIDIA Warp 1.17.0, PyTorch 2.14.0+cu130 and Python 3.14.7**, with the locked uv 0.12.12 workflow.

The actor is **86 → 128 ELU → 128 ELU → 12**, with **29,196 parameters**. The separate critic is **90 → 128 ELU → 128 ELU → 1**, with **28,289 parameters**. Twelve learned exploration scales and normalization buffers are separate. The history estimator is unused. The same feed-forward actor handles every command; there is no runtime checkpoint selection. Support observations are enabled during the standing curriculum while preserving the existing walking mapping. The critic estimates future return; the environment computes the weighted reward every 20 ms. [Implementation](../../experiments/adaptive_locomotion/src/adaptive_locomotion/train.py)

The final stage contains **2,048 moving-platform worlds plus 2,048 static rehearsal worlds**, all improving the same actor. The static half covers 27 body/surface groups. Training also reuses **36,000 walking examples** and **54,000 standing examples** from this lineage. None of these reference actors or datasets is required at deployment.

The nine bodies are the healthy dog, any one complete lower leg removed, and any one entire leg removed. Masses are **15.206 / 14.965 / 13.135 kg**, with **12 / 11 / 9 active actuators** respectively. Hip/thigh torque caps are **23.7 N·m**, calf caps **45.43 N·m**. Removed joints occupy masked slots in the common twelve-action output. Mass/strength randomization is not part of this recipe. [Physical specification](standing/PHYSICAL_SPEC.json)

## One final checkpoint, all commands

Every final result below uses `assets/locomotion/checkpoints/gpu_from_scratch/unified.pt`, SHA-256 `076099ef8fc47bad53cb0cbeb9cc311e64ea47accacfa8ca4080488934b10241`. The healthy, walking and standing milestone files are retained for provenance; deployment uses the final file for all behaviors.

| Final check | Strict/task passes | Upright trials |
| --- | ---: | ---: |
| Walking across all nine bodies | **36/36** | **36/36** |
| Healthy static surfaces and walk–hold–walk | **76/80** | **80/80** |
| Damaged flat holds and walk–hold–walk | **60/64** | **64/64** |
| Healthy moving supports, including stationary controls | **24/24** | **24/24** |

Four predetermined trials per condition use holdout seed 9507. Validation uses **CPU MuJoCo/mjbatch**: 0.5 ms walking physics, 2 ms standing/moving physics, 20 ms control. This verifies native CPU execution of the GPU-trained actor; it is not a new GPU evaluation matrix. All original healthy gait gates pass: mean stride **32.22 cm**, speed **0.558 m/s**, duty gap **11.80 percentage points** and body-height variation **4.59 mm**. Walking also passed 36/36 tasks and the healthy gait gates after both the walking and standing phases, using development seed 9501.

The eight static failures remain failures: all four 24° fore/aft-slope trials have unintended link support; one front-right lower-leg walk–hold–walk trial also has unintended support; three rear-left lower-leg transition trials exceed the 15 cm drift threshold, reaching **16.4–16.8 cm**. None falls. Strict checks also cover speed, tilt, torque, sampled penetration and required foot support. No thresholds or reward weights changed after these results.

These are reset variations on the existing presets, not unseen-terrain or real-hardware validation. Moving training covers healthy bodies; damaged moving transfer is not included in this final audit. Cameras are observer output, not policy inputs. Detailed damaged-foot gait quality beyond the declared retention checks was not re-audited, following the request to avoid unnecessary tests.

## Provenance and earlier experiments

The training checkout stayed at **`a84b3f5`** throughout. The recipe hash is **`94f8f1a2cdf1b2488ce066d833cd93941cb00d0e01afbdec69901d580bcff19e`**. Verification checks all 30 stage records, source identity, 4,096-world settings, exact rollout/update counts, parent continuity and that every trained reference belongs to an earlier stage in this same lineage. The initial checkpoint has no parent, no prior training and zero transitions.

The historical walking curriculum used 512 worlds. Its sample counts were rounded up to complete 4,096 × 24 GPU rollouts: **39,419,904 versus 38,400,000 transitions**, a 2.66% difference. Historical time-limited final minibatches are replaced by complete four-epoch updates. Rewards, stage order, reference roles, seeds and learning rates were preserved. This is a measured curriculum replay, **not an exact-sample CPU/GPU speedup experiment**.

The separate matched walking benchmark still measures **3.23× faster training with Warp physics**. It starts from pretrained weights and must not be presented as learning walking in 42 seconds. The earlier Mac-initialized standing/moving totals and their original results are archived in the [continuation report](GPU_CONTINUATION_REPORT.md) and [benchmark](WARP_TRAINING.md).

Two abandoned starts are kept outside this lineage: an improvised three-round healthy recipe used **328.928 s** of learning; the mistaken 512-world replay used **157.969 s in four completed stages**, plus an interrupted stage whose partial learning time was not fully recorded. They are additional experiment cost, not part of the selected policy's training time. [Retained records](gpu_from_scratch/RESULTS.json)

The existing accepted graphite film and release checkpoints remain unchanged. That film shows the earlier three historical checkpoints, not this new final actor. This replay is saved on its experiment branch for review; it does not silently replace an accepted release.

The separate [complete Ember film](ember/COMPLETE_VIDEO.md) now shows this final
actor across all 51 evaluated conditions, with one predetermined demonstration
trial per condition. All 51 stay upright; two strict misses remain labeled.
Those demonstration trials are separate from the 204-trial audit above.
[Concise summary and shareable cards](GPU_SUMMARY.md).
