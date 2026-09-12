# Adaptive dog: RTX 4090 training report

**One RTX 4090, 4,096 parallel worlds, approximately 49,000 world/action transitions per second including learning.** Extending the existing walker to stand on uneven terrain and balance on moving platforms took **10 min 55 s of GPU training**, collecting **32.34 million transitions**, equivalent to **179.68 hours of aggregate simulated experience**. A separate matched walking experiment measured **3.23× faster training with MuJoCo Warp than CPU physics**.

These are measured continuation runs. The starting walker was trained earlier on the Mac; its training is excluded from every GPU total below. We have not measured the complete walking curriculum from random weights on the RTX 4090. The walking row sums three independent benchmark runs from the same pretrained checkpoint; these are separate experiments and are not included in the standing/moving total.

[Watch the complete normal-speed film](../../previews/locomotion/graphite/adaptive_dog_complete_v2.mp4) · [Exact statistics, formulas and source hashes](GPU_REPORT.json)

**GPU training by stage**

| Stage | New GPU training | Parallel worlds | Experiences | Aggregate simulation hours | Experiences/s, including learning |
| --- | ---: | ---: | ---: | ---: | ---: |
| Walking fine-tuning benchmark, three runs | 42 seconds | 4,096 | 1.77 million | 9.83 | 41,811 |
| Standing on static supports, seven sequential rounds | 8m 57s | 4,096 | 26.84 million | 149.09 | 50,000 |
| Moving platforms, two sequential rounds | 1m 59s | 4,096 | 5.51 million | 30.58 | 46,435 |
| **Standing + moving total** | **10m 55s** | **4,096** | **32.34 million** | **179.68** | **49,355** |

Times include physics rollouts, host observation/reward work, transfers and neural-network optimization inside the learning loop. Setup, rehearsal-data collection outside the loop, evaluation, rendering and development time are excluded. The nine extension runs separately record **92.593 s of setup**; those fields are not a complete accounting of project overhead. Full experiment history includes unsuccessful attempts beyond these selected stages.

The summary uses rounded totals; exact values are retained below and in the JSON ledger. The same 4,096 world slots are reused across rounds. The approximately 180 hours are summed across worlds and episodes, including resets. Each experience (a transition) is one world's **20 ms** action interval, not one rendered frame. Reusing an experience for multiple PPO epochs does not count it as new simulated experience. Recorded transitions exclude rollouts discarded at a training deadline.

**Scale and training mechanics**

| Statistic | Measured setting or derived value |
| --- | --- |
| Physics | MuJoCo Warp on CUDA; **500 Hz**, 2 ms timestep |
| Policy frequency | **50 Hz**, 20 ms per action; **10 physics steps per action** |
| Physics work represented by extension data | **323,420,160 world-physics steps**, derived from recorded transitions |
| Aggregate experience rate | **987 simulated seconds per wall-clock second**, summed across worlds |
| PPO rollout | **24 steps × 4,096 worlds = 98,304 experiences**; 0.48 s per world |
| PPO updates | **Four epochs**, minibatches of **3,072**; **128 optimizer steps** per complete rollout |
| Extension totals | **329 recorded rollouts**, **41,819 actual optimizer steps** |
| Optimizer | Adam, learning rate **0.0001**; PPO clipping **0.2** |
| Return estimation | Discount **0.99**, GAE lambda **0.95** |
| CPU work | NumPy observation/reward assembly; host coordination and transfers |
| GPU work | Batched rigid-body/contact physics, policy inference, actor/critic optimization; selected standing/moving runs also accumulate contact summaries within GPU physics steps |

Time budgets can stop a final PPO update before all four epochs finish, which explains why the actual optimizer-step total is below 329 × 128. The 987× figure is aggregate experience production, not each world running 987× faster than real time. Camera rendering is outside training.

The environment computes each world's weighted reward every **20 ms**, using measured task progress, posture, motion, support and other configured terms. The critic learns expected discounted future reward; it does not award the reward. PPO uses returns and critic-based advantages to update the shared actor. Moving-world rewards measure holding and slip relative to the physical support, including its rotation. [Training implementation](../../experiments/adaptive_locomotion/src/adaptive_locomotion/train.py)

**Small networks, shared weights**

| Network | Architecture | Parameters | Role |
| --- | --- | ---: | --- |
| Actor | **86 → 128 ELU → 128 ELU → 12** | **29,196** | Produces joint-position offsets; finite-torque servos move the robot |
| Critic | **90 → 128 ELU → 128 ELU → 1** | **28,289** | Estimates future return during training |

These counts describe the two MLPs. Twelve learned exploration standard-deviation parameters and normalization buffers are separate; the package's history estimator is unused in these support-mode runs. The deployed actor is feed-forward, with no recurrence or explicit gait-phase clock. The critic's four extra channels are simulator velocity and height. RGB cameras are for observation, not policy input. [Network source](../../experiments/adaptive_locomotion/src/adaptive_locomotion/policy.py)

The final moving round combines **2,048 healthy dogs on moving platforms** and **2,048 static rehearsal worlds**. The static half spans **27 body/surface groups**: nine body configurations on flat ground plus 18 healthy nonflat supports. All worlds contribute to the same actor. Training also uses **36,000 saved walking examples** and, in the corrective moving round, **54,000 standing examples** to preserve earlier behavior. Teachers and rehearsal datasets are not used at deployment.

The nine body configurations are the intact dog, any one entire lower leg removed, and any one entire leg removed. Removal changes mass, geometry and active actuators. The healthy body is **15.206 kg / 12 actuators**; lower-leg removal **14.965 kg / 11 actuators**; whole-leg removal **13.135 kg / 9 actuators**. Hip/thigh torque caps are **23.7 N·m**, calf caps **45.43 N·m**. Missing joints occupy masked slots in the common twelve-action output. Mass and motor-strength randomization are disabled in these runs. [Physical specification](standing/PHYSICAL_SPEC.json)

**Walking: measured GPU acceleration**

Both sides use CUDA policy inference and learning on the same Ryzen 5950X / RTX 4090 desktop. The comparison changes CPU MuJoCo/mjbatch physics to MuJoCo Warp, while matching starting weights, network, rewards, experience and optimizer updates. We repeat the comparison three times with different random sampling. Each run uses **six PPO rollouts, 589,824 transitions and 768 optimizer steps**. Across the three GPU runs, that totals **42.321 seconds and 1,769,472 experiences**.

| Run | CPU physics + CUDA learning | Warp physics + CUDA learning | Speedup | Warp-trained task completions |
| --- | ---: | ---: | ---: | ---: |
| 1 | 46.740 s | 15.286 s | 3.06× | 70/72 |
| 2 | 45.269 s | 13.542 s | 3.34× | 71/72 |
| 3 | 44.763 s | 13.493 s | 3.32× | 64/72 |
| **Combined** | **136.772 s** | **42.321 s** | **3.23×** | **205/216** |

The CPU-trained controls complete **200/216** tasks. Both sets are evaluated on CPU physics; all 432 trials stay upright with allowed support, and the declared gait-retention comparison passes. These are three short continuation experiments, not a dog learning to walk from scratch in fifteen seconds.

Including cached-kernel setup, average short-process time is **48.17 s CPU vs 18.85 s Warp**, a **2.56×** speedup. First-time kernel compilation is excluded. Peak sampled total device memory is **6,808 MiB for Warp** versus **4,468 MiB for CPU physics with CUDA learning**, including other GPU workloads. These are whole-device measurements, not isolated process memory or a guaranteed memory requirement.

The earlier **512-world** comparison achieved **1.71×** speedup but failed gait/completion acceptance. A longer matched **3,932,160-transition** continuation took **90.406 s on Warp vs 301.874 s on CPU**, a **3.34×** speedup, but both policies regressed and were not promoted. Under equal approximately 90-second budgets, Warp collected **3.64× more experience**; more updates alone did not improve the gait. [Complete benchmark and retained failures](WARP_TRAINING.md)

**What the GPU-trained policies achieved**

| Check | Result | Evaluation physics |
| --- | --- | --- |
| Standing: healthy terrain and walk–hold–walk | **48/80 strict passes; 80/80 upright** | Warp |
| Standing: damaged bodies on flat ground | **48/64 strict passes; 64/64 upright** | Warp |
| Standing total | **96/144 strict passes; 144/144 upright** | Warp |
| Moving-platform holdout | **22/24 strict passes**, up from **16/24** before training; **24/24 upright** | Warp; same counts on CPU |
| Walking retained after standing and again after moving training | **36/36 tasks** and all original gait-quality retention checks at each stage | CPU, 0.5 ms physics / 20 ms control |
| Damaged moving-platform transfer, not trained | **0/8 strict passes; 6/8 upright** | Warp; same counts on CPU |

Strict standing checks include drift, speed, tilt, allowed support, torque caps and sampled penetration; upright survival alone is not a pass. Static terrain evaluation uses new reset perturbations on the same **19 presets** seen in training. Slopes reach **24°**, step risers **28 cm**, and pad height differences **24 cm**; the hardest conditions retain failures. Moving training covers the healthy dog and five motion families: translation, yaw, heave, rocking and combined motion. Its 24 holdout checks include four stationary-deck controls. The two remaining trained failures involve unintended link support.

The platform is physically actuated and provides ideal current pose/twist sensing. The experiment does not establish unseen-terrain generalization, arbitrary damage recovery, damaged moving-platform reliability or hardware deployment. Reset-only inverse kinematics supplies feasible initial standing poses. [Standing results](standing/README.md) · [Moving results](moving/README.md)

**Per-round GPU training ledger**

| Round | Seconds | Transitions | PPO rollouts | Actual optimizer steps |
| --- | ---: | ---: | ---: | ---: |
| [Healthy standing 1](standing/healthy_60s_seed12/training.json) | 59.749 | 3,440,640 | 35 | 4,480 |
| [Healthy standing 2](standing/healthy_120s_seed13/training.json) | 59.563 | 3,833,856 | 39 | 4,992 |
| [Healthy standing 3](standing/healthy_180s_seed14/training.json) | 59.743 | 4,128,768 | 42 | 5,376 |
| [Mixed bodies](standing/mixed_all_60s_seed12/training.json) | 59.747 | 2,359,296 | 24 | 3,072 |
| [Support refinement](standing/mixed_support_60s_seed13/training.json) | 59.528 | 2,457,600 | 25 | 3,200 |
| [Substep support](standing/substep_120s_seed14/training.json) | 119.202 | 5,308,416 | 54 | 6,842 |
| [Standing consolidation](standing/consolidate_120s_seed12/training.json) | 119.202 | 5,308,416 | 54 | 6,813 |
| [Moving round 1](moving/round1_training.json) | 59.201 | 3,145,728 | 32 | 3,972 |
| [Moving round 2](moving/round2_training.json) | 59.353 | 2,359,296 | 24 | 3,072 |
| **Total** | **655.288** | **32,342,016** | **329** | **41,819** |

Totals use unrounded measurements. Historical files are preserved: the mixed-bodies run records `source_dirty=true`, and older `body_environment_counts` omit repeated healthy terrain entries. This report verifies **4,096 worlds from each authoritative `standing_cases` list**. The complete raw records retain failed intermediate evaluations.

The recorded GPU stack is **MuJoCo 3.13.0, MuJoCo Warp 3.13.0, NVIDIA Warp 1.17.0, PyTorch 2.14.0+cu130, Python 3.14.7 and uv 0.12.12**. CPU observations/rewards mean this is a hybrid training pipeline even though physics and neural learning run on the GPU.

The final combined film uses **three successive checkpoints**, one shared actor within each stage; it does not show the latest weights in every chapter. It contains **39 recorded trials**, **1920×1080 at 25 fps**, **2 min 04 s** total: **118 s at 1×** plus a six-second results card. Following, overhead and head cameras are synchronized. The graphite styling and covered branding are native observer-only presentation changes. [Film provenance](graphite/COMPLETE_VIDEO.md)

For ancestry context only, the starting walker carries **32m 03s of earlier Mac training**; adding the GPU extensions gives the latest checkpoint **42m 58s total across both machines**. Neither is a GPU-only training duration. No new simulation, training or video generation was performed to compile this report.
