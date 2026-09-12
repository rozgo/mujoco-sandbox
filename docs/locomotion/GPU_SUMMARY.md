# One adaptive dog policy, trained on an RTX 4090

**22 minutes 5 seconds of GPU training. 4,096 worlds in parallel. One final policy for walking, balancing and moving supports.** The same actor controls an intact dog and eight physical limb-removal variants. All training starts from random weights and follows the recorded successful curriculum.

| What it learned | GPU learning time | New experiences |
| --- | ---: | ---: |
| Healthy walking | **1m 54s** | **9.63 million** |
| Walking with missing limbs | **9m 39s** additional | **29.79 million** |
| Standing on pads, gaps, slopes and steps | **8m 23s** additional | **26.84 million** |
| Balancing on moving platforms | **2m 09s** additional | **5.51 million** |
| **Complete shared policy** | **22m 05s** | **71.76 million** |

All four phases use **4,096 worlds**. Healthy and damaged walking together take **11m 33s**. These are successive stages of one policy's training, not separate deployed controllers. Values are rounded independently.

## The useful scale numbers

- **399 aggregate simulation hours—16.6 days of experience—in 22 minutes of training.** That experience is summed across worlds and resets.
- **54,159 new experiences per second**, including physics, rewards, transfers and learning. Each experience is one world's 20 ms action interval.
- **717.6 million world-physics steps.** MuJoCo Warp runs physics at **500 Hz**; the policy acts at **50 Hz**, with ten physics steps per action.
- **98,304 experiences per PPO rollout:** 24 steps × 4,096 worlds, followed by four optimization epochs. The complete run performs **730 rollouts and 93,440 optimizer steps**.
- **A small actor:** 86 inputs → 128 → 128 → 12 actions, **29,196 parameters**. A separate critic has 90 inputs → 128 → 128 → 1 output and **28,289 parameters**. Both use ELU hidden layers.
- **One actor learns from every world.** The final phase combines **2,048 moving-platform worlds** with **2,048 static rehearsal worlds**. Existing walking and standing rehearsal preserve earlier skills; those references are used only during training.
- **Real physical body changes:** the nine configurations cover a healthy dog, any one entire lower leg removed, and any one entire leg removed. They have **12, 11 or 9 active actuators**; absent joints are masked within the same twelve-action output.
- **Previously measured acceleration:** a separate matched walking continuation benchmark measured **3.23× faster training with Warp physics than CPU physics**. That benchmark is distinct from this from-scratch curriculum replay.

## What the final checkpoint can do

Every test below uses the same frozen weights; there is no checkpoint switching.

| Held-out check | Result |
| --- | ---: |
| Walking, across all nine bodies | **36/36 tasks completed** |
| Healthy and damaged static balance, including walk–hold–walk | **136/144 strict passes** |
| Healthy moving-platform balance | **24/24 strict passes** |
| Remaining upright across the complete task audit | **204/204** |

Walking is also checked after the walking and standing phases: both pass **36/36 tasks** and the established healthy gait checks. Final healthy stride averages **32.2 cm**, speed **0.558 m/s**, and body-height variation **4.6 mm**.

The eight static misses are retained: five unintended-link-support cases and three cases above the drift threshold. These are trials on the existing surface presets, not proof of arbitrary damage recovery, unseen-terrain generalization or real-hardware readiness. Moving training covers the healthy dog.

## What the timing includes

The **22m 05s** includes the actual training loop: physics, CPU observations/rewards, transfers, PPO updates and the recipe's existing rehearsal losses. Reused examples do not inflate the new-experience count. Setup is separate (**187.8 s**); the complete process span is **26m 37s**. The final native CPU evaluation takes **50.1 s**, outside training time. Video production is also separate.

The actor and physics are trained on the RTX 4090. Observation and reward assembly remain on the CPU. The final evaluation and new Ember footage execute the frozen GPU-trained policy in native CPU MuJoCo. Cameras show the scene; camera pixels are not policy inputs. High-level velocity commands are supplied by the demo; the actor learns joint control.

The prior accepted film and checkpoints are preserved. Earlier abandoned attempts are excluded from the selected policy's training time and remain documented in the full report.

[Complete measured report](GPU_REPORT.md) · [Exact ledger](GPU_REPORT.json) · [Reproduction commands](gpu_from_scratch/README.md) · [Matched speed benchmark](WARP_TRAINING.md)
