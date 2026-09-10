# Dedicated healthy-dog walking baseline

**Correction:** the original 119-second policy described below used its rear-right knee housing for ground support. The original distance and trunk checks missed that defect. `adaptive-dog walk --style compact` selects the [corrected 209-second policy](FOOT_SUPPORT_FIX.md), which passed 64/64 trials with no unintended support forces. [Watch the corrected video](../../previews/locomotion/healthy_walk_feet.mp4). The original reports and clips remain available as failure evidence.

Follow-up started September 10, 2026 at 14:48:45 UTC.

User request: "do we have a normal dog trained?" / "if not lets train a normal dog and normal walk".

The existing generalist policy already completed 32/32 healthy flat-ground tests,
but it was trained across damaged bodies. This follow-up trains a separate policy
from random weights with all four legs intact, full motor strength, no scheduled
faults and level ground. Existing generalist checkpoints are preserved.

The `walk` reward profile adds healthy posture preferences: a soft 0.30 m base-height
target, stronger upright and roll/pitch angular-velocity costs, a modest nominal
joint-posture cost, a hip-spread cost, and smoother actions. These are optimization
costs, not enforced poses, phase clocks, foot trajectories or mirrored actions.
MuJoCo dynamics and torque limits are unchanged. At this initial stage the `adaptive` reward was
unchanged; healthy-only training now explicitly disables initial motor weakening
as well as scheduled faults.

Start with 60 seconds of PPO using 512 batched environments, CPU MuJoCo and an MPS
learner. Continue only if evaluation indicates more training is useful, keeping
the complete checkpoint ancestry within the agreed short-run allowance.

## Original result (superseded)

**[Original knee-supported gait](../../previews/locomotion/healthy_walk.mp4)** · [Compare with the earlier generalist](../../previews/locomotion/healthy_vs_generalist.mp4)

The selected policy trained from random weights for **119.161 seconds total**
(59.642 s plus 59.519 s), collecting **2,801,664 control transitions**. No teacher,
vendor walking policy or pretrained checkpoint was used. Training ran on the Mac:
512 native CPU environments and MPS neural-network updates, with the corrected
firm contact model throughout.

It completed **64/64** fresh-seed, twelve-second healthy trials and **16/16** at
half the physics timestep, reaching 5 m and staying controlled for one second.
This is one training seed on level ground. [Full final evaluation](HEALTHY_VALIDATION.json).

In the inspected rollout, mean body height is **27.7 cm**, versus **21.4 cm** for
the earlier generalist on the same healthy body. All four terminal feet make and
break ground contact: measured support fractions are 34%, 58%, 65% and 22%.
There is no sampled trunk-ground contact, maximum sampled penetration is 3.44 mm,
and actuator torque remains within the same hardware limits. The gait is still
asymmetric; we have not imposed a canonical footfall pattern or validated it on
hardware. The command remains a scripted lane-following velocity request.

An additional minute reduced the inspected rollout's forward-speed RMSE from
0.254 to 0.126 m/s; it still travels faster than the 0.55 m/s request on average.
Both one- and two-minute checkpoints are retained, along with their training and
development reports. The comparison is between different training tasks and
budgets, not an algorithm ablation.

## Watch the corrected policy or reproduce the original training

From the repository root:

```sh
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog walk --style compact
```

`walk --style compact --seconds 5` is a bounded viewer check. The shortcut selects
`assets/locomotion/checkpoints/healthy_feet_210s_seed2.pt` and the healthy scene.
The native Mac launcher was tested.

To reproduce the original reward experiment with a new seed (including its missing support penalty):

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/healthy_seed3 --seconds 120 \
  --mode blind --bodies healthy --terrain flat --reward-profile walk \
  --seed 3 --num-envs 512 --threads 16 --support-weight 0
```

This fresh 120-second run has a different seed and no intermediate optimizer
restart, so it is a new experiment rather than a bitwise reproduction. Exact
staged settings, source commits, checkpoint hashes and parent links are in
[the 60-second report](runs/healthy_walk_60s_seed2.json) and
[the selected policy report](runs/healthy_walk_120s_seed2.json).

Both original 12-second videos use actual recorded trajectories at 1×, 1280×720 and
25 fps. The main video has synchronized following, head and overview cameras;
camera pixels are not actor inputs. All 600 encoded frames decoded successfully.
The project plus vendored batch suite passed **48 tests**.
