# Healthy walking: correcting knee support

User request: “i think we need to penalize walking on anything other then the foot, unless the foot is not there? the healthy dog is walking, his back right leg on the elbow, instead of foot/paw”. Follow-up started **September 10, 2026, 15:02:05 UTC**.

**[Watch the corrected walk](../../previews/locomotion/healthy_walk_feet.mp4)** · **[Before/after comparison](../../previews/locomotion/healthy_support_fix.mp4)**

[![Corrected healthy walking](../../previews/locomotion/healthy_walk_feet.png)](../../previews/locomotion/healthy_walk_feet.mp4)

The user was right: the original healthy policy rested on its rear-right knee housing in 155 of 600 sampled frames. Checking four foot contacts and absence of trunk contact missed this. Under the new force-based criterion, the old policy completes the distance in 32/32 development trials but has **0/32 valid walks**. [Legacy failure report](HEALTHY_LEGACY_SUPPORT.json).

## What changed

Every physical robot collision geom now has a native MuJoCo ground-contact force sensor. The allowed surfaces follow the actual body:

| Physical leg | Allowed ground support |
| --- | --- |
| Intact calf and foot | Terminal foot surface |
| Shortened calf | Its modeled terminal stump surface |
| Calf and distal joint removed | Designated stump at the remaining thigh tip |
| Weak motor, unchanged leg | Same terminal surface as before weakening |

Knee housings, shafts, thighs, hips and the chassis remain penalized. Collisions, masses, torque limits and joint limits are unchanged. No gait, foot trajectory, symmetry or forced posture was added. The actor still receives its original 66 observations; support sensors are used by the reward and validator only.

The new reward subtracts `support_weight * cost`, with default weight 2 and:

```text
cost = 0.25 * sum(min(F_geom / 5 N, 1))
     + 2 * min(sum(F_geom) / body_weight, 2)
```

Here `F_geom` is the magnitude of the net ground-contact force on each forbidden geom. This penalizes both contact presence and substantial load. The weight applies to newly trained `walk` and `adaptive` runs; `--support-weight 0` reproduces the earlier reward behavior.

MuJoCo's native contact sensors aggregate actual solver contact forces using `reduce="netforce"`, returning the resultant in world coordinates. [Official sensor reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact). Training samples these readings at the 50 Hz control rate for batching speed. Evaluation checks every 2 ms physics step, including between actions. An unintended geom force over **1 N** anywhere in a trial makes its support invalid. Sensor readings reflect the contact solve of each step, before integration; geometric touch with no reaction force is not load-bearing support.

## Training and measured result

We resumed the 119.161-second healthy policy for **89.412 seconds**, collecting another **2,088,960 control transitions**. Its complete lineage totals **208.573 seconds**, **4,890,624 transitions**, from random initial weights. PPO used 512 native CPU MuJoCo/mjbatch environments and MPS network updates on the Mac. No RTX training was needed for this correction. [Training configuration and source commit](runs/healthy_feet_210s_seed2.json).

| Evaluation | Valid support completions |
| --- | ---: |
| Development, seed 9137 | 32/32 |
| Fresh final initial conditions, seed 20260912 | **64/64** |
| Half timestep, 1 ms | **16/16** |

A valid completion reaches 5 m, stays controlled in the goal lane for one second, survives the whole 12-second trial and has no forbidden support above 1 N. All forbidden geom forces were actually **0 N** in both final sets. The inspected rollout has a mean body height of **30.1 cm**, all four feet making and breaking contact, no trunk contact, and maximum sampled penetration of **4.52 mm**. Posture and penetration diagnostics are sampled every 20 ms; support-force acceptance is checked every physics step. [Full validation](FOOT_SUPPORT_VALIDATION.json).

The before/after clip uses matching initial conditions and physics, preserves the failing old gait, and labels its support invalid. The comparison also includes additional training, so it is not an equal-budget reward ablation. This is one training seed, a healthy robot and level ground. The stump exception is physically unit-tested; damaged-body policies have not yet been retrained with this cost. Arbitrary damage and parkour remain future work. Net force is appropriate for these ground-support tests; opposing contacts in future confined scenes would require per-contact magnitudes to avoid cancellation.

## Run and verification

From the repository root:

```sh
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog walk --style compact
```

The compact style selects `healthy_feet_210s_seed2.pt`. The default `walk` now selects the subsequent [longer-stride refinement](LONGER_STRIDE.md). To repeat the refinement as a new run:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/feet_repeat --seconds 90 \
  --mode blind --bodies healthy --terrain flat --reward-profile walk \
  --support-weight 2 --seed 2 --num-envs 512 --threads 16 \
  --resume ../../assets/locomotion/checkpoints/healthy_walk_120s_seed2.pt
```

A wall-time budget can produce different update counts on different machines. Resuming restarts Adam, as in the earlier pilot. The selected checkpoint retains its source commit, parent and hashes; original checkpoints and videos are preserved.

**52 tests passed**, including an actually loaded forbidden knee, an allowed missing-calf stump, and exact dynamics agreement with sensors and substep instrumentation enabled/disabled. CPU/MPS action agreement was within **7.16e-7**. The native Mac viewer passed a five-second smoke test. Both twelve-second videos are 1280×720, 25 fps, 1× playback; all **600 frames** decode. The corrected main clip has synchronized following, head and overview cameras; camera images remain observer output. [Delivery checks](FOOT_SUPPORT_DELIVERY.json), [separate development/training time log](../TIME_LOG.md).
