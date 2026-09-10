# Longer-stride walking experiment

Started September 10, 2026 at **16:16:47 UTC**.

User request: “could we reward a longer gait, so it does look like its taking very short steps... its looking good otherwise”. The intended change is longer steps at the same requested speed. Preserve the current foot-valid policy and its videos.

The inspected existing gait advances each foot approximately 21 cm between placements, with approximately three cycles per second. Add an optional contact-event reward for forward swing displacement on landing, retaining the foot/stump support rule. The reward has no prescribed footfall order, phase clock, mirrored actions or target foot trajectories. It saturates at 40 cm, subtracts a 30 cm per-landing offset, and penalizes supporting foot slip and all-feet-off-ground flight. Air time alone earns no reward. The new `--stride-weight` defaults to zero so existing training behavior remains available.

Start with 90 seconds of refinement from the 208.573-second healthy policy, keeping total lineage under five minutes. Select on development seed 9137. Before final evaluation, require at least 25% greater mean touchdown-to-touchdown stride, no more than 10% increase in actual forward speed, unchanged support-valid completion, and no excessive hopping or penetration. Reserve final seed 20260913 for 64 trials and 16 half-timestep trials. Record failures as well as successes. Longer training is justified only by measured progress under the user's standing short-training allowance.

## Result

**[Watch longer strides, all three cameras](../../previews/locomotion/healthy_walk_longer.mp4)** · **[Compare with the compact gait](../../previews/locomotion/healthy_stride_comparison.mp4)**

The first refinement met the development criteria, so no extra training was needed. It ran for **89.216 seconds**, collecting **1,978,368 additional control transitions**. Complete policy lineage: **297.789 seconds**, **6,868,992 transitions** from random initial weights. Training used 512 CPU MuJoCo/mjbatch environments, a CPU actor and MPS updates on the Mac; no GPU physics or supplied gait.

The following measurements use matched fresh initial conditions, **64 trials per policy**, twelve seconds each, seed 20260913. Stride means forward distance between successive landings of the same foot; cadence is cycles per second per foot. Both are averaged over legs and trials after the first second. Position and force measurements are sampled every 20 ms, with a one-sample contact debounce. This diagnostic measures successive landings independently of the reward's liftoff-to-landing displacement.

| Measurement | Compact gait | Longer-stride gait |
| --- | ---: | ---: |
| Mean stride | 21.68 cm | **36.28 cm (+67%)** |
| Mean cadence | 2.86 cycles/s | **1.71 cycles/s** |
| Actual forward speed | 0.616 m/s | 0.618 m/s |
| Planted-foot horizontal speed, RMS | 0.185 m/s | **0.079 m/s** |
| Body height variation, standard deviation | 0.43 cm | 1.27 cm |
| Samples with all feet unloaded | 0% | 1.34% |

The larger strides bring more body motion and brief unloaded phases; this is not a guarantee of a strict no-flight walking gait. Foot slip is reduced, not eliminated. The fixed command is 0.55 m/s and both policies still travel slightly faster than requested. The comparison includes extra training and is not an equal-budget reward ablation. One training seed and level ground do not establish damaged-body adaptation or parkour.

The selected policy passes **64/64** support-valid completions and **16/16** at a 1 ms timestep. Every physics substep is checked; all forbidden support forces are **0 N** in both sets. The inspected rollout's maximum sampled penetration is **6.38 mm**, below the existing 8 mm check, with no trunk contact and unchanged actuator caps. [Full paired measurements and physical validation](LONGER_STRIDE_VALIDATION.json), [training provenance](runs/healthy_stride_300s_seed2.json), [development completion](runs/healthy_stride_300s_seed2_development.json), [development stride measurements](runs/healthy_stride_300s_seed2_stride_development.json).

## Reward details and reproduction

With `--stride-weight 1`, each qualified touchdown contributes `8 * (clip(forward_swing_metres, 0, 0.4) - 0.30)`. Qualification requires a contact transition after at least 60 ms without contact; force hysteresis is 5 N to establish contact and 1 N to retain it. Forward direction comes from the velocity command transformed into world coordinates. Supporting horizontal slip adds a cost of `0.15 * sum(min(foot_speed_squared, 4))`; having no loaded terminal adds 0.5. These terms operate only when commanded planar speed exceeds 0.15 m/s. The reward does not specify timing between different legs. The existing posture, velocity, torque and forbidden-support terms remain active.

From the repository root:

```sh
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog walk
# The previous good version remains directly available:
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog walk --style compact
```

To repeat the refinement as a new run:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/stride_repeat --seconds 90 \
  --mode blind --bodies healthy --terrain flat --reward-profile walk \
  --support-weight 2 --stride-weight 1 --seed 2 --num-envs 512 --threads 16 \
  --resume ../../assets/locomotion/checkpoints/healthy_feet_210s_seed2.pt
```

Wall-time budgets can yield different update counts on different machines; resuming restarts Adam. The compact policy and previous videos are unchanged. `--stride-weight 0` is the training default; this optional style was only trained and validated on the healthy body.

**54 tests passed**, including preference for longer forward swings, no reward for hovering/sliding, reset-state handling, original support-force tests and the native batch suite. CPU/MPS action disagreement was below **9.54e-7**. The native macOS viewer passed a five-second check. Both videos play at 1×, 1280×720, 25 fps, twelve seconds; all **600 frames** decode. Opening, stride and final frames were visually checked. Following, head and overview cameras are synchronized observer output. [Delivery checks](LONGER_STRIDE_DELIVERY.json), [elapsed time](../TIME_LOG.md).
