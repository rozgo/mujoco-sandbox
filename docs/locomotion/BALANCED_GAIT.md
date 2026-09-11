# Healthy gait balance using established rewards

Started September 10, 2026 at **16:45:40 UTC**.

User request: “ok, lets go with know rewards first, lets train”. The preceding review found that longer strides retained uneven stance timing. Keep existing healthy policies and videos for comparison.

Implement Isaac Lab Spot's completed air/contact-duration variance and body-motion penalties in the existing MuJoCo/mjbatch learner. Start with balance weight 20 and body-motion weight 1, retaining support weight 2 and stride weight 1. The variance formula clips each last completed interval at 0.5 seconds and uses sample variance across four feet. The body cost is `0.8 * vertical_velocity² + 0.2 * sum(abs(roll/pitch_angular_velocity))`. No fixed phase, mirrored actions or diagonal-pair requirement is introduced in this first run. [Pinned source and license](../../experiments/adaptive_locomotion/third_party/isaaclab_rewards/PROVENANCE.md).

The timing regularizer is only active for intact geometry and full-strength motors during motion commands. This uses privileged training state, adds no actor observations and does not claim deployed damage detection. Body stability remains useful across body conditions. Both new weights default to zero for reproducibility of previous runs.

First budget: 90 seconds from the 297.789-second longer-stride policy. Its existing 64/64 support-valid progress and persistent uneven duty factors justify this short extension under the user's standing ten-minute allowance; all ancestry remains counted. Use development seed 9137. Reserve final seed 20260914 for 64 trials and sixteen half-timestep trials. Acceptance: halve the mean left/right duty-factor gap and bring it below 10 percentage points, retain mean stride at least 30 cm, preserve comparable speed (within 10%), reduce body-height variation, retain foot-only support and inspect the video. Keep failed candidates if more refinement is needed.

## First attempt

The 20/1 balance/body-motion weighting regressed in 89.732 seconds: **0/32** development completions, little forward progress, and forbidden support. The checkpoint and complete development failure report are retained; it is not selected. Restart from the original longer-stride policy using balance weight 5 and body-motion weight 0.5 for a 120-second trial. This tests less aggressive regularization rather than extending the failed policy.

## Selected result

**[Watch the balanced-gait version](../../previews/locomotion/healthy_walk_balanced.mp4)** · **[Compare against longer strides](../../previews/locomotion/healthy_gait_balance.mp4)**

The gentler run uses balance weight **5** and body-motion weight **0.5**. Both its intermediate and final checkpoints completed 32/32 development trials with valid support. We selected **iteration 200**, saved at **105.540 seconds**, because its mean left/right duty-factor gap was 8.22 percentage points versus 9.95 for the final checkpoint. Selection used development seed 9137 before accessing the fresh final seed.

The selected policy has **403.329 seconds (6 min 43 s)** of cumulative training, including its 297.789-second ancestor, and **9,326,592 total control transitions**. This follow-up actually spent **209.180 seconds training across two runs**: 89.732 seconds on the failed attempt plus 119.448 seconds on the successful run. The selected checkpoint predates the end of the latter run by 13.907 seconds. That additional compute is included in experiment cost, rather than hidden in the selected-policy figure.

Fresh paired measurements use 64 initial conditions per policy, twelve seconds per trial, seed 20260914:

| Measurement | Longer strides | Balance rewards |
| --- | ---: | ---: |
| Mean left/right duty-factor gap | 21.56 percentage points | **8.44 points (61% lower)** |
| Body-height variation, standard deviation | 1.27 cm | **0.78 cm (38% lower)** |
| Mean stride | 36.28 cm | **31.30 cm** |
| Actual forward speed | 0.617 m/s | 0.609 m/s |
| Planted-foot horizontal speed, RMS | 0.084 m/s | 0.052 m/s |
| Mean left/right share-of-load gap | 9.28 percentage points | 4.55 points |
| Samples with all feet unloaded | 1.33% | 0% |

Support duration remains measurably asymmetric:

| Foot | Previous ground-contact fraction | Selected ground-contact fraction |
| --- | ---: | ---: |
| FL | 47.3% | 48.4% |
| FR | 63.5% | 59.6% |
| RL | 62.3% | 54.4% |
| RR | 35.3% | 48.7% |

Duty-factor gap is the mean of absolute FL/FR and RL/RR differences per trial, then averaged over trials. It does not require front and rear legs to have identical loads. Duration and load diagnostics use 20 ms samples after the first second, with one-sample contact debounce; load share is based on the magnitude of sampled net terminal reaction force, not a claimed calibrated pressure measurement. [All paired measurements](BALANCED_GAIT_VALIDATION.json).

The selected policy passed **64/64 support-valid completions**, plus **16/16** at a 1 ms timestep. Every physics substep was checked and forbidden support forces remained **0 N** in both sets. In the inspected rollout, maximum sampled penetration was **3.00 mm**. Torque caps, body geometry, masses and timestep are unchanged. A scripted lane follower still provides velocity commands; the actor learns joint control.

This is improved healthy-gait balance, not perfect symmetry, a new training algorithm or arbitrary-damage recovery. The weights were tuned locally and the comparison includes additional training. We did not apply the stronger diagonal-pair gait reward, mirrored data augmentation or hard action tying. One training seed on level ground does not establish generalization across terrain or failures.

## Run and retained artifacts

From the repository root:

```sh
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog walk
# Earlier policies remain selectable:
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog walk --style longer
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog walk --style compact
```

Repeat the successful training recipe as a new run:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/balance_repeat --seconds 120 \
  --allowance 600 --extension-reason 'Test known gait regularizers on the longer-stride policy' \
  --mode blind --bodies healthy --terrain flat --reward-profile walk \
  --support-weight 2 --stride-weight 1 --balance-weight 5 --body-motion-weight 0.5 \
  --seed 2 --num-envs 512 --threads 16 \
  --resume ../../assets/locomotion/checkpoints/healthy_stride_300s_seed2.pt
```

Wall-time budgets can yield different update counts. The selected artifact is an intermediate checkpoint; the final checkpoint is retained separately. Source commits, original and curated hashes, and parent paths are in [selected training provenance](runs/healthy_balanced_405s_seed2.json), [complete successful run](runs/healthy_balance_420s_seed2.json), and [failed attempt](runs/healthy_balance_390s_seed2.json). Development timing and physical outcomes are retained alongside each report. No prior checkpoint or video was overwritten.

**56 tests passed**, including phase-independent timing balance, partial-reset isolation, inactive balance costs on a shortened body, and unchanged scalar/batch dynamics. CPU/MPS inference agreement and the native macOS viewer passed. Both videos are twelve seconds at 1280×720, 25 fps and 1×; all **600 frames** decode. Opening, stride and final frames were inspected, including label placement and contact indicators. Three cameras are synchronized observer output. Contact dots indicate an allowed-terminal force above 1 N anywhere in the displayed 20 ms control window; they are not policy inputs. [Delivery checks](BALANCED_GAIT_DELIVERY.json), [time log](../TIME_LOG.md).
