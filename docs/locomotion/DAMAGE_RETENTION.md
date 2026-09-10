# Damaged walking with healthy gait retention

Started September 10, 2026 at **17:09:52 UTC**.

User request: “could we do again the damaged training, but rewarding the ones that preserve the healthy walk better?” Preserve the balanced healthy policy and all prior experiments on `experiment/adaptive-dog`.

Start from the 403.329-second balanced walker. Train for 90 seconds, with permission already granted for a cumulative ten-minute lineage if a short extension is supported by results. Keep a frozen copy of the balanced actor as a training reference. On intact/full-strength states only, add `0.5 * exp(-mean((action-reference_action)^2)/0.25)` to reward and `2 * mean((actor_mean-reference_action)^2)` to the PPO loss. The latter averages over healthy samples only. This is soft retention during training, not a second controller at inference. No teacher action is applied to MuJoCo. Freeze the inherited observation normalization, and lower Adam's learning rate from 0.001 to 0.0003. The optimizer restarts as in previous follow-ups.

The 512-environment curriculum allocates 384 to intact bodies and 32 to each of four 70%-length calf variants. At each intact reset, two thirds remain healthy; one third gets a single motor weakness, half from the beginning and half at 2–5 seconds. This gives approximately 50% healthy, 25% weakened and 25% shortened-body episodes before early-termination effects. Strength is 40–80% of normal torque capacity. Shortened variants are not additionally weakened in this round. Geometry changes at episode setup, never through a mid-walk pose edit. Weakness changes torque limits; it does not remove a foot or permit knee support.

Retain support weight 2, stride weight 1, body-motion weight 0.5 and the walking posture costs. Balance weight 5 remains active only for healthy states. Damaged bodies keep progress, physical support, stride and stability rewards; exact healthy symmetry and teacher imitation are disabled. The actor still receives 66 proprioceptive/range/command channels, with no private damage labels, and produces twelve independent actions. The teacher, health mask and private critic context are training-only.

Development seed: **9137**. Save intermediate checkpoints every 25 iterations. Before final evaluation, select by strict healthy retention gates: all 16 development trials complete with allowed support; mean healthy stride 0.28–0.35 m; speed within 10% of the frozen reference; mean left/right duty gap at most 0.12; body-height standard deviation at most 0.011 m. Among qualifying candidates, favor the mean support-valid completion rate across all four shortened-calf positions, then their mean distance. A useful improvement is at least 20 percentage points over the reference across these four cases. Do not choose by video appearance alone.

Freeze final evaluation seed **20260915**, with 32 paired trials per condition, plus half-timestep checks on healthy and shortened cases. Include held-out 58%-length and paired-shortening bodies and the existing 25%-strength unexpected motor fault, even if unsuccessful. Predetermine demonstration cases `healthy`, `short_fr` and `unseen_weak`, using seed 9137 trial 0. Cameras remain observer output, with scripted lane velocity commands and learned joint control. Preserve failure results and all actual training time, including unselected updates.

## Result and videos

**[Watch the three-camera demonstration](../../previews/locomotion/damage_retention.mp4)** · **[Healthy gait before/after](../../previews/locomotion/damage_healthy_retention.mp4)** · **[Shortened-calf before/after](../../previews/locomotion/damage_short_fr_comparison.mp4)**

One **89.637-second** fine-tuning run was sufficient. The selected final checkpoint has **492.966 seconds (8 min 13 s)** of cumulative training and **10,149,888 total trained control transitions**. This round contributed 823,296 transitions over 67 PPO iterations. Training used CPU MuJoCo/mjbatch with an MPS learner on the Mac; the remote GPU was not needed for this run. Model/trainer setup took 0.740 seconds outside the training clock. There was no extra training extension or discarded training fork in this round.

Selection followed the declared gates. Iteration 25 preserved healthy gait but passed only 51/64 shortened-calf trials. Iteration 50 passed all 64 damaged trials but failed the healthy duty-gap limit, so it was rejected. The final checkpoint passed both. All three checkpoints and their development reports are retained. [Selection record](DAMAGE_RETENTION_SELECTION.json), [full training provenance](runs/damage_retention_495s_seed2.json).

Fresh evaluation uses seed 20260915, 32 matched initial conditions per case, twelve seconds without resets. Completion requires five meters of progress, one second controlled in the destination lane, full-trial survival, and no forbidden ground support above 1 N at any physics substep.

| Physical condition | Healthy reference | Damage + retention |
| --- | ---: | ---: |
| Intact dog | 32/32 | **32/32** |
| Front-left calf at 70% length | 0/32 | **32/32** |
| Front-right calf at 70% length | 0/32 | **32/32** |
| Rear-left calf at 70% length | 0/32 | **32/32** |
| Rear-right calf at 70% length | 0/32 | **32/32** |
| Unseen front-left calf at 58% length | 0/32 | **32/32** |
| Unseen pair: FL 75%, RR 65% | 0/32 | **1/32** |
| FR thigh suddenly limited to 25% torque at 3 s | 32/32 | **32/32** |

The motor-weakness test was already handled by the reference; it does not demonstrate a new capability. Paired shortening still fails reliably. This is progress on single shortened calves while retaining intact walking, not arbitrary-damage recovery, missing-leg mastery or parkour.

| Healthy gait measurement | Reference | After damage training |
| --- | ---: | ---: |
| Mean stride | 31.30 cm | **30.60 cm** |
| Actual forward speed | 0.609 m/s | **0.600 m/s** |
| Left/right duty-factor gap | 8.23 percentage points | 10.54 points |
| Body-height standard deviation | 7.87 mm | 7.07 mm |
| Planted-foot horizontal speed, RMS | 0.052 m/s | 0.061 m/s |
| Roll/pitch angular rate, RMS | 0.136 rad/s | 0.105 rad/s |

Healthy walking is largely retained, with slightly more timing asymmetry and slip. Damaged walking is allowed to differ: across the four shortened positions, mean strides span 26.3–32.8 cm and speed spans 0.530–0.740 m/s for a nominal 0.55 m/s command. Per-leg stance, swing and load measurements are saved; passing the course does not establish precise speed tracking or identical gait. [Selected validation](DAMAGE_RETENTION_VALIDATION.json), [matched reference](DAMAGE_RETENTION_REFERENCE.json).

All **128/128** trained-length damaged trials and **32/32** healthy trials passed support validation. Another **40/40** trials across those five conditions passed at a 1 ms timestep. Forbidden support force was zero throughout these sets, and measured torque never exceeded the effective actuator cap. The three selected recordings have maximum sampled penetration of 3.77 mm healthy, 2.37 mm shortened FR and 4.09 mm after weakness. Masses, joint limits, servo settings, contact geometry and friction are unchanged from the [documented model](README.md#physical-specification-and-verification).

This experiment combines mixed training, healthy reference rewards, soft action retention, frozen normalization and a smaller learning rate. It does not isolate which ingredient caused the improvement. The teacher only regularizes training; the saved actor runs alone. One training seed and flat-ground initial-condition variation do not establish broad robustness.

## Run and reproduce

From the repository root:

```sh
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/damage_retention_495s_seed2.pt \
  --case short_fr
# Use --case healthy or --case unseen_weak for the other video conditions.
```

Repeat the training in a new output directory:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/retention_repeat --seconds 90 \
  --allowance 600 --extension-reason 'Preserve healthy walking while learning damage' \
  --mode blind --bodies all --terrain flat --reward-profile walk \
  --support-weight 2 --stride-weight 1 --balance-weight 5 --body-motion-weight 0.5 \
  --retention-curriculum --seed 2 --num-envs 512 --threads 16 \
  --resume ../../assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt \
  --reference ../../assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt \
  --reference-reward-weight 0.5 --reference-loss-weight 2 --learning-rate 0.0003

uv tool run --from uv==0.12.12 uv run --locked python -m adaptive_locomotion.retention_eval \
  --checkpoint ../../assets/locomotion/checkpoints/damage_retention_495s_seed2.pt \
  --output ../../outputs/locomotion/retention_check.json --trials 32 --seed 20260915 --final
```

Wall-time limits can produce different iteration counts across machines. The published weights came from source commit `4fd45ed`, with a clean source tree at training setup. The frozen teacher hash is recorded and the original teacher remains unchanged. Intermediate checkpoints come from this same 89.637-second run and must not be counted as separate training jobs.

**58 tests passed**, including healthy/damage reward gating, zero teacher gradients on damaged samples, curriculum segregation, actual fault torque caps, and the existing native dynamics suite. CPU/MPS action agreement was within 1.91e-6 and native macOS viewing passed. The 36-second main video and two twelve-second comparisons are 1280×720, 25 fps, 1×; all **1,500 encoded frames** decode. Opening, stride and final frames were inspected. The overview was reframed by replaying the saved trajectory, with no new physics rollout. Contact dots and the torque-strength display are observer annotations. [Delivery checks](DAMAGE_RETENTION_DELIVERY.json), [time log](../TIME_LOG.md).
