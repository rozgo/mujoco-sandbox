# Paired damage while retaining earlier walking skills

Started September 10, 2026 at **19:57:30 UTC**. User approved the next step: paired partial-leg damage, preserving healthy and single-leg walking, with a 90-second first training allowance. Preserve all prior policies and media on `experiment/adaptive-dog`.

Start from the 492.966-second single-damage policy. Use two stages: 30 seconds at mild paired lengths (90% / 85%), then 60 seconds at stronger lengths (75% / 65%), counting both stages and the full parent ancestry. Four trained combinations: FL+RR, FR+RL, FL+FR and RL+RR. Hold out the left-side and right-side combinations entirely. These are discrete compiled bodies; geometry and inertia never change inside a live episode.

Each 512-environment batch contains 256 intact bodies, 32 for each of four single shortened calves, and 32 for each of four paired bodies. The existing intact reset rule leaves two thirds healthy and weakens one motor in one third, half initially and half after 2–5 seconds. Thus approximately 33% of episode assignments are fully healthy, 17% weakened, 25% single shortening and 25% paired shortening, before early-termination effects. Native worker counts are proportional to group size in the new paired stages (8+eight groups of 1), preserving physics while avoiding a bottleneck in the largest group.

Keep the previous healthy reward/retention recipe: healthy-reference reward 0.5 and loss 2, support weight 2, stride 1, healthy timing balance 5, body-motion 0.5, walking posture costs, Adam learning rate 0.0003 and frozen inherited observation normalization. Add a second frozen reference, the 492.966-second parent, with action loss weight 1 only on full-strength single-shortened states. Paired states receive no reference imitation and no intact timing requirement. Both reference networks are training-only; one 66-input reactive actor controls all twelve joints at inference. No damage labels enter its observation.

Development seed **9139**. Retention gates: all sixteen healthy, all 64 single-shortened and all sixteen motor-fault trials complete with allowed support; healthy mean stride 28–35 cm, speed within 10% of the published healthy reference, left/right duty gap at most 12 percentage points, and height standard deviation at most 11 mm. Among candidates passing every retention gate, select the highest average support-valid success over the four strong paired bodies, then the FL+RR target. Useful progress means at least 12/16 completions on that previously failing target. Keep all results if these gates fail.

Freeze final seed **20260916** before selection, with 32 paired initial conditions per case. Final-only tests add FL+RL and FR+RR at 78% / 68%, plus FL+RR at unseen lengths 68% / 78%. The previous `unseen_pair` case is now explicitly part of the training family and is called `pair_fl_rr_hard` in this experiment. Do not call it held out. Check healthy, single FR and target-pair trials at 1 ms as well. Physical support and torque rules remain unchanged; count failed trials and forbidden loads without resets.

Predetermined video cases: healthy, FL+RR target, and held-out left pair, seed 9137 trial 0, twelve seconds each. The target comparison uses identical physical geometry, initial conditions and camera settings before/after training. Videos and the viewer use physical rollouts; cameras and contact indicators are observers. Inspect a static paired-body preview before training, then complete numerical, support, gait and encoded-video checks. Record both training compute and elapsed task time separately.

## Selected result

**[Watch the paired-damage comparison](../../previews/locomotion/paired_damage_comparison.mp4)** · **[Three-camera demonstration](../../previews/locomotion/paired_damage.mp4)** · **[Healthy gait comparison](../../previews/locomotion/paired_healthy_retention.mp4)** · [Static preview](../../previews/locomotion/paired_damage_preview.png)

The mild stage took **29.703 seconds**, followed by **59.692 seconds** of stronger-pair training: **89.396 seconds actual training this task**, with 1,449,984 trained control transitions. CPU MuJoCo/mjbatch physics and MPS learning ran on the Mac; no new remote GPU training was needed.

The selected checkpoint is **iteration 25 of the hard stage**, after 17.953 seconds in that stage. Its cumulative ancestry is **540.622 seconds (9 min 1 s)** and **10,948,608 transitions**. Its new training ancestry is 47.656 seconds, while this task spent 89.396 seconds including later updates. Both figures are reported deliberately. [Selection and compute accounting](PAIRED_DAMAGE_SELECTION.json).

Development selection used seed 9139:

| Candidate | Healthy/single/motor retention gates | Strong paired completions |
| --- | --- | ---: |
| Previous single-damage policy | Pass | 17/64 |
| Mild stage, final | Pass | 27/64 |
| Hard stage, iteration 25 | **Pass; selected** | **64/64** |
| Hard stage, iteration 50 | Pass | 58/64 |
| Hard stage, iteration 75 | Pass | 64/64 |
| Hard stage, final | Fail: healthy speed outside limit | 64/64 |

Iterations 25 and 75 tied on the declared paired success criteria; the earlier one was selected to minimize training ancestry. This tie resolution preceded final evaluation. The later final checkpoint handled the trained damage but violated the predefined healthy-speed gate, so it was not selected. All five stage/candidate checkpoints and their full reports remain in [runs](runs/paired_selected_545s_seed2.json).

## Fresh paired evaluation

Seed 20260916, 32 matched initial conditions per row, twelve seconds without resets. Success requires five meters of progress, one controlled second in the goal lane, full-trial survival and no forbidden support force above 1 N at any physics substep.

| Condition | Previous policy | Selected policy |
| --- | ---: | ---: |
| Healthy | 32/32 | **32/32** |
| Four single-shortened positions, combined | 128/128 | **128/128** |
| FL + RR at 75% / 65% | 1/32 | **32/32** |
| FR + RL at 75% / 65% | 0/32 | **32/32** |
| Both front calves at 75% / 65% | 0/32 | **31/32** |
| Both rear calves at 75% / 65% | 32/32 | **32/32** |
| Held-out left-side pair at 78% / 68% | 32/32 | **32/32** |
| Held-out right-side pair at 78% / 68% | 30/32 | **32/32** |
| Held-out FL + RR lengths, 68% / 78% | 32/32 | **32/32** |
| FR thigh drops to 25% torque at 3 s | 32/32 | **32/32** |

Across the four trained pairs, support-valid completion improved **33/128 → 127/128**. Across the three held-out conditions, it was **94/96 → 96/96**; the parent already generalized well to most of these. Do not present all held-out performance as a capability created by this follow-up.

One front-pair trial failed at 0.96 seconds with a fall and forbidden body support. It remains in the denominator and the report. Its subsequent contact forces are recorded after failure too; they are not a successful gait. No MuJoCo numerical warning was accepted or reset away. [Selected validation](PAIRED_DAMAGE_VALIDATION.json), [matched reference](PAIRED_DAMAGE_REFERENCE.json).

| Healthy gait measurement | Previous policy | Selected policy |
| --- | ---: | ---: |
| Mean stride | 30.60 cm | 31.17 cm |
| Actual forward speed | 0.601 m/s | 0.578 m/s |
| Left/right duty-factor gap | 10.64 percentage points | 6.64 points |
| Body-height standard deviation | 7.07 mm | 8.90 mm |
| Planted-foot horizontal speed, RMS | 0.062 m/s | 0.050 m/s |

Healthy speed decreased about 3.8%, timing balance improved, and bounce increased. All healthy gait gates still passed. The diagonal damaged gait is intentionally different: mean stride 29.64 cm, forward speed 0.535 m/s, body-height variation 17.36 mm and a large 41.66-point left/right duty-factor gap. Its asymmetric support pattern is allowed; there is no hard symmetry constraint. Per-leg timing and load diagnostics are in the validation report. Course completion alone does not establish precise velocity tracking in every damage pattern.

At half timestep, healthy, single FR and target-pair cases passed **24/24** combined. Main-recording maximum sampled penetration is 2.99 mm healthy, 3.95 mm for the target pair and 5.52 mm for the held-out left pair. Those recordings have zero forbidden support and stay inside physical torque limits. The paired proxy masses are 15.164 kg mild and 15.104 kg strong, compared with 15.206 kg intact. Geometry, inertia and allowed distal surfaces are compiled from the existing body model; actuator caps, joint limits, contacts and friction follow the [physical specification](README.md#physical-specification-and-verification).

This is a short curriculum experiment with one training seed and selected flat-ground bodies. It does not establish arbitrary limb loss, continuous damage adaptation, noisy-sensor robustness, parkour or real-hardware transfer. Both reference actors are used only during training; the saved controller runs alone, with scripted velocity commands from ideal localization. No ablation isolates which of curriculum, mixed episodes, reference regularization or additional training caused each improvement.

## Run and reproduce

From the repository root:

```sh
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/paired_selected_545s_seed2.pt \
  --case pair_fl_rr_hard
# Other recorded cases: healthy, pair_holdout_left.
```

The two stages use these common flags in Bash or Zsh:

```sh
paired_flags=(
  --allowance 600 --mode blind --bodies all --terrain flat --reward-profile walk
  --support-weight 2 --stride-weight 1 --balance-weight 5 --body-motion-weight 0.5
  --retention-curriculum --seed 2 --num-envs 512 --threads 16 --learning-rate 0.0003
  --reference ../../assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt
  --reference-reward-weight 0.5 --reference-loss-weight 2
  --single-reference ../../assets/locomotion/checkpoints/damage_retention_495s_seed2.pt
  --single-reference-weight 1
)
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train "${paired_flags[@]}" \
  --output ../../outputs/locomotion/pair_repeat_mild --seconds 30 --pair-level mild \
  --extension-reason 'Mild paired damage while retaining prior skills' \
  --resume ../../assets/locomotion/checkpoints/damage_retention_495s_seed2.pt
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train "${paired_flags[@]}" \
  --output ../../outputs/locomotion/pair_repeat_hard --seconds 60 --pair-level hard \
  --extension-reason 'Progress to stronger pairs after checking mild-stage retention' \
  --resume ../../outputs/locomotion/pair_repeat_mild/policy.pt
uv tool run --from uv==0.12.12 uv run --locked python -m adaptive_locomotion.paired_eval \
  --checkpoint ../../assets/locomotion/checkpoints/paired_selected_545s_seed2.pt \
  --output ../../outputs/locomotion/pair_check.json --trials 32 --seed 20260916 --final
```

Evaluate the mild stage before progressing when repeating the experiment. Wall-time limits produce different update counts across machines. Both published stages started from clean source commit `a37117b`; Adam restarts at each stage, while frozen observation normalization and learned weights are inherited. The selected intermediate predates the full hard-stage training budget. [Mild stage](runs/paired_mild_525s_seed2.json), [full hard stage](runs/paired_hard_585s_seed2.json), [selected artifact](runs/paired_selected_545s_seed2.json).

**60 tests passed**, including held-out split separation, group/reset isolation, compiled mass/inertia changes and zero single-skill reference gradients on paired states. Native macOS viewing passed and CPU/MPS action differences were below 2.87e-6. Three videos total sixty seconds at 1280×720, 25 fps, 1×; all **1,500 frames** decoded. Encoded openings, stride and endings were inspected, with readable captions and synchronized follow/head/overview views. Cameras and contact dots are observer output. [Delivery checks](PAIRED_DAMAGE_DELIVERY.json), [time log](../TIME_LOG.md).
