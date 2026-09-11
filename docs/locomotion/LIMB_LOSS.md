# Complete limb-loss experiment

Started **2026-09-10 23:01:09 UTC**. User approved retiring shortened-leg training and focusing on complete lower-leg removal and whole-leg removal, with one deployed policy. Preserve the earlier experiments, videos and `main`; work on `experiment/adaptive-dog`.

## Result

**[Watch all nine conditions together in 4K](../../previews/locomotion/limb_loss_all_cases.mp4)** · [Encoded preview](../../previews/locomotion/limb_loss_all_cases.png) · [Static bodies](../../previews/locomotion/limb_loss_static.png)

One selected reactive policy completes **254/256 removal trials**, versus 0/256 for the healthy parent, and retains **32/32 healthy completions** and all healthy-gait gates. All 288 selected-policy trials survive, cross 5 m and use only allowed support; two whole-front-right trials cross too late to satisfy the full one-second goal-window requirement before twelve seconds. They remain failures in the timed task score. This is flat-ground walking with one segment or one leg already absent at episode start, not dynamic severing, multiple-leg loss or parkour.

| Condition, 32 fresh initial states each | Healthy parent | Selected single policy |
| --- | ---: | ---: |
| Healthy | 32/32 | 32/32 |
| Entire lower leg removed: FL | 0/32 | 32/32 |
| Entire lower leg removed: FR | 0/32 | 32/32 |
| Entire lower leg removed: RL | 0/32 | 32/32 |
| Entire lower leg removed: RR | 0/32 | 32/32 |
| Entire leg removed: FL | 0/32 | 32/32 |
| Entire leg removed: FR | 0/32 | **30/32** |
| Entire leg removed: RL | 0/32 | 32/32 |
| Entire leg removed: RR | 0/32 | 32/32 |

Healthy mean stride changes **31.30 → 33.09 cm**, speed **0.609 → 0.635 m/s**, left/right duty-factor gap **8.46 → 5.23 percentage points**, and body-height standard deviation **7.84 → 7.40 mm**. The missing-limb gaits are free to be asymmetric. The controller receives joint/IMU/range feedback and encoder-validity bits; scripted lane commands supply direction. No reference policies, switching or online weight updates occur during deployment.

The task spent **597.233 seconds (9 min 57 s)** on five new training runs, totaling **11,968,512 transitions**. CPU MuJoCo/mjbatch physics and MPS learning ran on the Mac; no new RTX training was required. The selected policy is consolidation iteration 50: **764.683 seconds (12 min 45 s)** and **16,625,664 transitions** across its complete ancestry, including the 403.329-second healthy parent. Its new ancestry is 361.354 seconds; later and unsuccessful updates remain counted in the larger 597.233-second experiment cost. Consolidation iteration 150 tied at 64/64 development removals; later candidates regressed, so the earlier checkpoint was frozen before final evaluation.

Additional checks: **72/72** completions at a 1 ms physics timestep across all nine conditions; **62 tests passed**; native Mac viewer passed; CPU/MPS action difference below 7.16e-7. All recorded actuator torques stay within their original caps, and maximum sampled penetration is **5.04 mm**. The video contains nine fixed successful demonstrations, twelve seconds at 1×, 3840×2160 and 25 fps. All 300 encoded frames decoded; opening, stride and final frames were inspected. Capture took 4.193 seconds and rendering/export 39.979 seconds, excluding renderer/context setup.

[Final outcomes and retained failures](LIMB_LOSS_VALIDATION.json) · [Matched healthy-parent evaluation](LIMB_LOSS_REFERENCE.json) · [Selection, checkpoint hashes and all training artifacts](LIMB_LOSS_SELECTION.json) · [Physics/backend checks and ancestry](LIMB_LOSS_PHYSICS.json) · [Video delivery checks](LIMB_LOSS_DELIVERY.json)

## Bodies and support

Nine target conditions: healthy; four positions with the entire calf and foot removed; four positions with the entire hip/thigh/calf chain removed. Remove the physical subtrees, masses, inertia, collision surfaces and actuators. A lower-leg removal leaves an explicitly modeled 15 mm upper-leg stump as an allowed contact; a whole-leg removal adds no support surface. Only remaining feet and designated upper-leg stumps may bear load. Strict evaluation checks unintended contacts above 1 N at every 2 ms physics step.

The intact body and retained joint masses, limits, friction and torque caps are unchanged. Removed actuator channels are masked by actual topology. Actor observations retain the 66-channel interface, including encoder-validity bits: these reveal which joints are available. No additional damage label or privileged geometry enters the reactive actor. Camera views remain observer output. Removal happens before each episode, not by cutting a running model.

## Bounded first attempt

Start from `healthy_balanced_405s_seed2.pt`, actual ancestry 403.329 seconds. Use 512 CPU MuJoCo/mjbatch environments with MPS network updates. One shared reactive policy is fine-tuned sequentially; the healthy reference is training-only. No shortened bodies or motor faults occur in these stages.

1. Front-right lower leg removed, 50% healthy / 50% removed: initial 60-second round.
2. All four lower-leg removals, 25% healthy / 75% evenly removed: initial 90-second round.
3. All nine bodies, 25% healthy / 25% lower removals / 50% whole removals: initial 150-second round.

These are intended allocations, subject to measured progress. First decision within 300 new training seconds; extend toward 600 only if the runs show useful traction. Ancestry and all failed/later training updates are accounted separately. The CLI lineage allowance includes the 403-second parent and is not a claim of fresh training time.

Retain healthy-only gait timing, posture and frozen-reference objectives. Damaged bodies retain velocity, uprightness, support, effort and smoothness rewards with freedom to change stance. Keep existing actuator limits and physical contact intact. Freeze development seed 9141, final seed 20260917, demonstration seed 9143. Select on development results before inspecting final outcomes.

Acceptance: survive twelve seconds, travel 5 m, remain controlled in the goal lane for one second, and use only allowed support throughout. First evaluate eight initial conditions per target; final report uses 32 per target and retains all failures. Healthy gait gates remain 28–35 cm stride, speed within 10% of 0.6092 m/s, duty-factor gap at most 12 percentage points and body-height standard deviation at most 11 mm. Failed missing-limb attempts are progress evidence only, never successful walking. Flat-ground single removals are the scope; multiple missing limbs and parkour remain later work.

## First decision

The 59.266-second front-right-only stage preserved all healthy gates but completed 0/8 missing-lower-leg trials; mean forward travel before failure was 0.216 m. The healthy parent completed 0/64 across the eight removal cases. Before expanding to all lower-leg positions, disable the long-stride preference on damaged bodies, increase the fall penalty from 2 to 10, and try a 0.0006 learning rate. These are measured local reward adjustments, not a claim of a new RL algorithm. Healthy stride/reference objectives remain active. The failed first checkpoint and report are retained.

The next lower-leg stage passed 8/8 each for rear-left and rear-right complete calf removals, and retained all healthy gait gates. Front removals still passed 0/8 each. Proceed to the planned 150-second whole-leg stage with all nine bodies. Healthy pretraining leaves validity-channel variance near zero and corresponding input weights untrained: an absent joint then creates a clipped 10-standard-deviation input. Zero-initialize only the newly used hip/thigh validity columns before this stage; exact action equality was checked on healthy and calf-only inputs. The calf validity weights already trained in the prior stage are preserved. All these columns remain learnable through PPO.

After 297.967 seconds of total new training, whole-stage iterations 100, 200 and final all passed 8/8 in six removal conditions and all healthy gates. Both front-right conditions remained stationary with valid support, completing 0/8 each. Select the earliest tied candidate, iteration 100, for a 150-second focused extension. Allocate 128 healthy environments, 144 to each failed front-right case and 16 to each other removal. Reset exploration standard deviation to 0.3, and increase the damaged-only forward-progress coefficient from 0.3 to 1.0. Retain the healthy reference and all previous bodies. This extension is justified by six successful removal conditions and a specific stationary local optimum in the remaining two; it stays within the user-approved ten-minute ceiling for new training.

The focused stage spent 149.545 seconds. Iteration 200 learned both front-right conditions (16/16) but regressed on lower-left-front and whole-left-rear removal. The final also lost whole-right-rear survival. Healthy gait remained within its gates. Use the remaining 150-second allowance to consolidate: 128 healthy environments and 48 per removal, ordinary 0.3 progress coefficient, learning rate 0.0003. Resume focused iteration 200. Frozen whole-stage iteration 100 supplies reference actions for the other six removals; frozen focused iteration 200 supplies the two front-right targets. The existing healthy reference remains. Damaged-reference action loss weight is 4. All reference routing is training-only, using compiled body context; the resulting deployed actor is one reactive network. Both reference lineages are already included in the chosen parent's ancestry. Stop new training after this round, regardless of outcome.

## Reproduce and view

From the repository root, enter the isolated package:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv sync --locked
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_selected_seed2.pt \
  --case whole_fr
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog grid \
  --family limb_loss --seed 9143 \
  --checkpoint ../../assets/locomotion/checkpoints/limb_selected_seed2.pt \
  --output ../../previews/locomotion/limb_loss_all_cases.mp4
```

Case names are `healthy`, `lower_fl`, `lower_fr`, `lower_rl`, `lower_rr`, `whole_fl`, `whole_fr`, `whole_rl`, `whole_rr`. The native viewer runs the actual policy/physics loop. The grid first saves physical rollouts, then replays their recorded states with synchronized following cameras. It labels failed or incomplete attempts, and does not reset them.

Training uses the existing `adaptive-dog train` command with `--limb-stage one|lower|whole|front|consolidate`. Full per-stage configurations, source commits, seeds, parent dependencies and measured times are retained in the run reports. `--neutralize-validity hips_thighs` is an initialization step for the first whole-leg stage; `--initial-std 0.3` is the focused-stage exploration reset. Reference checkpoints and reference loss flags are training inputs only and are unnecessary for viewing or recording.
