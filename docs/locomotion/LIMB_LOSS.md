# Complete limb-loss experiment

Started **2026-09-10 23:01:09 UTC**. User approved retiring shortened-leg training and focusing on complete lower-leg removal and whole-leg removal, with one deployed policy. Preserve the earlier experiments, videos and `main`; work on `experiment/adaptive-dog`.

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
