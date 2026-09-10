# Adaptive dog: minute-scale RL pilot

Implemented on **`experiment/adaptive-dog`**, with `main` preserved. One learned controller moves a Go2-derived robot with shortened calves and weakened motors over low steps. A 329-second reactive policy is the strongest candidate. Adding a sensor-history estimator did not improve this pilot, and extending training to 599 seconds introduced forgetting.

**[Watch the 48-second three-camera demonstration](../../previews/locomotion/reactive_5m30s.mp4)** · [History versus reactive controller](../../previews/locomotion/history_vs_reactive.mp4) · [What the extra training did](../../previews/locomotion/extra_training.mp4)

[![Learned control with a shortened calf on low steps](../../previews/locomotion/frame-steps.png)](../../previews/locomotion/reactive_5m30s.mp4)

All clips play at **1×** from actual MuJoCo rollouts. The main video uses the same checkpoint for all four cases, with following, head and overview cameras. RGB is observer output, not policy input. The comparison clips retain incomplete trials. This is useful locomotion traction, **not general quadruped parkour or arbitrary-damage recovery**.

## Run on Mac or Linux

From the repository root:

```sh
git switch experiment/adaptive-dog
git lfs pull
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv sync --locked
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/blind_steps_330s_seed0.pt \
  --case short_steps
```

The viewer launches through `mjpython` automatically on macOS. Close its window to stop; `--seconds 5` gives a bounded smoke test. `blind` is the internal name for the reactive controller: it receives joint/IMU/range observations and its previous action, but no learned history encoder or private damage description. It is not blind to the terrain.

The isolated stack uses **uv 0.12.12, Python 3.14.7, MuJoCo 3.13.0, PyTorch 2.14.0 and NumPy 2.5.3**. Its committed lockfile leaves the existing demos' environments unchanged. The vendored [mjbatch code and provenance](../../experiments/adaptive_locomotion/third_party/mjbatch/PROVENANCE.md) retain the upstream Apache-2.0 license. A C++ compiler is needed for its first local build.

```sh
# Fresh short learning experiment; output and trajectories stay ignored.
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog train \
  --output ../../outputs/locomotion/my_run --seconds 60 --mode history \
  --bodies all --terrain flat --seed 2 --num-envs 512 --threads 16

# Reproduce a development evaluation or the main video.
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog evaluate \
  --checkpoint ../../assets/locomotion/checkpoints/blind_steps_330s_seed0.pt \
  --output ../../outputs/locomotion/my_evaluation.json --trials 32
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog record \
  --checkpoint ../../assets/locomotion/checkpoints/blind_steps_330s_seed0.pt \
  --output ../../outputs/locomotion/my_video.mp4
uv tool run --from uv==0.12.12 uv run --locked pytest -q
uv tool run --from uv==0.12.12 uv run --locked pytest -q third_party/mjbatch/tests
```

Use `--device cuda` on Linux for the learner, or `--device mps` on Apple Silicon. **Physics remains CPU MuJoCo in mjbatch**, not MuJoCo Warp. Multiple native simulation pools run concurrently across body variants. A GPU accelerates neural-network updates but cannot remove this implementation's CPU simulation bottleneck. This environment exposes a direct batched Python API, not a Gymnasium wrapper.

## What was actually learned

PPO learns twelve independent joint-position offsets through torque-limited native servos. There is **no gait clock, supplied walking policy, mirrored-action constraint, foot trajectory, live base pose correction or external propulsive force**. A scripted high-level lane follower uses ideal localization to request forward/lateral/yaw velocity; route selection is not learned.

The actor sees 66 channels: twelve joint angles, twelve joint velocities, gyro, an ideal IMU-style gravity direction, previous action, velocity command, twelve encoder-validity bits and nine body-mounted ranges. Native rays intersect scene geometry; observations receive ranges at 20 Hz, although MuJoCo computes them more often. Noisy sensing, independent encoder outages and latency are future work. Missing joints have masked semantic slots; a shortened member with a surviving joint keeps its encoder.

The reactive actor is a 128×128 MLP with a fixed context. The history variant adds a learned 64-wide estimator over 0.5 seconds of causal feedback, pooled into five chronological windows. Its training uses privileged geometry/strength labels as auxiliary supervision. Both have an asymmetric critic that can see exact body velocity, height and damage; those extra values do not enter the deployed actors. The Mac policies share a 59.81-second oracle-context pretraining stage; its time is included below. The independent GPU history run started from random weights without that stage.

## Measured results with corrected contacts

Completion means reaching 5 m within a 12-second trial and remaining controlled in the goal lane for one second. Failures remain in the denominator; no reset occurs inside a trial. The final cases and seed were frozen before the initial comparison. We then repeated that same evaluation after fixing contact stiffness, without changing either 329-second policy's weights.

| Frozen case, 32 initial conditions each | Reactive, 329.05 s | History, 329.35 s | Independent GPU history, 269.59 s |
| --- | ---: | ---: | ---: |
| Healthy | 32/32 | 32/32 | 32/32 |
| Unseen front-right calf at 62% length | 32/32 | 32/32 | 32/32 |
| Two left calves at 78% and 86% | 32/32 | 32/32 | 32/32 |
| Short rear-right calf plus rear-left hip dropping to 35% torque | 32/32 | 32/32 | 32/32 |
| New step layout, 3.5/5/4.5 cm, with two shortened calves | **32/32** | **0/32** | **14/32** |
| Rear-right calf and distal joint removed | 0/32 | 0/32 | 0/32 |

[Full paired results and confidence intervals](VALIDATION.json) · [Independent GPU results](gpu_firm_validation.json). A 32/32 result has a Wilson 95% interval of approximately 89–100%; this describes initial-condition sampling, not training-seed reliability. There is only one matched Mac seed. The GPU seed used different hardware and curriculum, so it is supporting traction, not a controlled replication or device benchmark.

The comparison is an **equal wall-time engineering test**, not a causal architecture ablation: the simpler actor collected more transitions, and the history curriculum used several resumed stages whereas the reactive fine-tune was continuous. Resuming restarts Adam; optimizer state is not saved. These differences prevent attributing the result solely to memory.

On the separate development suite, the selected reactive policy completed 32/32 low-step trials and 28/32 larger 6/9/6 cm step trials. Those larger steps are not a robust parkour result. See [all development outcomes](runs/blind_steps_330s_seed0_firm_development.json).

## Was ten minutes worth it?

The additional experiment resumed the 329-second reactive policy for 269.54 seconds, using healthy, shortened-front-left and absent-front-left-calf bodies on flat ground. **Total: 598.60 seconds**, including ancestry. Its training process had already loaded the earlier contact model when the geometry fix was made; its reported evaluation and comparison video use the corrected firm model.

In 32 development trials, missing-calf progress increased from **0.67 m to 3.92 m** on average, but **neither policy completed the 5 m task**. The longer-trained policy retained healthy and weak-motor completion, while losing all 32 two-shortened-calf completions and all 32 low-step completions. The inspected missing-calf trajectory also used trunk contact in 50 of 600 sampled frames. This is incomplete, partly body-supported locomotion, not successful three-leg parkour. [Longer-run outcomes](runs/blind_missing_600s_seed0_firm_development.json), [contact diagnostic](MISSING_CALF_CONTACTS.json).

Keep the shorter policy as the default. The next experiment should preserve varied bodies and terrain while gradually introducing severe missing segments, then test whether a useful history representation improves performance under genuinely ambiguous faults. Blindly adding training time is not the next step. Upper-leg removal, complete missing legs, one-/two-leg mobility, locks, delay, friction randomization, perception and learned navigation remain unimplemented. The broader three-seed/100-trial acceptance plan and off-policy learner comparison remain outstanding.

## Physical specification and verification

World axes are x forward, y left, z up, in SI units. Healthy mass is **15.206408 kg**, retaining unaffected Menagerie properties. Each 0.213 m calf is replaced by a declared primitive mass proxy: 70 g proximal housing, 151.352 g removable member and 20 g terminal surface at full length. The last two masses scale with remaining length; MuJoCo derives their combined inertia. Healthy terminal radius is 22 mm, shortened terminal radius 14 mm; these are illustrative contact surfaces, not a calibrated fracture model. Removing a calf removes its body, mass, collision geometry and distal joint, leaving a thigh-tip stump and eleven actuators. A thigh's retained vendor inertia already accounts for its support surface; no extra stump mass is added to that explicit inertia.

Hip limits are ±1.0472 rad; front thigh −1.5708…3.4907, rear thigh −0.5236…4.5379, calf −2.7227…−0.83776 rad. Native position servos use Kp=20 Nm/rad and Kd=0.5 Nm·s/rad, unit gearing, and actual torque caps of **23.7 Nm for hip/thigh and 45.43 Nm for calf**. Passive damping is 2 Nm·s/rad, armature 0.01 kg·m² and friction loss 0.2 Nm. Commands are nominal pose plus 0.65 times actions clipped to ±3; physical joint stops and actuator force limits remain active. Weakening reduces available torque; zero strength also removes active servo damping.

Physics uses a 2 ms `implicitfast` step, Newton solver with 30 iterations and pyramidal contacts; policy rate is 50 Hz. Full trunk, leg and stump contacts remain enabled, with MuJoCo's ordinary adjacent-body exclusions. Ground/healthy-terminal sliding friction is 0.8; shortened members use 0.6. Corrected contact parameters are `solref=".006 1"`, `solimp=".95 .99 .001"`, margin zero on every physical surface.

The first primitive implementation inherited a 20 ms contact constant and allowed approximately 22 mm sampled foot penetration in the selected rollout. That was corrected before final recording. The selected step rollout now has a sampled maximum of **7.95 mm**, no trunk-contact frames, and **16/16** step completions when rerun at 1 ms. Penetration is reconstructed every 20 ms, not bounded at every substep; these compliant contacts are still an approximation. [Contact report](CONTACTS.json). The earlier [soft-contact results](LEGACY_SOFT_VALIDATION.json) are retained and are not the final physics claim.

Nine project tests cover scalar/batch agreement, partial-body topology and inertia, sensor/private-context isolation, zero-strength actuation, history causality and foot-drop penetration. The 38 vendored mjbatch tests cover the native state/batch interface. MuJoCo warnings reject a run; actual measured actuator torque remains within its effective caps. Native macOS viewing and complete MP4 decoding are checked separately.

## Artifacts and time accounting

[Checkpoint directory](../../assets/locomotion/checkpoints/) · [Training configurations, progress and hashes](runs/) · [Static body manifests](../../previews/locomotion/static/bodies.json) · [Original brief](BRIEF.md) · [Original broader plan](ADAPTIVE_LOCOMOTION_PLAN.md) · [Time log](../TIME_LOG.md)

Checkpoints and curated media use Git LFS. Raw trajectories, generated models and scratch logs remain under ignored `outputs/locomotion/`. Training reports retain source commits, actual per-stage and cumulative time, seed, device, transition counts and checkpoint hashes. Curation only makes parent paths repository-relative; learned tensors are unchanged. Runs trained before the contact fix are explicitly labeled.

Training time excludes installation, model compilation/setup, evaluation, recording and agent development. Separate experiments are not secretly combined into one claimed five-minute run; their shared ancestry is counted once per deployed policy. The full implementation elapsed time is recorded independently in the project time log. No main-branch merge or real-hardware validation has been performed.
