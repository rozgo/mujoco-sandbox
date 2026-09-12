# Balancing on a moving support

Accepted on main and pinned by `adaptive-moving-v1`; [the release manifest](RELEASE.json) identifies the unchanged checkpoint and video. Development history remains on `feature/moving-supports`. The earlier static standing release is preserved by `adaptive-standing-v1`.

A free-body Go2 balances on a physically actuated deck that translates, yaws, heaves and rocks. The platform uses MuJoCo joint servos with finite mass and force limits. The robot's base is never repositioned during live simulation. One actor produces the twelve joint targets; absent joints remain masked in static damaged-body rehearsal.

The command “hold still” now refers to the support. Hold position and height use the platform frame. Velocity subtracts the actual platform velocity **at the robot**, including angular velocity crossed with distance from the platform origin. Foot slip subtracts the rigid motion of the support. Angular-rate reward uses relative rotation, while body tilt still uses gravity: rocking the deck does not instruct the dog to tip over with it. Static worlds keep their previous reward and observation paths.

The actor stays **86 → 128 ELU → 128 ELU → 12**; the separate training critic stays **90 → 128 ELU → 128 ELU → 1**. Thirteen existing balance channels are reinterpreted where needed; seven previously unused channels expose the actual platform linear/angular velocity in the robot frame and relative height. Platform pose/twist is ideal simulator sensing in this prototype. There is no prediction of future platform commands, no added physical IMU model, no camera input, and no policy switching. Existing body orientation, angular velocity, joint state, contact bits and downward rays remain. Support-frame estimation from real sensors is future work.

Contact sensing matches robot geometries against the complete support-environment subtree, including the moving deck and static floor. It excludes robot self contacts. MuJoCo frame sensors report current physical support state, not the commanded target. As in the existing environment, derived contact/frame sensors have the normal stepping pipeline timing (up to one 2 ms integration step behind integrated qpos). [MuJoCo contact and frame sensors](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-contact).

## Physical setup

| Component | Values | Rationale |
| --- | --- | --- |
| Robot | Pinned Menagerie Go2, original masses/inertias/joint limits | Same robot as accepted standing release |
| Robot actuation | Position servos kp=20, kv=0.5; hip/thigh 23.7 N·m, calf 45.43 N·m | Preserve original physical force limits |
| Deck | 1.6 × 1.2 × 0.1 m, 35 kg, diagonal inertia 4.23/7.50/11.67 kg·m² | Uniform-box inertia; ample foot clearance |
| Deck reference | Center z=0.45 m; top z=0.50 m | Space for finite heave and tilt above ground |
| Six axes | x/y ±0.55 m; z ±0.20 m; yaw ±1.4 rad; pitch/roll ±0.45 rad | Mechanical travel envelope |
| Deck slides | kp=30,000 N/m, kv=1,800 N·s/m, ±3,000 N | Bounded industrial motion-table approximation; small loaded sag |
| Deck rotation | kp=6,000 N·m/rad, kv=350 N·m·s/rad, ±1,000 N·m | Bounded gimbal approximation |
| Contact | Both feet/deck sliding friction 0.8; condim=3; pyramidal cone | Preserve selected standing contact configuration |
| Solver | implicitfast, Newton, 30 iterations, 2 ms physics / 20 ms control | Same standing backend recipe |

The deck is a simplified six-joint motion table, not a mechanically modeled Stewart linkage. No gravity compensation is added: the servos carry both platform and robot weight, producing measurable sag. The ground fixture is physical; deck grid lines are observer-only paint.

Nominal presets (training and evaluation randomize amplitude ×0.75–1.05, frequency ×0.85–1.15 and phase using recorded seeds):

| Preset | Amplitudes | Frequencies |
| --- | --- | --- |
| Translate | x 0.30 m; y 0.18 m | 0.20 / 0.27 Hz |
| Yaw | 0.65 rad (37°) | 0.17 Hz |
| Heave | z 0.10 m | 0.40 Hz |
| Rock | pitch 0.18 rad; roll 0.14 rad | 0.24 / 0.31 Hz |
| Combined | x/y/z 0.22/0.14/0.07 m; yaw/pitch/roll 0.45/0.14/0.11 rad | 0.19/0.23/0.35/0.14/0.21/0.28 Hz |

A 1.5-second raised-cosine startup provides continuous zero initial position and velocity, including randomized phase. Sinusoidal targets are held for each control interval; actual motion comes from dynamics.

## Learning and interpretation

MuJoCo Warp runs physics on the RTX 4090; PyTorch CUDA runs PPO. Observation and reward assembly remains NumPy on the host. Batch size 4,096, rollout 24 steps, four PPO epochs, minibatches 3,072, learning rate 1e-4. Half the worlds are healthy dogs on moving decks. The other half rehearse static tasks, including the nine robot bodies. Damaged moving supports are transfer tests, not part of training.

The initial round included nine-body flat training and five healthy terrain cases. It improved moving development checks but lost upright behavior in two omitted difficult static conditions. The corrective round includes every original static surface plus a training-only frozen-standing rehearsal loss. The existing walking rehearsal remains. Neither teacher is used at deployment.

Three matched comparisons distinguish coordinate changes from RL: the frozen standing actor with original inputs, the same weights with relative inputs, and newly trained weights with relative inputs. Original physical trajectories and failures are retained. Passing is stricter than staying upright; thresholds were saved in [the brief](BRIEF.md) before tuning.

## Measured results

Selected checkpoint: [round 2](../../../assets/locomotion/checkpoints/moving/round2_60s_seed22.pt), SHA-256 `5c349861f504bdb43e29e3a4351998cc22ed50bacfde86d3772f3684bc8b7879`. Selection was recorded before held-out seed 9407 and video seed 9411.

Two rounds used **118.554 seconds of new GPU learning** and **5,505,024 transitions**. The policy inherited 2,459.460 seconds of earlier walking/standing learning, so its full selected lineage is **2,578.014 seconds (42 min 58 s)**. This is a fast extension of an existing policy, not learning a dog from scratch in two minutes. Setup/compilation/rehearsal collection took 13.740 and 21.713 seconds separately; short CPU interface checks are excluded from the training claim.

Development retention: all **144/144 static standing/transition trials stay upright**; strict passes **100/144** versus **98/144** for the accepted parent on the same seed. All original **36/36 walking tasks and every gait-quality gate pass**, including individual foot lift, visible steps, stride/stance, speed and body motion. The original failed static conditions remain failures.

Both CPU and Warp held-out moving tests improve from **16/24 strict passes** with the frozen parent to **22/24** after training; both remain **24/24 upright**. Merely changing input coordinates stays at **16/24**. The 24 checks include four stationary-deck controls. The two trained strict failures on each backend are brief unintended support in combined motion; torque caps hold and sampled penetration stays below 8 mm.

**Damaged moving transfer is not reliable.** The untrained whole-FR and whole-RR removal probes on combined motion pass **0/8** strict checks; only **6/8** survive the complete hold on each backend. These cases are not part of moving training. Do not infer arbitrary damaged-platform robustness from retained damaged walking and standing.

## Video

[Watch the 64-second comparison](../../../previews/locomotion/moving/moving_supports_v1.mp4). Five ten-second motion chapters compare all three methods. A labeled ten-second replay shows the trained combined-motion trajectory through following, overhead and head cameras; a four-second results card closes the film. Playback is 1×. The capture uses **MuJoCo Warp physics on NVIDIA**, with a CPU actor for four-world evaluation batches. PPO training used the CUDA actor/learner. Rendering is separate Linux EGL rendering of the recorded states.

Four video-seed trials per motion/method are evaluated: frozen original **15/20**, relative inputs only **14/20**, trained **20/20** strict passes; all 60 trials remain upright. These video-seed numbers do not replace the 22/24 trained holdout result. Predetermined trial 0 is shown; all four results are retained. This is a separate seed from model selection and final holdout. The reproduction/replay report preserves every captured model and trajectory hash. Camera pixels are not actor inputs.

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked --extra warp adaptive-moving record --checkpoint assets/locomotion/checkpoints/moving/round2_60s_seed22.pt --output previews/locomotion/moving/my_reproduction.mp4 --physics-backend warp
```

On Linux headless systems, prefix the command with `MUJOCO_GL=egl`. Omit `--extra warp` and select `--physics-backend mjbatch` for CPU capture on macOS. For presentation-only changes, `--replay-from previews/locomotion/moving/moving_supports_v1.json` verifies and uses cached trajectories/model binaries on the original capture host without simulating again.

## Run

From repository root:

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-moving preview
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-moving view --checkpoint assets/locomotion/checkpoints/moving/round2_60s_seed22.pt --motion combined
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked --extra warp adaptive-moving train --output outputs/locomotion/moving/new_round --seconds 60
```

The viewer uses the native macOS mjpython launcher. CPU physics works on macOS; CUDA/Warp is optional on the NVIDIA host. The training command above targets NVIDIA; use `--physics-backend mjbatch --device cpu --num-envs 64 --max-iterations 1` for a bounded CPU interface check. Checkpoint, preview and video files use Git LFS; raw trajectories, compiled models, rehearsal datasets and logs stay in ignored outputs/build directories. Both machines synchronize via commits and LFS.

This prototype assumes one known rigid support per moving world. It does not infer which moving surface a foot has reached, transition between independently moving platforms, or estimate platform motion from cameras. Static damage variants remain physically distinct bodies; moving training currently uses the healthy robot only.

Validation: **78 Mac tests passed / 25 CUDA checks skipped; 103 NVIDIA tests passed**. Both frozen and selected policies pass the native Mac viewer smoke. All 1,600 video frames decode at 1920×1080 / 25 fps; chapter, contact, camera and result-card samples were inspected. Capture/save **50.806 s**, render/encode **40.214 s**. These are separate from learning time. See [validation](VALIDATION.json), [selection](SELECTION.json), CPU/Warp holdout folders, and both preserved training-round reports.
