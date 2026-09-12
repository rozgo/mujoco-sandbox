# Balancing on a moving support

Experimental branch `feature/moving-supports`. The accepted static standing release remains unchanged on main, tagged `adaptive-standing-v1`.

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

## Run

From repository root:

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-moving preview
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-moving view --checkpoint assets/locomotion/checkpoints/moving/round2_60s_seed22.pt --motion combined
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked --extra warp adaptive-moving train --output outputs/locomotion/moving/new_round --seconds 60
```

The viewer uses the native macOS mjpython launcher. CPU physics works on macOS; CUDA/Warp is optional on the NVIDIA host. The training command above targets NVIDIA; use `--physics-backend mjbatch --device cpu --num-envs 64 --max-iterations 1` for a bounded CPU interface check. Checkpoint, preview and video files use Git LFS; raw trajectories, compiled models, rehearsal datasets and logs stay in ignored outputs/build directories. Both machines synchronize via commits and LFS.

This prototype assumes one known rigid support per moving world. It does not infer which moving surface a foot has reached, transition between independently moving platforms, or estimate platform motion from cameras. Static damage variants remain physically distinct bodies; moving training currently uses the healthy robot only.
