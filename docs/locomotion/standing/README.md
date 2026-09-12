# Learning to stand on uneven supports

One shared policy walks, stops to balance, and walks again. It also holds standing
poses on uneven pads, slopes and steps, including missing support under a front
foot. This is a limited extension of the accepted walker, on
`feature/adaptive-standing`; it does not replace adaptive dog v1.

[Watch the 45-second video](../../../previews/locomotion/standing/adaptive_standing_v2.mp4) ·
[All evaluation results](RESULTS.json) · [Network, rewards and physics](IMPLEMENTATION.md) ·
[Original brief and gates](BRIEF.md)

The film shows the same weights in 17 predetermined cases, at real time:
walk–stand–walk with following/head/overhead views, six ordinary support scenes,
six aggressive challenges and four physically damaged bodies. Failures stay in
the film and its results card. Camera pixels are observer output.

## What learned

PPO fine-tuned the existing actor across 4,096 MuJoCo Warp worlds on an RTX 4090.
The controller receives a requested velocity: zero means hold, positive means
walk. The actor learns joint targets; torque-limited actuators and physical
contacts move the robot. A frozen walker and recorded walking examples guide
training so that learning to stand does not erase the gait. Neither is deployed.

The actor remains 86 → 128 → 128 → 12 with ELU hidden activations. The critic is
90 → 128 → 128 → 1. Standing adds four contact bits, four downward ray distances,
ideal body velocity and ideal hold-position error in formerly reserved inputs.
Those inputs are zeroed while walking. Existing body orientation/angular velocity
remain available; no additional IMU model was added. There is no camera perception,
live inverse kinematics, hidden support constraint or online weight update.

Seven sequential rounds used **536.734 seconds (8 min 57 s) of new training** and
**26,836,992 world/control transitions**. The selected weights inherit
**1,922.725 seconds** of prior walking training, for **2,459.460 seconds
(40 min 59 s) total ancestry**. These are learning-loop times, excluding setup,
rehearsal collection, tests, evaluation and recording. This was fine-tuning, not
training a dog from scratch in nine minutes. Failed intermediate runs are retained.

## Validation and limits

The checkpoint was selected on development seed 9301, then frozen before seed
9307 evaluation and seed 9311 video capture. Four reset trials are used per case.
The terrain geometry is familiar from training: this tests new initial
perturbations, not unseen-terrain generalization. There are no mass/strength
variations or external shove tests in this first version.

Standing acceptance requires full-trial survival, only allowed supports (1 N
threshold checked every physics step), mean idle speed ≤0.06 m/s, drift ≤0.15 m,
tilt ≤20°, respected torque caps and sampled penetration ≤8 mm. Missing-corner
cases additionally require the designated foot to remain unsupported for at least
95% of the measured hold. Settling takes two seconds; acceptance includes initial
support impacts. Penetration is sampled at 50 Hz, not guaranteed between samples.

CPU holdout: **48/80 healthy terrain/transition trials** and **46/64 damaged flat
standing/transition trials** pass every gate. Warp gives **48/80** and **48/64**,
respectively. All 144 trials per backend survive; survival alone is insufficient
for passing. Every trial respects torque caps within the numerical tolerance.
The largest sampled penetration is 10.318 mm on CPU and 9.492 mm on Warp, both
from damaged transitions and both retained as failures. Healthy flat
walk–stand–walk passes 4/4. Flat, ordinary pads, both 6° and 12° slopes, the 18° Y slope, high pads,
12 cm steps and both front missing-support cases pass 4/4 each. On both front
gaps, the designated foot remains unsupported for the entire measured hold.

The existing walking-retention evaluation separately passes **36/36 tasks**
across all nine bodies at 0.5 ms physics, including every original visible-step,
stride/stance, speed and body-motion retention gate. Standing uses 2 ms physics;
this is not a claim that every damaged stop is already reliable.

The rear-left gap often finds a fourth support on the adjacent pad and therefore
fails the deliberate foot-in-air criterion. The rear-right gap has unintended
support and excessive tilt. Extreme pads/steep slopes can settle beyond the hold
region; 20/28 cm steps can lean on unintended links or exceed the tilt limit.
Some front-damaged standing/stop trials still make unintended link contacts;
a few damaged transitions exceed the 8 mm sampled penetration target. All raw
failures are retained. Damaged bodies have been trained on flat ground only.

Reset-only inverse kinematics initializes a feasible pose and starts the gap-side
foot raised. The learned claim is maintaining balance from that pose, not
independently discovering how to lift a leg from arbitrary initial conditions.
The same actuator caps and body properties apply
([physical specification](PHYSICAL_SPEC.json)): healthy mass 15.206 kg / 12 actuators, lower-leg removal 14.965 kg / 11 actuators, whole-leg
removal 13.135 kg / 9 actuators. Hip/thigh torque caps are 23.7 Nm and calf caps
45.43 Nm; these inherited limits prevent the new task from gaining extra strength.
Actual limb removals remove geometry, mass and actuators, not merely a visual mesh.

## Run on Mac or Linux

From the repository root, after pulling Git LFS files:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv sync --locked
uv tool run --from uv==0.12.12 uv run --locked adaptive-balance view \
  --checkpoint ../../assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt \
  --surface gap_fr
```

Use `--surface pads_high`, `--surface slope_y_18`, or `--surface steps_12` for
other supports. `--surface flat --transition` shows walk–stand–walk.
`--body whole_rr --surface flat` selects a damaged body. The Mac launcher uses
MuJoCo's `mjpython`; no global Python/library changes are required. The native
viewer was smoke-tested with the selected checkpoint.

Reproduce the full capture (Linux NVIDIA; omit `--extra warp` and select
`--physics-backend mjbatch` for CPU physics):

```sh
uv tool run --from uv==0.12.12 uv run --locked --extra warp adaptive-balance record \
  --checkpoint ../../assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt \
  --physics-backend warp --seed 9311 \
  --output ../../previews/locomotion/standing/adaptive_standing_reproduction.mp4
```

A short new training round, writing separate output:

```sh
uv tool run --from uv==0.12.12 uv run --locked --extra warp adaptive-balance train \
  --resume ../../assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt \
  --output ../../outputs/locomotion/standing/my_trial \
  --seconds 60 --seed 15 --profile mixed --surfaces all \
  --physics-backend warp --device cuda --num-envs 4096 \
  --walking-replay-weight 15 --idle-support-weight 8 --idle-drift-weight 3 \
  --idle-episode-steps 500 --substep-support \
  --reason "New short standing experiment; preserve selected weights."
```

The final setup is MuJoCo 3.13.0, MuJoCo Warp 3.13.0, Warp 1.17.0,
PyTorch 2.14.0, Python 3.14.7 and uv 0.12.12. CPU physics remains available via
mjbatch. Observation/reward assembly stays on the host even when physics and PPO
run on CUDA. Camera rendering is separate from batched learning.

[Selected checkpoint hash](SELECTION.json), per-round `training.json` reports,
CPU/Warp holdouts, captured trajectory/model hashes and encoded-video QA preserve
the provenance. Full trajectories remain in ignored `outputs/`; the checkpoint,
video and preview use Git LFS. Historical reports retain the world-count reporting
limitation and one `source_dirty=true` run noted in the implementation guide.

Validation: **73 Mac tests passed, 24 CUDA-only tests skipped; all 97 passed on
the NVIDIA host**. Ruff lint/format checks pass across 86 Python files. Existing
Warp struct-deprecation and capsule–cylinder multicontact warnings remain
documented; no new numerical instability was hidden by resets.

The final video is a layout-only replay of the first physical capture. All 17
trajectory hashes and acceptance metrics are identical. To revise rendering
on the capture host, pass `record --replay-from` the original video JSON with
the same checkpoint, backend and seed; its ignored trajectory files must be
available. The command verifies both trajectory and compiled-model hashes and
fails on mismatches. It does not rerun the policy or physics. The first render
remains archived alongside v2.

Capture took **45.154 s** including setup and trajectory saving; the first render
took **33.059 s**. The final layout replay took **16.711 s** to load/verify/save
states and **33.637 s** to render/encode, with zero new physical simulation or
learning. All 1,125 frames decode at 1920×1080 / 25 fps; chapter boundaries,
key interactions and the results card were visually inspected. The video passes
10/17 task cases; all recorded torque caps hold and maximum sampled penetration
is **3.495 mm**. This video seed does not replace the wider holdout results above.

Verified implementation/results/video milestone: **2026-09-12 00:41:58 UTC**,
**1 h 23 min 27 s** after the goal start, before final documentation commit and
machine synchronization. See the [time log](../../TIME_LOG.md) for the sequence.
