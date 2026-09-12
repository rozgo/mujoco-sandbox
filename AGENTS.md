# Building MuJoCo simulations in this repository

These instructions apply throughout this repository. Follow explicit user instructions when they differ from this guide. Preserve working demos while adding new experiences.

## Start with the task and a visible scene

- Read the relevant demo documentation and inspect the existing code before changing it. Check `git status` and preserve unrelated user work.
- For a new experience, save the user's brief in `docs/<demo>/INITIAL_PROMPT.md` or `BRIEF.md`. Record assumptions, required actions, cameras and measurable completion criteria.
- Save a public-safe task summary when a request includes private context. Never commit client identifiers, private project or repository names, local paths to private sources, or copied private-source provenance. Check documentation, prompts, logs, source comments, and media metadata before committing or publishing.
- Build and compile the static scene first. Render an overview and close views of the mechanism, object interactions and clearance; inspect the actual images and show a preview before adding movement.
- Respect a user-requested scene review. Existing approval and authorization persist; do not add repeated confirmation steps for routine implementation, testing or recording.
- Use a separate package, CLI and output directory for a new demo. Use a named preset and separate artifacts for variants, keeping the approved original available. The wind `standard` and `aggressive` profiles are the existing example.
- Track requested work in `docs/TIME_LOG.md` using actual clock readings. Separate simulation time, training time, rendering time and elapsed project time. Record follow-ups separately; do not invent an earlier start time or call wall time compute time.

## Repository map and commands

| Demo | Implementation | CLI | Read first |
| --- | --- | --- | --- |
| Hexapod transfer | `src/sixlegs/` | `sixlegs` | [Run guide](docs/hexapod/README.md), [design](docs/DESIGN.md), [validation](docs/VALIDATION.md) |
| RC rovers and communications | `src/sixlegs/rovers/` | `rover-comms` | [Rover documentation](docs/rovers/README.md) |
| Amphibious attachments | `src/sixlegs/amphibious/` | `amphibious` | [Amphibious documentation](docs/amphibious/README.md) |
| Neural wind and payload delivery | `src/sixlegs/wind/` | `wind-demo` | [Wind documentation](docs/wind/README.md), [aggressive variant](docs/wind/aggressive/README.md) |
| Adaptive dog RL | `experiments/adaptive_locomotion/` | `adaptive-dog demo` | [Official v1 baseline](docs/locomotion/OFFICIAL_V1.md), [isolated uv commands and history](docs/locomotion/README.md) |

Use `uv`, `pyproject.toml` and `uv.lock` for Python dependencies. Keep optional stacks in extras. Avoid global Python installs and unrelated dependency upgrades.

```sh
git lfs install
git lfs pull
uv sync --locked
uv run --locked sixlegs view
uv run --locked amphibious view --speed 1
uv run --locked --extra reticulum rover-comms view --reticulum --case degraded
uv run --locked --extra wind wind-demo view
uv run --locked --extra wind wind-demo view --profile aggressive
```

Most CLIs expose `preview`, `view`, `run` and `record`; the hexapod uses `render` for static previews. Check `--help` rather than assuming identical flags. Use `--static` for frozen inspection and `--seconds` for a bounded viewer smoke test.

## Model physical mechanisms explicitly

- Use SI units. Document world axes, dimensions and frame conventions. Prefer simple, solid geometry with readable silhouettes before adding decorative detail.
- Record component masses, inertias, joint travel, actuator limits, damping, friction and timestep, with a rationale. Distinguish vendor values from illustrative simulation choices. Sanity-check total mass and available support force or torque against gravity and leverage.
- Keep moving bases and manipulated objects physically free unless the mechanism requires a real joint or constraint. Represent gripper linkages, cable tension, suspension and attachments with their corresponding mechanics.
- Preserve pinned Menagerie source assets, licenses and hash manifests. Apply mounting changes and namespacing in scene composition. A vendor model does not include a vendor gait, calibrated controller or capability unless those are actually supplied.
- Plan clearance for the whole moving assembly: feet, wheels at full steering lock, wrist housings, grippers, floats and carried objects. Check both the initial pose and representative motion envelopes. Sampled clearance is not proof over every possible configuration.
- Distinguish visual geometry from collision geometry deliberately. Do not disable meaningful collisions or add invisible supports to make a task pass. Document necessary adjacent-link exclusions.
- Enforce physical actuator force limits as well as command ranges. A position target limit is not a torque limit; transmissions affect how actuator force maps to joint torque or fingertip force. Measure actual saturation during runs. See the [MuJoCo modeling guide](https://mujoco.readthedocs.io/en/stable/modeling.html).

## Keep motion physical and control reproducible

- Live movement must result from `mj_step`, bounded actuator commands and documented external forces. Do not advance the task by writing base/object poses, resetting velocities, teleporting feet or welding a grasped object to a hand.
- State assignment is appropriate for initialization, explicit reset, scratch inverse-kinematics calculations and replay of recorded states. Keep those paths separate from the live controller.
- Apply modeled wind, water or other environmental loads at meaningful points, including their moments, using `mj_applyFT` or the appropriate applied-force arrays. Rebuild the total each step without accumulating stale forces or erasing another contributor. Check force direction, units and energy behavior.
- Use named body, joint, site, camera and actuator access where practical. `qpos` and `qvel` layouts differ for quaternion joints; do not assume their indices or lengths match. Copy MuJoCo array views when logging snapshots that must survive the next step.
- After assigning replay/reset poses, run `mj_forward` before querying derived geometry or rendering. For live feedback, account for when contacts and other derived quantities are computed; use a suitable callback or stepping arrangement when current-step values are required. See the [simulation loop documentation](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#simulation-loop).
- Keep physics, controller, sensing and rendering clocks explicit. Existing articulated demos use a 2 ms `implicitfast` step; this is a tested starting point, not a universal setting. Preserve the fixed physics timestep when changing playback speed.
- Start with a stable hold, then one motion primitive, then the full task. Anchor gait progress and transitions to measured state and contact. Bound tracking error and acceleration rather than allowing an open-loop reference to move arbitrarily far ahead of the robot.
- Define transitions by physical evidence where it matters: opposing finger contacts before lifting, support contact and low object speed before release, immersion before water thrust, and recovered foot support before walking out of water.
- Use the same stepping and control implementation for headless runs, tests and the live viewer. Describe a scripted controller, fixed route, ideal sensor or learned policy accurately.

## External physics, learning and communications

- Keep subsystem boundaries explicit. MuJoCo supplies rigid-body dynamics and contact; fluid and packet models supply their own behavior. Record assumptions about coupling, calibration and observations.
- For learned forecasts, use an independent numerical reference for training labels and evaluation truth. Within a comparison, keep actual wind, initial conditions, route, controller gains and actuator limits matched; change only the intended experimental variable.
- Separate training, model-selection, development and evaluation seeds. Select demonstration cases before inspecting final evaluation results. Save training configuration, losses, checkpoint hashes and evaluation settings.
- An FNO predicts field evolution; PINO here is an FNO trained with an additional PDE-residual loss. The physics penalty is a soft constraint, not a guarantee of exact physical consistency. Do not describe either as a trained flight policy or claim PINO outperforms FNO without evidence.
- Check forecast causality: future reference samples must not influence the prediction. Keep exact-future diagnostics separate from deployable inputs. If forecasts are cached, disclose that and invalidate caches when model hashes, resolution, duration, horizon or physical scaling change.
- Test resolution transfer and numerical agreement on the devices used. Running the same weights on a finer grid does not establish new fine-scale accuracy. Benchmark live inference separately from cache playback.
- In communications demos, keep observer-wide state out of each agent's local observations. Only delivered packets may change remote beliefs. Distinguish a simulated radio channel from the real protocol stack and identify which stack features are exercised.

## Test the claim, not just the code path

- Define success before tuning: measurable progress, physical support/release, destination tolerance, stability and prohibited contacts. Reaching the last animation phase is insufficient.
- For locomotion, enumerate the allowed load-bearing collision surfaces for each physical body: intact feet or explicitly modeled distal stumps. A missing foot does not make the whole remaining leg an allowed support; motor weakness alone does not change the allowed surfaces. Check reaction forces at every physics substep during acceptance evaluation. Foot contacts plus no trunk contact can still miss a knee-supported gait. Keep progress scores separate from support-valid completion.
- When improving gait quality, measure each leg's stance/swing timing and support loads as well as stride length, speed, slip and body motion. Equal average strides can hide a limp. Preserve failed reward-weight trials; established formulas still need local tuning. Report selected-checkpoint training ancestry separately from the full compute spent on unsuccessful or later updates.
- Check scene structure, intended initial contacts, relevant clearance, static stability and actuator limits first. Then test the affected mechanism and at least one complete physical mission for a dynamics change.
- Detect nonfinite state, MuJoCo warnings, excessive penetration, lost payloads and departure from a declared operating envelope. Fix geometry, units, frame errors, contact settings or control causes before relaxing thresholds. Do not hide instability with repeated resets or oversized forces.
- Separate physical task failure from numerical failure. Save failure reasons, partial trajectories and elapsed simulation time. Evaluation should retain failed cases and report unsuccessful comparisons with an appropriate exit status.
- Match metric windows and denominators. Do not count an early-terminated run as zero full-task error. Label conditional metrics such as RMSE on complete paired windows, and report success rates across all attempted cases separately.
- For comparisons, run multiple predetermined seeds and retain inconvenient outcomes. A more dramatic camera or stronger scenario does not establish a better model. Report measured improvements without implying general robustness or causation from unmatched experiments.
- Run focused checks first; run the relevant complete suite before finishing a substantial change. Report skipped tests and missing optional dependencies or caches. Do not rerun expensive simulations for a documentation-only or simple video edit unless its content changes a simulation claim.

```sh
# Complete suite with both optional stacks enabled.
uv run --locked --extra wind --extra reticulum pytest -q

# Wind validation and device agreement when those components change.
uv run --locked --extra wind wind-demo validate
uv run --locked --extra wind python scripts/check_wind_backend.py
```

## macOS and GPU portability

- Keep ordinary MuJoCo demos usable on Apple Silicon with CPU physics. Standard MuJoCo, MuJoCo Warp and PyTorch CUDA/Metal are different backends; state which actually ran. A GPU used for training does not make the rigid-body simulation GPU-based.
- Reuse the existing native macOS launchers. They invoke `mjpython` and pass uv's Python library directory through a subprocess environment. Do not patch system libraries or require global shell changes. The passive viewer's macOS requirement is documented in the [MuJoCo Python guide](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer).
- Use the supported rendering backend for the machine; headless Linux EGL settings must not be forced on macOS. Smoke-test the actual native viewer as well as headless execution when changing scene or viewer behavior.
- Prefer CPU/Metal fallbacks for optional learning workloads. Use a remote NVIDIA machine when authorized and useful, checking checkpoint transfer hashes and device agreement. Keep credentials and machine-specific connection details out of Git.
- Preserve the pinned NeuralOperator revision until a replacement passes the backend checks. This repo previously hit a CUDA inverse-FFT discrepancy; matching shapes and a successful training run did not establish cross-device correctness.

## Cameras and videos are part of the deliverable

- Provide an overview, a useful following/detail view and task-relevant mounted cameras: head/wrists for manipulation, front views for rovers, and aircraft/payload views for delivery. Render each camera and check framing and occlusion during the task.
- State whether camera pixels are controller inputs or observer output. A visible camera feed does not imply computer vision or autonomy.
- Record timestamps, poses, velocities, controls, measured forces and task events before producing the final video. Include seeds, configuration and scene/checkpoint hashes in run provenance. Re-render a validated run for camera or layout changes; store changing cable/attachment/release state needed to reconstruct it.
- Synchronize all panels to the same simulation timestamp. Default new videos to real-time motion and show the playback multiplier whenever it differs. Use actual states for trails and annotations; do not exaggerate displacement independently of physics.
- Make comparisons visually legible with matched viewpoints, clearly identified methods and useful quantities. Add stronger dynamics through a documented physical preset, with the same conditions for both methods.
- Start moving promptly. Avoid repeated opening frames or unexplained frozen holds that look like a stalled simulation. Cut directly to a clearly identified results card after the complete action; preserve existing approved edits unless asked to change them.
- Inspect encoded frames at the opening, strongest interaction, camera transitions, release and final results. Decode the entire MP4 and verify duration, dimensions, frame rate and frame count. Use the bundled `imageio-ffmpeg`; a new system FFmpeg install is normally unnecessary.
- Include a clickable video path and an exact viewer command in the handoff so the user can watch immediately. Retain previous versions when requested.
- Generative video styling uses the isolated `experiments/video_styling` project and [its guide](docs/video_styling/README.md). Keep original simulation footage and hashes, label generated presentation, and review motion/contact changes in a timestamp-matched comparison. Generated pixels cannot establish physics or controller results. Keep API credentials in ignored local environment files and resume saved provider task IDs instead of retrying billable submissions.

## Artifacts, documentation and Git

- Adaptive dog v1 is the user-accepted baseline on `main`, pinned by the `adaptive-dog-v1` tag and `docs/locomotion/OFFICIAL_V1.json`. Preserve its checkpoint/video and documented limitations. Put later learning trials on a new branch and use new artifact names; do not overwrite v1 or silently reinterpret its recorded gates.
- The accepted static-standing release is pinned by `adaptive-standing-v1` and `docs/locomotion/standing/RELEASE.json`. Preserve its checkpoint, v3 video and original failed gates. The accepted moving-support release is pinned by `adaptive-moving-v1` and `docs/locomotion/moving/RELEASE.json`. Preserve its checkpoint, comparison video and documented failed transfer tests; subsequent experiments use separate artifacts and branches.
- The shared origin is `git@github.com:rozgo/mujoco-sandbox.git`. Synchronize Mac/GPU work through commits and Git LFS; follow [the multi-machine workflow](docs/DEVELOPMENT.md). Pull with `--ff-only` before starting work, preserve local changes, and record the source commit for training. Use separate branches/worktrees for concurrent jobs rather than changing a running job's checkout.
- Keep reproducible source in `src/` and scripts, authoritative assets/checkpoints in `assets/`, scenario documentation and concise result reports in `docs/`, and selected review images/videos in `previews/`.
- Keep generated MJCF, datasets, weather caches, trajectories, scratch renders and logs in ignored `build/` or `outputs/`. Do not commit virtual environments, secrets or temporary machine configuration.
- Use the existing `.gitattributes` LFS patterns for meshes, images, MP4, checkpoints and NPZ files. Update LFS patterns when introducing another large binary format. Verify binary availability and `git lfs fsck` at a completed asset/video checkpoint.
- Update run instructions, assumptions, provenance and measured results with the implementation. Preserve third-party attribution. Cite original papers or official documentation for external technical claims; separate those results from this repo's measurements.
- Keep public explanations concise and accurate: distinguish learned prediction from programmed control, sensor rendering from perception, physical simulation from replay, and elapsed development time from training time. Do not infer the acting model's version from a user's draft wording.
- Review `git diff --check` and the staged changes, then commit completed work when requested or already authorized. Keep checkpoints understandable and the working tree clean without discarding user changes. Publishing or messaging others requires its own authorization.
