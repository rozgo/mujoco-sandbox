# Time log

All local times use America/Los_Angeles (PDT, UTC−07:00).

| Milestone | Local time | Notes |
| --- | --- | --- |
| Approximate session start | 2026-09-07 20:01:31 | Workspace creation timestamp; proxy, not an exact first-message timestamp. |
| Git repository initialized | 2026-09-07 20:03:39 | Filesystem creation timestamp. |
| Explicit time tracking started | 2026-09-07 20:04:34 | Clock reading: 2026-09-08 03:04:34 UTC. |
| Static scene checkpoint complete | 2026-09-07 20:14:26 | All five tests pass; six renders inspected; native macOS viewer smoke test passes. |

Elapsed to scene checkpoint: **12 minutes 55 seconds** from the approximate session start, or **9 minutes 52 seconds** since explicit clock tracking began.

## Movement milestone

| Milestone | Local time | Notes |
| --- | --- | --- |
| Scene approved; movement work resumed | 2026-09-07 20:20:32 | User confirmed that the scene looked great. |
| Full transfer achieved | 2026-09-07 20:29:04 | First complete run: physical grasp, carry, release and success report. Timestamp is the next recorded clock reading, an upper bound. |
| Tests and recording inspected | 2026-09-07 20:36:09 | Seven tests pass; native viewer and multi-camera MP4 verified. |

The scene-review pause was approximately 6 minutes 6 seconds. Total elapsed time includes this user review pause; this is not a measurement of active compute time. Final repository-completion time is recorded below.


**Full demo implementation and validation complete: 2026-09-07 20:37:19 PDT** (2026-09-08 03:37:19 UTC; final clock reading before the completion commit).

- Approximate total elapsed from the initial workspace/session timestamp: **35 minutes 48 seconds**.
- Elapsed since explicit tracking started: **32 minutes 45 seconds**.
- Movement implementation and validation since scene approval: **16 minutes 47 seconds**.
- Approximate elapsed excluding the recorded scene-review pause: **29 minutes 42 seconds**. This still includes tools, tests, rendering and communication, not just compute.


## Full-length all-view video

- User requested a full video of the task with all views.
- Work started: **2026-09-07 20:41:49 PDT** (2026-09-08 03:41:49 UTC).
- Rendering the saved, validated physical trajectory at real time with six synchronized views.

- All-view video finished and verified: **2026-09-07 20:46:29 PDT**.
- Video follow-up elapsed: **4 minutes 40 seconds**.
- Total elapsed since the approximate initial session start, including review pauses and this follow-up: **44 minutes 58 seconds**.

## New goal: physical RC rovers and communications

- Goal created and implementation started: **2026-09-07 21:57:21 PDT** (2026-09-08 04:57:21 UTC).
- A separate six-rover inspection yard was built and previewed before motion control.
- Implemented contact-driven steering/suspension, independent local-evidence agents, packet-level LoRa, optional six-process real Reticulum integration, cameras, metrics and a full recording.
- Fixed steering clearance, local avoidance deadlock and marker overshoot during testing.
- Validation: **16 tests passed**, **18/18** paired 150-second rover missions completed, direct and Reticulum native macOS viewer smoke tests passed.
- The 78.07-second 1920 × 1080 video covers the complete 150-second degraded Reticulum run at 2× speed with seven simultaneous camera views and a three-second final hold. The complete H.264 stream decoded without errors; outage and final frames were visually inspected.
- Implementation and validation complete, immediately before the repository commit: **2026-09-07 22:33:11 PDT** (2026-09-08 05:33:11 UTC).
- Elapsed for this new goal: **35 minutes 50 seconds**. Includes development, experiments, tests, rendering and communication; the interval since the earlier hexapod goal is excluded.

## New goal: amphibious attachment demonstration

- Goal created and implementation started: **2026-09-09 12:50:04 PDT** (2026-09-09 19:50:04 UTC).
- Built and previewed a separate unarmed Go2 model with four leg-mounted flotation capsules and a basin.
- User removed the rough-terrain requirement during development; all rocks were removed and smooth, gentler banks retained.
- Corrected the floating posture to extend both front legs and their attached floats forward. This uses a custom scripted controller, not a vendor walking policy.
- Fixed soft foot contact, ramp-height disagreement, accumulated gait-reference drift, and incomplete traction recovery during the wet exit.
- Complete crossing validated twice, including **10.816 seconds of unsupported flotation** and four-foot dry completion. All **20 tests passed** (16 existing tests plus 4 amphibious tests).
- Captured and inspected the complete **68.48-second, 1920 × 1080, 25 fps** three-view video. Walking is replayed at 8× and attachment transitions at 2×. Full H.264 stream decoded without errors; native macOS viewer smoke test passed.
- Implementation, recording and validation complete immediately before the completion commit: **2026-09-09 13:30:09 PDT** (2026-09-09 20:30:09 UTC).
- Elapsed for this goal: **40 minutes 5 seconds**. Includes user steering, development, failed experiments, tests, rendering and communication; excludes the interval since previous goals.

## New goal: neural wind forecasting and physical payload delivery

- Goal created and implementation started: **2026-09-09 17:28:08 PDT** (2026-09-10 00:28:08 UTC).
- Built and showed a separate static quadrotor/cable/parcel scene before motion, with two gates, two platforms and onboard/external cameras.
- Implemented an independent dealiased Navier–Stokes reference, official FNO and PINO training, matched predictive controllers, physical rotor forces, cable tension, contact and release.
- Used the user's RTX 4090 through a verified SSH tunnel for training; evaluated and rendered on the Apple M3 Max. Fixed an upstream CUDA inverse-FFT portability issue, pinned the corrected library commit, and retrained both models.
- Final matched training: FNO **73.4 s**, PINO **82.3 s**, excluding setup/data/validation. Validated forecasts at 64², 128² and unseen 256²; CPU/CUDA/Metal agreement checked.
- Final physical evaluation: **24/24 deliveries** over six held-out winds; no gate collisions or numerical warnings. Mean crossing RMSE: frozen field **10.89 cm**, FNO **4.28 cm**, PINO **4.42 cm**. PINO reduces baseline error **59.4%**; FNO and PINO perform similarly.
- **27 tests passed**; the seven wind tests were repeated after final provenance changes. Native macOS viewer, lint, checkpoint hashes and Git LFS integrity passed.
- Captured the complete **62.00-second, 1920 × 1080, 25 fps** multi-view video. The 43-second flight plays at real time. All **1,550 frames** decoded without error; onboard, crossing, release and final results frames were visually inspected.
- Implementation, video and validation complete immediately before the completion commit: **2026-09-09 18:22:29 PDT** (2026-09-10 01:22:29 UTC).
- Elapsed for this goal: **54 minutes 21 seconds**, including GPU setup, user steering, experiments, the portability fix, training, tests, rendering and communication. The interval since the preceding goal is excluded.

## Follow-up: separate aggressive wind video

- Work started: **2026-09-09 18:41:05 PDT** (2026-09-10 01:41:05 UTC).
- Preserved the original video, checkpoints and standard preset; added a separate aggressive preset and output directory.
- Increased wind speed and field evolution by 1.5× using Navier–Stokes similarity scaling, and doubled crossing speed with unchanged controller gains and actuator limits. Added actual parcel trails and a horizontal target overlay.
- Evaluated all four methods on six fresh winds. Each completed **5/6** clean missions; the shared failure case is retained in the reports and video. On the five shared complete crossing windows, PINO reduced baseline tracking RMSE by **77.0%**; this is a conditional subset result, not an improvement in success rate.
- Captured the complete **50.00-second, 1920 × 1080, 25 fps** comparison. The physical flight plays at real time. All **1,250 frames** decoded without errors; dynamic crossing, delivery and final results frames were inspected.
- **32 tests passed**, native macOS viewer and lint checks passed. Rerunning the original standard trajectory produced exactly identical poses, and the original video hash is unchanged.
- Implementation, recording and validation complete immediately before the follow-up commit: **2026-09-09 19:05:34 PDT** (2026-09-10 02:05:34 UTC).
- Elapsed for this follow-up: **24 minutes 29 seconds**, including experiments, physical failures, tests, rendering and communication. The interval since the previous goal is excluded.


## Follow-up: remove frozen video holds

- Started **2026-09-10 02:29:24 UTC**; finished **2026-09-10 02:32:07 UTC**, immediately before commit.
- Removed the four-second opening hold and five-second delivery hold from Version 2. All 31 seconds of physical flight and the ten-second results card remain, for a **41.00-second** video.
- Updated the aggressive recorder to reproduce this pacing. The original standard video and its recording timing are unchanged; the untrimmed aggressive edit is retained in Git history and a local ignored backup.
- Verified all **1,025 frames** decode, inspected the opening and results transition, confirmed motion starts immediately and checked the original video hash. Lint and formatting checks passed; simulation physics did not change.
- Elapsed: **2 minutes 43 seconds**.


## Follow-up: repository simulation guidelines

- Started **2026-09-10 04:18:07 UTC**; completed **2026-09-10 04:21:49 UTC**, immediately before commit.
- Added root `AGENTS.md` with practices drawn from the hexapod, rovers, amphibious and neural-wind demos: static previews, physical actuation, model limits and clearance, measured success, causal subsystem boundaries, macOS/GPU portability, video verification, version preservation and Git LFS.
- Checked official MuJoCo references, all local documentation links and the four CLI entry points. This documentation-only change does not alter simulations or videos; no physics reruns were necessary.
- Elapsed: **3 minutes 42 seconds**.


## Follow-up: GitHub and GPU synchronization

- Started **2026-09-10 05:18:12 UTC**; setup and checks completed **2026-09-10 05:24:21 UTC**, immediately before the documentation commit and final pull.
- Connected `origin` to `git@github.com:rozgo/mujoco-sandbox.git` and pushed `main`, including **85 Git LFS objects totaling approximately 221 MB** across the published history.
- Cloned the repository on the RTX 4090 machine through the existing verified RustDesk SSH tunnel. The previous training directory was preserved. GitHub SSH read access and a write dry-run passed; no temporary remote branch was created.
- Installed the locked uv environment with wind and Reticulum extras. Confirmed MuJoCo **3.12.0**, PyTorch **2.14.0+cu130** and the RTX 4090. Both committed wind models passed CPU/CUDA agreement checks at 64², 128² and 256²; maximum observed absolute vorticity difference was **8.23e-6**.
- Added the multi-machine workflow and expanded LFS coverage for additional image/video and policy formats. Fast-forward-only pulls protect shared history; machine-specific connection details and credentials remain outside Git.
- LFS integrity, local documentation links and file attributes checked. No simulation behavior or finished video changed.
- Elapsed to setup verification: **6 minutes 9 seconds**.
