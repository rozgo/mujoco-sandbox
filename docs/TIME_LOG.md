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


## Follow-up: SWAP parkour research and environment proposal

- First recorded timestamp for this follow-up: **2026-09-10 05:43:38 UTC**. Research and proposal checks completed **2026-09-10 05:56:54 UTC**, before commit and synchronization. Earlier RL-method discussion and initial reading before the first timestamp are excluded.
- Read the SWAP paper, inspected its public project repository and sampled its gap, climbing and architecture videos. Recorded missing reproduction dependencies and separated author results from our proposed implementation.
- Audited the existing Go2 model's mass, actuator count and bilateral symmetry. Identified a front-calf collision discrepancy and a small base-inertia asymmetry; vendor assets remain unchanged.
- Saved the user's direction and a proposed terrain, sensor, learning, evaluation and video specification in `docs/parkour/`. Checked local documentation links and whitespace. No parkour scene or policy was built, no learning dependencies were installed, and no training was run.
- Recorded research interval: **13 minutes 16 seconds**. This is research/documentation elapsed time, not training time or a completed simulation goal.


## Follow-up: demo gallery README

- Started **2026-09-10 05:47:35 UTC**; content and verification completed **2026-09-10 05:50:59 UTC**.
- Reframed the root README as MuJoCo Sandbox, with four demo sections, screenshots linked to existing videos, playback captions, concise descriptions, and one launch command per demo. Featured the stronger-wind version while retaining links to the original experiment and video.
- Preserved the Sixlegs controls, recording commands, platform notes, implementation map, and evidence links in `docs/hexapod/README.md`; added the guide to the `AGENTS.md` repository map.
- Verified **43 local links**, **11 available PNG/MP4 assets**, all four CLI help entry points and documented launch flags, Git LFS integrity, and whitespace. Existing screenshots were visually reviewed during the preceding proposal; no media files changed.
- Elapsed for this implementation follow-up: **3 minutes 24 seconds**, excluding the earlier proposal and intervening wait. Simulation time: **0 s**; training time: **0 s**; media rendering time: **0 s**. Physics tests and viewer runs were not repeated for this documentation-only change.


## Follow-up: mjbatch source audit and minute-scale training benchmark

- Started **2026-09-10 06:04:57 UTC**; research, benchmark and report checks completed **2026-09-10 06:17:47 UTC**, before commit and synchronization.
- Inspected mjbatch's batching implementation, dependency pins and Go1 example. Recorded the example's trot reward, mirrored-action policy and simplified contact scope; revised the immediate direction toward body-change adaptation.
- Built an isolated checkout using uv 0.12.12, Python 3.14.7, MuJoCo 3.13.0 and PyTorch 2.14.0. Removed its obsolete CUDA package index for current dependency resolution. All 38 upstream tests passed; existing project dependencies were unchanged.
- Ran **60.0123 seconds** of Go1 PPO training from random weights on the M3 Max: CPU simulation/actor, MPS learner, 294 updates and 7,225,344 control transitions. No RTX training was used.
- Evaluated matched initial/final policies on 16 predetermined ten-second trials. Upright survival was 12/16 then 16/16; forward velocity RMSE on four complete paired trials changed from 0.7200 to 0.1532 m/s. Inspected rendered trained-policy poses. This does not demonstrate three-leg adaptation.
- Saved a source review, measured report and proposed adaptation experiment. Marked the earlier SWAP-specific plan as superseded. Documentation links, JSON consistency and whitespace checks passed.
- Elapsed research/setup/benchmark interval: **12 minutes 50 seconds**. Training time is the separate 60.0123-second measurement above.

## Follow-up: public documentation hygiene

- Started **2026-09-10 06:24:42 UTC**; content and repository preparation completed **2026-09-10 06:36:00 UTC**, before the final documentation commit and push.
- Replaced private-source provenance with a public task summary in two rover documents throughout repository history. Retained all 13 existing commits and verified that every other tracked file object was unchanged by the content rewrite.
- Added a rule to review public artifacts for private context before committing or publishing. Cleaned both development clones, pruned old reflogs and unreachable Git objects, and verified zero matches across all 344 remaining Git objects in each clone.
- Recreated the public GitHub repository under the same name at the user's request, with only `main` to be published. Uploaded all **85 Git LFS assets, approximately 221 MB**, to the new repository. The old raw-file URLs checked after recreation returned HTTP 404. No support request was submitted.
- Elapsed to publication preparation: **11 minutes 18 seconds**, including review, history rewriting, synchronization, repository recreation, and asset transfer. Simulation time: **0 s**; training time: **0 s**; rendering time: **0 s**. No simulation behavior or media changed; physics tests were not rerun.

## Follow-up: general adaptation plan for partial leg damage

- Started **2026-09-10 06:41:02 UTC**; research, plan and document checks completed **2026-09-10 06:48:07 UTC**, before commit and synchronization.
- Expanded the user brief from leg-count variants to one controller adapting across partial geometry, restricted joints, degraded actuation and unseen combinations. Saved the implementation plan and linked it from the existing mjbatch review.
- Read the August 2026 embodiment-adaptation paper and current official learner/modeling references. Inspected existing scene composition, Go2 properties, dependency constraints and mjbatch model/state handling. Identified observability, realistic stump/inertia modeling, topology batching, native servo/sensing cost, estimator feedback errors and evaluation leakage as implementation risks.
- Defined staged body/terrain curricula, sensor and policy interfaces, a current-tool integration strategy, optimizer comparison gates, held-out evaluation and proposed success criteria. No new environment, policy or video was implemented; no training goal was started.
- Checked three documents, six local links, code fences and whitespace. Recorded interval: **7 minutes 5 seconds**. Simulation time: **0 s**; training time: **0 s**; rendering time: **0 s**. Physics tests were not rerun for this documentation-only work.

## New goal: bounded adaptive-dog learning experiment

- First implementation timestamp: **2026-09-10 06:53:59 UTC**; goal created at **06:54:16 UTC**. Work is isolated on `experiment/adaptive-dog`; `main` remains at `e889444`.
- Built and showed static body/course previews before movement. Implemented partial calf geometry and derived inertia, missing-calf topology, torque degradation, native servos, causal observations, CPU batching, bounded PPO training, evaluation, native viewing and recording in an isolated current uv project.
- The selected reactive policy has **329.051 seconds** of cumulative training, including its **59.811-second** oracle-context ancestor. The comparison history policy totals **329.349 seconds**. The user-approved extension reached **598.596 seconds** and was preserved separately because it improved missing-calf progress but lost earlier skills.
- An independent history policy trained from random weights on the RTX 4090 machine for **269.590 seconds**; physics ran on CPU, with CUDA neural-network updates. Mac learning used CPU rollout/inference and MPS updates. There is no GPU-physics claim.
- All **nine distinct training stages together consumed 1,137.724 seconds (18 minutes 57.7 seconds)** across the experiments, sharing ancestors where noted. Their logs record **21,037,056 trained control transitions** and **16.590 seconds of model/trainer setup**. This aggregate experiment cost is separate from a single policy's lineage; it excludes package installation, development, evaluation and rendering.
- Corrected excessively soft contacts before recording. The 329-second reactive policy retained 32/32 completions in each of five frozen healthy/partial-body/motor/low-course cases. Missing-calf completion remained 0/32. History gave 0/32 on the new low course, and the independent GPU history seed gave 14/32. All denominators and failures are retained.
- Verified the selected low-step policy at half the physics timestep: **16/16** completions. The inspected corrected rollout had **7.95 mm** maximum sampled penetration and no trunk-contact frames. The longer missing-calf run retained trunk contact and failed the task; it is not presented as successful three-leg walking.
- **47 tests passed** (9 project tests and 38 upstream batch tests). Expected NaN-warning injection in the upstream warning-counter test writes an ignored MuJoCo log; accepted policy runs reject numerical warnings. Native macOS viewer passed. Checkpoint hashes, documentation links, LFS attributes and CPU/MPS/CUDA inference agreement passed; maximum checked action difference was **2.87e-6**.
- Recorded three **1280 × 720, 25 fps, real-time** videos: a 48-second synchronized following/head/overview demonstration, a 12-second history/reactive comparison, and a 12-second extra-training comparison. All **1,800 encoded frames** decoded without errors; opening, terrain, fault, failure and final states were visually inspected. Camera pixels are observer output. Rendering wall time was not independently instrumented; it is included in the elapsed interval below.
- Implementation, media and validation checkpoint: **2026-09-10 07:45:44 UTC**, before the completion commit/push and final GPU synchronization. Elapsed to this checkpoint: **51 minutes 45 seconds**, including user steering, experiments, code, debugging, recording and communication. This is not a claim that the full environment was built in five minutes.

## Follow-up: healthy dog and normal walking

- Started **2026-09-10 14:48:45 UTC**. The intervening pause since the adaptive-dog pilot is excluded.
- Trained a separate healthy-only walking policy from random weights, with four intact legs, full motor strength, no faults, level ground and soft posture/smoothness rewards. No supplied gait, phase clock, foot trajectory or pretrained policy was used.
- First stage: **59.642 seconds**. Additional refinement: **59.519 seconds**. Total training: **119.161 seconds**, **2,801,664 control transitions**, CPU MuJoCo/mjbatch and an MPS learner on the Mac. Both stages are retained.
- Original distance-based healthy evaluation: **64/64** completions; half-timestep check: **16/16**. The later foot-support follow-up below found rear-right knee support; these original scores did not certify correct walking. The inspected gait uses all four feet with no trunk-ground contact, mean base height 27.7 cm and maximum sampled penetration 3.44 mm. Gait asymmetry and remaining speed-tracking error are documented.
- Added `adaptive-dog walk`, tested the native macOS launcher, passed **48 tests**, checked CPU/MPS inference agreement and checkpoint hashes, and recorded two 12-second 1280×720/25 fps videos. All **600 encoded frames** decoded; beginning, stride and final frames were inspected. The healthy video provides three synchronized observer cameras.
- Implementation, recording and validation complete **2026-09-10 14:58:58 UTC**, before the completion commit and synchronization. Total elapsed to this checkpoint: **10 minutes 13 seconds**. Includes development, training, evaluation, rendering and communication; rendering wall time was not separately instrumented. Existing adaptive policies and `main` remain unchanged.

## Follow-up: stop the healthy dog using its knee for support

- Started **2026-09-10 15:02:05 UTC** after the user identified rear-right knee-supported walking. The original foot/trunk diagnostic missed it. Strict replay validation gives the original healthy policy 32/32 distance completions but **0/32 support-valid completions**; old artifacts are retained and labeled accordingly.
- Added native contact-force sensing for every physical robot geom, with allowed support derived from actual leg geometry. Intact terminal feet and designated distal stumps are allowed; the remaining leg and chassis are penalized. Collisions, masses, torque limits and actor observations remain unchanged.
- Refined the existing healthy policy for **89.412 seconds**, bringing its full training ancestry to **208.573 seconds** and **4,890,624 transitions**. Used 512 CPU MuJoCo/mjbatch environments and MPS learning on the Mac. No additional GPU training was required.
- New policy: **32/32 development**, **64/64 fresh final**, and **16/16 half-timestep** support-valid completions. Every physics step was checked; forbidden ground-support forces were zero throughout both final sets. Mean inspected height is 30.1 cm, all four feet contribute, and maximum sampled penetration is 4.52 mm. This remains a healthy, level-ground test with one training seed.
- **52 tests passed**; loaded-knee detection, allowed-stump loading and unchanged dynamics with contact instrumentation were tested. CPU/MPS maximum action difference was **7.16e-7**; native Mac viewing passed. Both new 12-second, 1280×720/25 fps videos decode all **600 frames**; opening, stride and ending views were inspected. Fixed a comparison-caption overlap before delivery.
- Implementation, media and validation checkpoint: **2026-09-10 15:22:04 UTC**, before final commit, push and GPU synchronization. Elapsed: **19 minutes 59 seconds**, including diagnosis, code, training, evaluation, rendering and documentation. Rendering wall time was not independently measured. All work stays on `experiment/adaptive-dog`; `main` and previous videos/checkpoints are preserved.

## Follow-up: longer healthy-dog strides

- Started **2026-09-10 16:16:47 UTC**. The user requested longer steps while retaining the otherwise good healthy gait.
- Added an optional landing-event reward for forward swing travel, with slip and flight costs. It does not prescribe gait phases, leg pairing, symmetry or foot trajectories. The existing support rule and physical parameters remain active; `--stride-weight 0` preserves the previous training behavior.
- A single **89.216-second** refinement collected **1,978,368 transitions**. The selected policy totals **297.789 seconds**, **6,868,992 transitions**, including all ancestors. Training ran on 512 native CPU MuJoCo/mjbatch environments with MPS network updates. No extra allowance or RTX training was needed.
- Matched fresh-seed measurements over 64 trials per policy: mean stride **21.68 → 36.28 cm (+67%)**, cadence **2.86 → 1.71 cycles/s**, forward speed **0.616 → 0.618 m/s**, planted-foot horizontal speed RMS **0.185 → 0.079 m/s**. Body-height variation increases from 0.43 to 1.27 cm standard deviation; brief all-feet-unloaded samples occur 1.34% of the time. These tradeoffs are retained in the report.
- New policy passed **64/64** support-valid trials and **16/16** at half timestep. Forbidden support force was zero throughout both sets; inspected maximum sampled penetration was **6.38 mm**, and torque caps remained unchanged. **54 tests passed**. CPU/MPS action difference was below **9.54e-7**; native macOS viewing passed.
- Preserved the compact policy as `walk --style compact`; the longer gait is the default. Recorded matched comparison and three-camera videos, each twelve seconds at 1280×720/25 fps/1×. Decoded all **600 frames** and inspected opening, stride and final frames.
- Implementation, media and validation checkpoint: **2026-09-10 16:25:36 UTC**, before final artifact checks, commit, push and GPU synchronization. Elapsed: **8 minutes 49 seconds**. Includes development, measurement, training, rendering and documentation; rendering wall time was not separately instrumented. Work remains on `experiment/adaptive-dog`, with `main` and previous media preserved.

## Follow-up: established rewards for a more balanced gait

- Started **2026-09-10 16:45:40 UTC**, following the user's approval to train with known rewards. Adapted Isaac Lab Spot's completed stance/swing variance and body-motion costs into the current NumPy/MuJoCo learner, with pinned source attribution and BSD-3-Clause license. Physical parameters and actor observations remain unchanged.
- First attempt, balance/body-motion weights 20/1: **89.732 seconds**, 2,138,112 transitions; failed all 32 development completions and used forbidden support. This failed checkpoint and its reports are retained.
- Restarted from the prior longer-stride policy with weights 5/0.5: **119.448 seconds**, 2,789,376 transitions. Its intermediate and final checkpoints both passed 32/32 support-valid development trials. Selected iteration 200 at **105.540 seconds** because it had lower duty-factor imbalance while retaining strides above 30 cm. Selection preceded fresh final evaluation.
- This follow-up spent **209.180 seconds (3 min 29 s)** on the two complete training runs. The selected policy's full ancestry is **403.329 seconds (6 min 43 s)** and **9,326,592 control transitions**. The failed run and the 13.907 seconds after the selected checkpoint are accounted for as experiment cost. Training used 512 native CPU MuJoCo/mjbatch environments with MPS network updates; no new RTX training was needed.
- Fresh paired 64-trial measurements: mean left/right duty-factor gap **21.56 → 8.44 percentage points (61% lower)**, body-height standard deviation **1.27 → 0.78 cm (38% lower)**, stride **36.28 → 31.30 cm**, speed **0.617 → 0.609 m/s**, and planted-foot horizontal speed RMS **0.084 → 0.052 m/s**. Measurable asymmetry remains and is shown per leg in the report.
- Selected policy passed **64/64 support-valid trials**, plus **16/16 at half timestep**, with zero forbidden support forces at every physics substep. Inspected maximum sampled penetration was **3.00 mm**. All **56 tests passed**, CPU/MPS action agreement passed, and the native macOS viewer passed.
- Added observer contact indicators and recorded a matched comparison plus following/head/overview video. Each is twelve seconds at 1280×720, 25 fps and 1×. All **600 frames** decoded; opening, stride and final views were inspected. Previous policies and media remain selectable and unchanged.
- Implementation, media and validation checkpoint: **2026-09-10 17:00:47 UTC**, before final commit, push and GPU synchronization. Elapsed: **15 minutes 7 seconds**, including development, both training trials, evaluation, checkpoint selection, rendering and documentation. Rendering wall time was not measured separately. Work remains isolated on `experiment/adaptive-dog`.

## Follow-up: damaged training while retaining healthy walking

- Started **2026-09-10 17:09:52 UTC**. Resumed the balanced healthy policy with mixed intact, weakened and shortened-calf episodes. Added healthy-only frozen-reference similarity rewards and soft actor retention, with fixed observation normalization and a smaller learning rate. Damaged states retain progress/support/stride/stability rewards without teacher imitation or an intact timing-balance requirement.
- One **89.637-second** training run, 823,296 transitions and 67 PPO iterations. The selected final checkpoint totals **492.966 seconds (8 min 13 s)** and **10,149,888 transitions**, including all its ancestors. Setup took 0.740 seconds separately. CPU MuJoCo/mjbatch and MPS learning ran on the Mac; no new remote GPU training was necessary. Intermediate checkpoints are from this same run and add no separate compute.
- Selection preserved the predefined healthy-gait gates. Iteration 25 passed 51/64 shortened-calf development trials; iteration 50 passed 64/64 but failed the healthy timing-asymmetry gate. The final policy passed both and was selected before fresh evaluation. All checkpoint candidates and reports remain available.
- Fresh seed 20260915: **128/128** support-valid trials across the four 70%-calf variants, **32/32** healthy, **32/32** unseen 58%-calf and **32/32** unexpected 25%-torque trials. The healthy reference had 0/128 shortened-calf completions but already passed that motor fault. Two shortened calves remain unreliable: **1/32**. Another **40/40** healthy/single-shortened trials pass at a 1 ms timestep; forbidden support forces are zero in these healthy/single-shortened sets.
- Healthy mean stride **31.30 → 30.60 cm**, speed **0.609 → 0.600 m/s**, duty-factor gap **8.23 → 10.54 percentage points**, body-height variation **7.87 → 7.07 mm**. This retains useful healthy walking with measurable tradeoffs; it does not establish perfect gait preservation, arbitrary damage adaptation or precise damaged speed tracking.
- **58 tests passed**. CPU/MPS action difference is below 1.91e-6, native Mac viewing passed, and all measured actuator forces remain within effective caps. Three videos total **60 seconds / 1,500 frames**, 1280×720 at 25 fps and 1×. All frames decoded, encoded opening/stride/end views were inspected, and the overview was reframed by replaying saved states. Maximum sampled penetration in the main recordings is 4.09 mm.
- Implementation, documentation, media and validation checkpoint: **2026-09-10 17:23:36 UTC**, before final repository checks, commit, push and GPU synchronization. Elapsed: **13 minutes 44 seconds**. Training is measured separately above; rendering/evaluation wall time was not independently instrumented. Work remains on `experiment/adaptive-dog`, preserving `main` and prior demos.

## Follow-up: paired calf damage with earlier skills retained

- Started **2026-09-10 19:57:30 UTC** after approval of the paired-damage plan. Added four trained pair combinations, two entirely held-out combinations, mild/strong stages, and a training-only reference loss for single-damage states. Rendered and inspected the static healthy/mild/strong preview before training. No live geometry or pose edits were introduced.
- Mild stage: **29.703 seconds**, 491,520 transitions. Strong stage: **59.692 seconds**, 958,464 transitions. This task spent **89.396 seconds** training across both stages, 1,449,984 transitions, using CPU MuJoCo/mjbatch and MPS learning on the Mac. No extra extension or new remote GPU training was required.
- Selected strong-stage iteration 25, at **17.953 seconds** into that stage. Its cumulative training is **540.622 seconds (9 min 1 s)** and **10,948,608 transitions**. Selected new ancestry is 47.656 seconds, while all later updates remain counted in the 89.396-second task cost. Iterations 25 and 75 tied on success and retention; the earlier checkpoint was selected before final evaluation. The final checkpoint failed the healthy-speed gate and was retained as unselected.
- Fresh seed 20260916: trained paired-damage completion **33/128 → 127/128**, including the target diagonal case **1/32 → 32/32**. Healthy **32/32**, single shortened calves **128/128** and the tested motor fault **32/32** remain intact. Held-out pair/length cases improved **94/96 → 96/96**; most of that generalization already existed in the parent. One both-front-leg trial fell at 0.96 seconds and remains in the failure report.
- Healthy mean stride **30.60 → 31.17 cm**, speed **0.601 → 0.578 m/s**, duty-factor gap **10.64 → 6.64 percentage points**, body-height variation **7.07 → 8.90 mm**. Damaged gaits may remain highly asymmetric. This remains a flat-ground, partial-damage experiment, not arbitrary limb-loss recovery or parkour.
- Half-timestep healthy/single/target checks pass **24/24**. **60 tests passed**, native Mac viewing passed and CPU/MPS action differences stayed below 2.87e-6. All selected recordings used allowed support and respected torque caps; maximum sampled penetration was 5.52 mm.
- Recorded a 36-second three-camera demonstration plus two twelve-second matched comparisons, all 1280×720, 25 fps and 1×. Decoded all **1,500 frames** and inspected encoded openings, stride and endings. All previous checkpoints, videos and `main` remain preserved.
- Implementation, documentation, media and validation checkpoint: **2026-09-10 20:11:49 UTC**, before final repository checks, commit, push and GPU synchronization. Elapsed: **14 minutes 19 seconds**, including development, training, selection, evaluation, rendering and documentation. Rendering/evaluation wall time was not separately instrumented; training time is explicitly separated above.

## Follow-up: all trained body variants in one video

- Started **2026-09-10 20:24:00 UTC**. Added a synchronized grid of fourteen independent rollouts: all thirteen geometry variants across the mild/strong curriculum plus one representative motor fault. Every panel uses the same selected checkpoint, with no teacher inference, policy switching or new training. Earlier media are preserved.
- Reused two matching saved trajectories and captured twelve new ones at seed 9137. Each trial spans twelve simulation seconds. Capture took **5.438 seconds**; rendering/export took **51.925 seconds**, excluding renderer/context setup. The policy's existing cumulative training remains **540.622 seconds**; new training time is **0 seconds**.
- All fourteen demonstrations reached 5 m, survived and passed allowed-support checks at every physics substep. Torque caps, synchronized timestamps and trajectory/checkpoint hashes passed. These fixed demonstrations do not replace the broader evaluation or its retained failed trial.
- Recorded **3840×2160, 25 fps, twelve seconds at 1×**. All **300 encoded frames** decoded, and opening, fault, stride and ending frames were inspected. Ruff formatting/checks and CLI help passed; this video-only follow-up did not rerun the full dynamics test suite.
- Implementation, documentation, media and validation checkpoint: **2026-09-10 20:34:29 UTC**, before final repository checks, commit, push and GPU synchronization. Elapsed: **10 minutes 29 seconds**, including implementation, capture, rendering, QA and communication. Work stays on `experiment/adaptive-dog`; `main` is preserved.

## Follow-up: complete lower-leg and whole-leg loss

- Started **2026-09-10 23:01:09 UTC** after approval to retire shortened-leg training and focus on complete removals. Added actual whole-leg subtree removal, semantic missing-joint/contact slots, nine body conditions and a static preview before movement. Work remains on `experiment/adaptive-dog`.
- Five new training rounds: **59.266**, **89.345**, **149.356**, **149.545** and **149.721 seconds**. Total **597.233 seconds (9 min 57 s)** and **11,968,512 transitions**, CPU MuJoCo/mjbatch with MPS learning on the Mac. No new RTX training was needed. Training stopped at the agreed ceiling.
- The first round failed the removed-limb task while retaining healthy gait. The next learned both rear calf removals. The whole-leg round learned six of eight removals, with both front-right cases stationary. A focused extension learned those two but regressed on earlier skills. Final balanced consolidation used frozen training-only references to combine the successful behaviors into one deployed actor. All stages and inspected candidates are retained.
- Selected consolidation iteration 50, **31.735 seconds** into that round, before final evaluation. Full ancestry is **764.683 seconds (12 min 45 s)** and **16,625,664 transitions**, including the **403.329-second** healthy parent. New selected ancestry is 361.354 seconds; all unused/later training remains counted in the larger 597.233-second experiment cost. Reference lineages are included in the selected parent's ancestry.
- Fresh seed 20260917: **254/256** timed removal completions versus **0/256** for the healthy parent, and **32/32** healthy completions. All selected-policy trials stayed upright, crossed 5 m and used only allowed support. Two whole-front-right trials crossed too late for the required full one-second goal window; both remain failures. Healthy stride **31.30 → 33.09 cm**, speed **0.609 → 0.635 m/s**, duty-factor gap **8.46 → 5.23 percentage points**, body-height variation **7.84 → 7.40 mm**.
- **72/72** half-timestep trials passed across all nine bodies. **62 tests passed**, native Mac viewing passed and CPU/MPS maximum action difference was **7.15e-7**. Every physics substep was checked for forbidden support. All recorded torques stayed within original caps; maximum sampled penetration was **5.04 mm**.
- Recorded nine synchronized successful demonstration cases using the same checkpoint, **3840×2160, 25 fps, twelve seconds at 1×**. All **300 encoded frames** decoded; opening, stride and final frames were inspected. Capture took **4.193 seconds** and rendering/export **39.979 seconds**, excluding renderer/context setup. Camera pixels remain observer output; commands are scripted.
- Implementation, documentation, media and validation checkpoint: **2026-09-10 23:30:40 UTC**, before final repository checks, commit, push and GPU synchronization. Elapsed to this checkpoint: **29 minutes 31 seconds**, including development, training, selection, evaluation, recording and communication. The earlier demos, their media and `main` are preserved.

## Follow-up: general smoothness and bilateral consistency after limb loss

- Started **2026-09-10 23:37:39 UTC** after the user identified oscillatory front-right removal gaits. The user then requested general symmetry-encouraging behavior instead of tuning one side. The first side-focused diagnostic was retained but excluded from selection.
- Added soft active-joint command-change, roll/pitch angular-rate and finite-difference joint-acceleration costs, applied equally across missing-limb bodies. Added a soft mirror-consistency loss that reflects the missing-joint mask together with observations/actions. It does not tie opposite limbs within one damaged body, mirror actions at runtime, filter actions or alter physics.
- Three rounds used **149.628**, **149.615** and **149.747 seconds**: **448.991 seconds (7 min 29 s)** total new training, **7,876,608 transitions**. The two general rounds used **299.363 seconds**. CPU MuJoCo/mjbatch with MPS learning on the Mac; no new NVIDIA training. All tried checkpoints and reports are retained.
- Selected the second general round's iteration 100 using predeclared development gates across all nine bodies. It adds **223.475 seconds** of selected ancestry; full ancestry is **988.158 seconds (16 min 28 s)** and **20,312,064 transitions**, including the previous limb-loss and healthy training.
- Fresh seed 20260918: **288/288** valid completions for both old and refined policies. The refined policy passed **72/72** half-timestep checks and all healthy gait gates. Across eight removal cases, mean per-body relative reductions were **32.2%** in command-step RMS, **15.0%** in joint-motion RMS above 6 Hz and **53.6%** in roll/pitch angular-rate RMS. Two cases retain slightly higher high-frequency motion; healthy timing imbalance increases within its declared bound. All limitations and failed development candidates are documented.
- On 1,080 identical reference-policy states, mean per-case mirror RMSE fell **0.619 → 0.128**, approximately **79%**. This measures reflected-state action consistency, not identical gait phases. **67 tests passed**, Ruff passed, native Mac viewing passed and CPU/MPS maximum action difference was **8.35e-7**.
- Recorded a new nine-panel **3840×2160, 25 fps, twelve-second** video at **1×**, preserving the original. All nine demonstration runs succeeded; all **300 encoded frames** decoded and opening/middle/ending frames were inspected. Capture took **3.890 seconds**, rendering/export **39.245 seconds**, excluding renderer/context setup. Maximum sampled penetration was **4.953 mm** and peak recorded torque ratio **0.918** of the original cap. Support acceptance still checks every physics substep.
- Implementation, documentation and verified-media checkpoint: **2026-09-11 00:01:40 UTC**, before final commit, push and GPU synchronization. Elapsed to this checkpoint: **24 minutes 1 second**, including diagnosis, implementation, training, evaluation, rendering and communication. Work remains on `experiment/adaptive-dog`; prior media and `main` are preserved.

## Follow-up: clearer damage markers and walking contact shadows

- Started **2026-09-11 00:39:35 UTC**. The user requested WALKING labels, orange spheres marking the cut ends, and clearer shadows for ground contact.
- Added an observer-only `damage` presentation preset: orange spheres at the actual calf cut or original hip attachment, cameras facing the damaged side, reduced headlight fill, a brighter floor and a following spotlight with a 4096-pixel shadow map. Markers add no physical geometry, mass or sensor surfaces. Existing media and the original visual preset are retained.
- Replayed all nine previously validated trajectories, with identical checkpoint, physical-model hashes, timestamps, state arrays, forces and outcomes. **No new training** and **no new full mission rollouts**. Replay preparation took **0.537 seconds**; rendering/export took **40.760 seconds**, excluding renderer setup.
- Both lower-FR and whole-FR produced exactly unchanged joint positions and observations over 100 live control steps with the visual preset enabled. Native Mac viewing passed a five-second smoke test. Ruff passed; the full RL suite was not rerun for a presentation-only edit.
- New **3840×2160, 25 fps, twelve-second** video at **1×**. All **300 encoded frames** decoded. Opening, middle and ending frames were inspected; every orange marker was visible in six sampled frames spanning the clip.
- Verified-media checkpoint **2026-09-11 00:47:46 UTC**, before final commit, push and GPU sync. Elapsed **8 minutes 11 seconds**. Work remains on `experiment/adaptive-dog`; main and earlier videos are preserved.

## Follow-up: minimum ground-support penalty

- Started **2026-09-11 00:57:18 UTC**. The user asked to try only the first, smallest correction before proceeding to further gait rewards. Restored a standalone 0.5 no-support cost for moving bodies with missing joints; no phase clock, other new rewards, actor inputs or physical changes.
- One continuation used **119.483 seconds**, **2,052,096 transitions**, CPU MuJoCo/mjbatch with MPS learning. The final iteration 167 was selected; iterations 50 and 100 remain archived with their outcomes. Full selected ancestry: **1107.641 seconds (18 min 28 s)** and **22,364,160 transitions**. No new NVIDIA training.
- Fresh seed 20260919: **288/288 valid completions** for both parent and candidate. Equally weighted mean airborne fraction across eight removals fell **5.64% → 3.34% (40.8%)**, and no-support-force fraction **11.92% → 6.91% (42.0%)**. Mean speed **0.568 → 0.573 m/s**; healthy gait gates retained. Sample window 1–12 seconds, 20 ms support/clearance samples; forbidden loads still checked every 2 ms. Hopping remains and whole-FL vertical bounce increases slightly. Stop at this first intervention for review.
- **68 tests passed**, Ruff passed, all **72 half-timestep trials** passed, native Mac viewer smoke test passed. CPU/MPS maximum action difference **9.54e-7**. All nine new recorded trials pass; maximum sampled penetration **4.06 mm**, peak recorded torque **87.3%** of cap.
- New nine-panel **3840×2160, 25 fps, 12-second** video at **1×**, with orange markers and shadows. All **300 encoded frames** decoded, opening/middle/ending inspected, model/trajectory/checkpoint hashes verified. Capture **4.485 seconds**, rendering/export **44.401 seconds**, excluding renderer setup. Prior policies and media are preserved.
- Verified implementation, evaluation, documentation and video checkpoint: **2026-09-11 01:12:31 UTC**, elapsed **15 minutes 13 seconds**, before final commit/push/GPU synchronization. Work remains on `experiment/adaptive-dog`; main is unchanged.

## Follow-up: simple healthy-walk style and preview review

- Started **2026-09-11 01:17:07 UTC**. The user wanted longer support and strides, then requested a simpler objective: compensate while looking as much as possible like the healthy dog. A proposed stance-timer change was removed before training.
- Added an optional healthy-policy prior over surviving joints, then a healthy-motion reference library with each leg independently matched by joint position/velocity and command. No actor phase inputs, new joint limits or physical changes. Removed the extra damaged-gait heuristics from these trial configurations. All existing selected policies remain available.
- Two new rounds used **149.317** and **149.655 seconds**, total **298.972 seconds (4 min 59 s)** and **4,534,272 transitions**, CPU MuJoCo/mjbatch with MPS learning. Direct policy imitation lost task completion; snapshot motion imitation passed 72/72 development trials but shortened average stance **0.194 → 0.178 s** and stride **0.176 → 0.162 m**. Neither trial met the predeclared gait gates; no checkpoint was promoted. All six inspected candidates and outcomes are retained.
- The user requested a video before further training. A proposed temporal-reference extension was archived locally and reverted from delivered source **without training it**. Final holdout seed 20260920 remains unused; no new final acceptance or half-timestep claim. Further training is paused for review.
- Recorded the final snapshot-reference checkpoint as **EXPERIMENT / HEALTHY STYLE / NOT SELECTED**, all nine cases, 3840×2160, 25 fps, twelve seconds at 1×. Its ancestry is **1257.296 seconds (20 min 57 s)**; both failed rounds are counted in experiment cost. Capture **3.804 seconds**, rendering/export **39.951 seconds**. All 300 frames decoded, opening/middle/end inspected, hashes verified; nine captured tasks pass, maximum sampled penetration **5.16 mm**, peak recorded torque **93.3%** of cap. Native Mac viewer passed; delivered code **71 tests passed**, Ruff passed.
- Verified preview and review checkpoint **2026-09-11 01:45:23 UTC**, elapsed **28 minutes 16 seconds**, before final commit/push/GPU synchronization. Video links were delivered immediately for review. Work remains on `experiment/adaptive-dog`; main and the selected ground-support policy are unchanged.

## Follow-up: causal healthy-motion reference

- Started **2026-09-11 01:48:49 UTC** after approval to continue the temporal-reference plan. Added the current joint state and the same leg's state 0.12 seconds earlier to training-only reference matching. Each leg chooses its own reference phase. The deployed actor and physical model are unchanged.
- Two bounded rounds used **89.685 + 89.218 = 178.903 seconds (2 min 59 s)** and **3,330,048 transitions**, CPU MuJoCo/mjbatch with MPS learning. The first late checkpoints improved, justifying one predeclared continuation under the user's standing allowance. The continuation regressed; training stopped at that limit. Including the two earlier style trials, this line of experiments used **477.875 seconds (7 min 58 s)**.
- The first final checkpoint passes **72/72** development tasks and preserves healthy gait. Mean intact-leg stance **0.194 → 0.201 s** and stride **0.176 → 0.178 m** improve slightly, but fail the declared 20% healthy-gap reduction. Rear-left stride regressions and higher airborne time also fail gates. The extension ends at **59/72** tasks. All six inspected candidates and failures are retained; no new selected policy. Final seed 20260920 remains unused and no new half-timestep acceptance is claimed.
- Preserved the first final checkpoint as a visual candidate and recorded all nine conditions. The user liked the video during review. Preview ancestry is **1197.326 seconds (19 min 57 s)** and **23,973,888 transitions**; later unsuccessful training is excluded from that ancestry but included in experiment cost.
- **74 tests passed**, Ruff passed, native Mac viewing passed. CPU/MPS maximum action difference **7.16e-7**. The new 3840×2160, 25 fps, twelve-second video plays at 1× and labels the policy experimental/unselected. All nine captures complete with allowed support, all 300 frames decode, and encoded opening/middle/end views were inspected. Timestamps, model/checkpoint/trajectory hashes verified; maximum sampled penetration **4.19 mm**, peak torque **82.4%** of cap.
- Capture **3.929 seconds**, rendering/export **40.699 seconds**, excluding renderer setup. Verified implementation, documentation and media checkpoint **2026-09-11 02:05:51 UTC**, elapsed **17 minutes 2 seconds**, before final commit/push/GPU synchronization. Work remains on `experiment/adaptive-dog`; main and all previous videos are preserved.

## Follow-up: rear-foot clearance, numerically passing but visually rejected

- Started **2026-09-11 03:40:11 UTC**. Three bounded rounds used **328.148 seconds (5 min 28 s)**, **6,291,456 transitions**, CPU MuJoCo/mjbatch and MPS learning. Selected ancestry **1412.865 seconds**; every tried checkpoint retained.
- Fresh final seed 20260921: **288/288** tasks; half timestep **72/72**. Rear moving clearance **4.65 → 12.64 mm**, dragging metric **67.1%** lower. All declared retention gates, **78 tests**, Ruff and native viewer passed; CPU/MPS difference **8.35e-7**.
- Nine-case 4K capture **4.269 s**, render **45.850 s**; 1440p matched rear comparison render **26.111 s**. Both twelve-second, 25 fps, 1× videos fully decoded (300 frames each).
- At **2026-09-11 04:06:15 UTC**, the user rejected the gait visually: all feet still appeared to drag. Elapsed **26 min 4 s** to this review. Numerical results remain archived, but do not establish the requested visible steps. This review starts a new all-intact-feet follow-up.
