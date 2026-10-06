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

## Follow-up: every intact foot must visibly lift

- Started **2026-09-11 04:06:15 UTC** after the previous rear-clearance video was rejected as still dragging. Added an optional completed-swing reward for every intact foot and reduced imitation of the old shuffle. No actor phase input, commanded foot trajectories or extra physical forces.
- Two bounded rounds used **179.748 + 179.395 = 359.143 seconds (5 min 59 s)** and **7,225,344 transitions**, CPU MuJoCo/mjbatch with MPS learning. The first learned visible swings but added body flight; the second retained the swings while reducing flight. All six inspected checkpoints and reports are retained. Full selected ancestry **1772.007 seconds (29 min 32 s)**, **35,340,288 transitions**. No new NVIDIA training.
- The selected final policy passed all development and initial final gait gates, but the 2 ms video failed the existing penetration limit (9.39 mm >8 mm). The earlier eligible checkpoint also failed that check. A numerical-resolution probe reduced penetration without changing weights, limits, contact parameters or control frequency. Final delivery uses **0.5 ms physics / 20 ms control**; original 2 ms defaults and media remain available. No training was repeated for this change.
- Fresh final seed **20260923** at 0.5 ms: **288/288** support-valid tasks; every intact foot/trial completes at least **19** qualifying visible steps. All 28 body/foot combinations pass. Mean swing peaks **2.41 → 7.26 cm**, per-foot new means **5.54–10.84 cm**. Mean damaged-body airborne fraction **4.38% → 2.57%**. Healthy gait and all stride/stance/speed/body-motion retention gates pass. **72/72** checks at 0.25 ms pass.
- **81 tests**, Ruff, native Mac viewer at the final timestep, and CPU/MPS action agreement (max **9.54e-7**) pass. New nine-case **3840×2160, 25 fps, 12-second, 1×** video: all **300 frames** decoded; opening/middle/end inspected. Maximum sampled penetration **7.62 mm**, actual recorded torque within original caps. Capture **9.985 s**, rendering/export **45.549 s**, excluding context setup. Archived 2 ms render and failed QA remain documented.
- Verified implementation, media and documentation milestone **2026-09-11 04:35:12 UTC**, elapsed **28 min 57 s**, before final repository commit/push/GPU sync. The user called this version almost perfect, then requested a diagnosis of rear-leg synchrony in front-right removal cases.

## Follow-up: diagnose front-right rear-leg synchrony

- Started **2026-09-11 04:35:38 UTC**. Reused recorded final-video states; no training and no new full missions. Both whole-FR and lower-FR have nearly synchronous rear swings (**6.5% and 10.6%** cycle separation), versus **58.7% and 58.2%** for corresponding FL cases. Both rear feet exceed 1 cm clearance together for **7.8% / 6.9%** of time in FR cases and **0%** in FL cases. These are single-demo measurements, not a generalization claim.
- Saved the diagnostic code, hashes and proposed smallest next trial: a soft overlapping-rear-swing cost for front-damaged bodies with two intact rear feet, preserving all current lift objectives. No additional training was run in response to this request for advice.
- Diagnostic/documentation milestone **2026-09-11 04:37:31 UTC**, elapsed **1 min 53 s**, before final commit/push/GPU sync. Work remains on `experiment/adaptive-dog`; main and all earlier videos are preserved.

## Follow-up: minimal rear support refinement and symmetry evaluation

- Started **2026-09-11 04:39:57 UTC** after approval for the minimal rear-overlap cost. The cost applies equally to FL/FR injury when both rear legs remain intact. All existing lift/healthy-style objectives and physical parameters were retained.
- Two bounded rounds used **89.546 + 89.568 = 179.114 seconds (2 min 59 s)** and **3,624,960 transitions**, CPU MuJoCo/mjbatch with MPS learning. The second increased only the overlap cost from 1 to 3, following the first round's partial improvement. All six inspected candidates and their reports are retained; training stopped after this extension.
- Final experimental checkpoint passes **72/72** development tasks and all existing lift/healthy/retention gates. Lower-FR simultaneous rear swing falls **7.0% → 0.27%**, whole-FR **7.80% → 1.48%**. Phase separation reaches only **17.6% / 11.0%** of a cycle, below the predeclared 25% requirement. No policy is promoted; final holdout 20260924 remains unused. Candidate ancestry is **1922.725 seconds (32 min 03 s)** and **38,400,000 transitions**; this excludes the unused portion of round one, which remains counted in experiment cost.
- The user then requested evaluation of symmetry, explicitly without asking to apply it. A frozen-policy reflection probe used no training and changed no deployed behavior. Across four front-removal bodies, two methods and eight trials, **64/64** tasks complete. First-trial rear timing on lower-FR changes **10.4% → 41.9%** separation, whole-FR **6.4% → 41.0%**. The opposite reflection transfers poor FR timing to FL. This supports investigating soft mirror consistency, but does not establish the result of a trained mirror loss. Existing recent runs used symmetry weight zero; the option and research are documented in `SYMMETRY_OPTION.md`.
- **84 tests**, Ruff and native Mac viewing pass. Experimental checkpoint CPU/MPS maximum action difference **1.67e-6**. Archived nine-case overlap-only video is **3840×2160, 25 fps, 12 seconds at 1×**, explicitly NOT SELECTED. All nine captures complete; all 300 frames decode, opening/middle/ending inspected. Capture **9.421 seconds**, rendering/export **40.088 seconds**, excluding context setup. Video QA fails the unchanged penetration gate on lower-FR: **8.889 mm > 8 mm**. All original torque caps hold. This is a second reason to retain the previously approved policy/video, not to relax the gate.
- Documentation and experimental-media milestone **2026-09-11 04:59:04 UTC**, elapsed **19 min 7 s**, before final repository checks/commit/push/GPU synchronization. No symmetry fine-tune started. Work remains on `experiment/adaptive-dog`; main and the approved visible-step artifacts are preserved.

## Release: user-accepted adaptive dog v1 and shareable GIF

- Started **2026-09-11 05:13:11 UTC** after the user explicitly accepted the rear-support solution and requested its promotion to `main`, updated README and official video. The user subsequently requested a shareable GIF. This is an explicit selection decision after visual review; the original automatic gates are not relabeled as passed.
- Froze the exact existing stronger checkpoint as `adaptive_dog_v1.pt`, with byte-identical SHA-256 `174b3094eaf20233f151ff0ad19fc21c4612e8b1e7468e683c81ab8f980d846b`. Added a hash-pinned `adaptive-dog demo` launcher at the existing 0.5 ms runtime timestep. **No new training, rewards, gait changes or contact changes.** Updated the main README, run guide, release manifest and future-iteration instructions.
- Used reserved seed **20260924** only after user selection. Frozen checkpoint passes **288/288** full tasks and **72/72** checks at 0.25 ms. All prior visible-step, healthy-gait, stride/stance, speed and body-motion gates pass. Original FR phase separation >=0.25 cycle still fails, and the approved video retains **8.889 mm** sampled contact penetration against its original **8 mm** target. All original outcomes and the experimental-labeled video remain archived.
- **46 package tests + 38 mjbatch tests = 84 passed**, Ruff and formatting checks pass, and the new demo command passes a five-second native Mac viewer smoke test. Existing other-demo source, tests and dependency locks are unchanged; their suites were not repeated for this release.
- Replayed the identical approved nine trajectories with a v1 heading: preparation **0.590 seconds**, render/export **45.525 seconds**, excluding renderer setup. Official MP4: **3840×2160, 25 fps, 12 seconds at 1×**. All 300 frames decode; opening/middle/ending inspected. Checkpoint, physical-model and trajectory hashes match the approved original.
- GIF: **960×540, 10 fps, 12 seconds at 1×**, infinite loop, **8,934,941 bytes**. All 120 frames decode and a midpoint frame was visually checked. Final encoding took **2.687 seconds**; an earlier 1280×720, 15 fps encoding took **3.713 seconds** and was replaced because its 33 MB size was less suitable for sharing. Both attempts are recorded in GIF provenance.
- Release-preparation milestone **2026-09-11 05:21:28 UTC**, elapsed **8 min 17 s**, before final commit, merge to main, `adaptive-dog-v1` tag, push and GPU synchronization. New policy/video identities and known limitations are frozen in `docs/locomotion/OFFICIAL_V1.json`.

## README presentation order

- **2026-09-11 05:38:06–05:38:18 UTC**: moved adaptive dog to the end of the demo list, retaining Learning the Wind first. Section contents and links are identical; only order changed. Documentation-only validation passed; no training, simulation or rendering. Twelve seconds to the edit checkpoint, before commit/push/sync.
- Follow-up clarification: placed **Adaptive Dog last in the entire README**, after Run locally and Documentation. All section content remains unchanged.
- **2026-09-11 05:44:48–05:44:58 UTC**: changed the last demo heading to the user-approved “Adaptive walking — one policy, nine body configurations.” Documentation-only diff check passed; ten seconds to the edit checkpoint before commit/push/sync.
- **2026-09-11 14:38:48–14:39:08 UTC**: removed the temporary “Share the GIF” link from the root and locomotion READMEs at the user's request. The GIF file remains available. Documentation-only edit; twenty seconds to the edit checkpoint before commit/push/sync.


## Follow-up: matched CPU, Apple MPS and RTX 4090 comparison

- Started **2026-09-11 17:02:51 UTC** following the request to identify the training hardware and compare GPU training. The approved dog used **CPU MuJoCo/mjbatch physics and CPU actor inference, with Apple MPS learner updates**. Created `experiment/adaptive-dog-gpu-comparison`; main and all official v1 checkpoints/media remain unchanged.
- Predeclared three 90-second continuations from the same archived parent, seed 2, frozen healthy-motion bank, 512 environments, 16 physics threads and identical reward/optimizer recipe. All runs started from clean source commit `94b0dc0`, with identical initial parameter hashes. CPU/MPS/CUDA forward, critic and gradient agreement checks passed before learning. Added an optional validated frozen-reference loader; ordinary training defaults are unchanged.
- Measured training **89.398 + 89.406 + 89.971 = 268.776 seconds (4 min 29 s)**, totaling **3,747,840 transitions**. Mac M3 Max/MPS: **20,480 transitions/s**; desktop Ryzen 5950X CPU: **9,483/s**; same desktop RTX 4090 CUDA: **11,882/s**. CUDA improves same-desktop throughput **25.3%**; Mac/MPS remains **72.4% faster overall**. Physics remained on CPU throughout. The other existing GPU workload was left running and load telemetry retained; these are not exclusive-GPU measurements.
- Final checkpoints only, eight matched trials per body across all nine bodies, seed 9217 and 0.5 ms physics: Mac/MPS **72/72**, desktop CPU **68/72**, CUDA **56/72** full tasks. CPU misses four whole-FR goals; CUDA misses every lower-FR and whole-FR goal. All attempts and failed gait gates are retained. One training seed and different update counts do not establish intrinsic backend learning quality. No policy promotion, extra tuning or final-holdout reuse.
- **85 tests passed**, Ruff passed, and the actual CUDA-trained checkpoint passes a five-second native Mac viewer launch. Three-column, three-row video: **1920×1440, 25 fps, 12 seconds at 1×**. Captured physical trajectories once (**10.382 s**) and rendered (**27.368 s**); a status-label-only replay of identical hashed states rendered in **27.728 s** (reload/verification **0.793 s**). No training for recording. All 300 final frames decode; opening/middle/ending frames inspected. Original torque caps hold. Lower-FR sampled penetration exceeds the unchanged 8 mm target for MPS (**8.889 mm**) and CUDA (**8.217 mm**); these failures are documented, not relabeled as passes.
- Verified result/media milestone **2026-09-11 17:26:54 UTC**, elapsed **24 min 03 s** before final commit/push/GPU worktree synchronization. Final video SHA-256 `d4ee3b3c031ede036095cfe5ff59e340cc5d0e8a28e6303e22f6f193aceb2ea4`. Checkpoints/media use Git LFS; reports contain public-safe provenance. Full recipe, results, limits and viewer command: `docs/locomotion/GPU_COMPARISON.md`.

## Follow-up: optional MuJoCo Warp physics on the RTX 4090

- Started **2026-09-11 17:32:50 UTC**, the first retained clock checkpoint for the user's request to add a separate GPU-physics path. Work remains on `experiment/adaptive-dog-gpu-comparison`. Added optional MuJoCo Warp **3.13.0** / Warp **1.17.0** uv dependencies and `--physics-backend warp`, independent of `--device cuda`. CPU mjbatch stays the default. All nine actual body topologies, physical geometry, torques, sensors and clocks remain; NumPy observations/rewards/reference matching still execute on CPU.
- Initial CUDA validation failed during kernel loading after **134.84 s**, including cold compilation. A 256-thread CCD workaround passed ten tests in **149.00 s** but yielded poor throughput. A narrower version-scoped module-default fix passed in **95.31 s**, with speed still poor. Both incomplete matrices were intentionally stopped; completed results and failed-attempt descriptions remain archived. No learner ran during these investigations.
- Three uncaptured GPU steps attributed **97.9%** of kernel time to rangefinder rays. Added a camera-free BVH query path using the upstream ray API, keeping all sensors and geometry. **19 GPU checks passed in 20.77 s**, including 216 randomized poses across nine bodies matching CPU ranges within 0.1 mm. Bounded-plane and upstream capsule–cylinder contact limitations are documented. Mac package/mjbatch suite: **86 passed, 18 CUDA checks skipped**; those 18 passed on NVIDIA. Ruff and the five-second native Mac viewer passed.
- Completed three-repeat CPU/Warp matrices for healthy/all-nine families at 512/4096 worlds. All-nine 4096-world physics **18,008 → 82,740 control intervals/s (4.59×)**; including environment bridge **15,476 → 55,194/s (3.57×)**. The smaller nine-body GPU split remains slower than CPU. These are warm standing-command measurements, not complete training rates. Nine-body GPU setup was **4.696 s** with compiled kernels. Frozen official v1 also completed **8/8 GPU-physics missions**, twelve simulated seconds each at 0.5 ms; mission check wall time including setup **28.744 s**.
- One predeclared **90-second** continuation at 4096 worlds used **90.916 s actual training**, **5.823 s setup**, **96.815 s process wall time**, **24 updates** and **2,359,296 transitions** (**25,950/s**). Same archived parent/reference/rewards/seed 2 as the earlier comparison; source clean `5d20223`. Changing 512 → 4096 worlds makes this a scaling trial, not an isolated hardware comparison. No additional tuning followed. Selected ancestry **1924.074 s** includes prior training; only **90.916 s** is new training here.
- The other GPU workload was left running. Total sampled VRAM peaked at **13,432 MiB** including that workload; utilization ranged **14–90%**. No memory failure. The user's offer to reduce load was not required for capacity; competing load limits timing repeatability.
- CPU validation at 0.5 ms, eight trials per body, seed 9217: **65/72 full goals**, **72/72 upright with valid support**. Whole-FR passes 1/8 and misses the compound progress/lane/one-second dwell goal seven times. All eight other healthy/visible-step/stride/stance/speed/body-motion gates pass. Evaluation correctly exits 1; official v1 is unchanged, and every failed trial is retained.
- New experimental nine-case video: **3840×2160, 25 fps, twelve seconds at 1×**; GPU-trained weights executed in CPU MuJoCo. Capture **9.589 s**, render/export **40.589 s**, excluding renderer setup. All **300 frames** decoded; opening/middle/ending inspected. The whole-FR demonstration ends explicitly **INCOMPLETE**. State/model/checkpoint hashes verified; all torque caps hold, maximum 20 ms sampled penetration **7.772 mm < 8 mm**. No extra training for video.
- Verified result/media milestone **2026-09-11 18:17:37 UTC**, elapsed **44 min 47 s**, before final documentation commit/push/GPU synchronization. Video SHA-256 `e8aa78324f83919a6a23fed4a800412a697428fcf2a863b54b55421f02d8146e`. Checkpoint and media use Git LFS. Full measurements, failed attempts, limitations and viewer command: `docs/locomotion/WARP_BACKEND.md`.

## Follow-up: maintain the same adaptive-dog learning recipe on Warp

- New goal started **2026-09-11 18:27:44 UTC**: retain network/rewards, optimize Warp scheduling, match CPU learning conditions and behavior, compare throughput and equal-time learning, and deliver reproducible code/checkpoints/video. Work stays on the comparison branch; official v1 and main remain unchanged. Added explicit minibatch size, fixed PPO-round limits, optimizer-step accounting, concurrent streams for the nine body batches, and a maintained `adaptive-dog learn` command.
- Initial paired 512-world runs: six final checkpoints, seeds 2/3/4, each 48 rounds / 768 Adam steps / 589,824 transitions. **208.692 seconds new training**; **1.710×** pooled Warp speedup. CPU **198/216**, Warp **183/216** development task completions; failed completion and gait gates are retained. No reward, architecture or physical-model changes.
- Frozen-policy transfer diagnostic: **64 trials**, **68.177 seconds wall time**, no training. The same weights produce very similar distances on CPU and Warp at both 2 ms and 0.5 ms; weakness persists on the training engine. Archived parent and official v1 also evaluated without training: **71/72** and **72/72** tasks. This supports investigating continuation drift, not claiming exact engine equivalence.
- Second paired matrix: 4096 worlds on both engines, six rounds, same total samples/optimizer steps, parent, learning rate and three seeds. **179.093 seconds new training**; CPU **136.772 s**, Warp **42.321 s**, **3.232×** pooled speedup. CPU **200/216**, Warp **205/216**; all 432 trials upright with allowed support and all predeclared backend/gait-retention checks pass. Results cover short continuation of a trained walker, not a full curriculum from scratch.
- Equal 90-second budgets, seed 2 and 4096 worlds: CPU **89.219 s**, 11 rounds / 1,408 Adam steps / 1,081,344 samples; Warp **90.406 s**, 40 rounds / 5,120 Adam steps / 3,932,160 samples. **179.625 seconds new training** combined. Warp processes **3.636×** as much experience but scores **59/72** vs CPU **70/72**; all survive with valid support. The regression is retained and this longer continuation is not promoted. One additional bounded CPU control at exactly 40 rounds was started **2026-09-11 19:16:55 UTC**, to separate update-count drift from a backend-specific gap.
- The first continuing-trajectory scheduling test failed; its result and serial-repeat diagnostic remain documented. The revised test uses 960 fresh one-control-interval samples, explicit local numerical tolerances and exact unreset-state preservation. **23 GPU-host tests passed** in **23.66 s**; Mac package/mjbatch suite **89 passed, 19 CUDA checks skipped** in **6.49 s**. Existing warnings and GPU nondeterminism remain explicit. Maintained CLI smoke tests add **0.120 s CPU + 0.946 s Warp** training, excluded from performance claims. Native Mac viewer launch with the matched Warp checkpoint passed for five seconds.
- Predetermined seed-2 comparison video captures all nine bodies in three chapters, both evaluated in CPU MuJoCo at 0.5 ms. Capture **19.776 s**, render **61.641 s**, excluding renderer setup; no new training for video. **1920×1440 / 25 fps / 36 s / 1×**; all **900 frames** decoded, chapter openings/midpoints/endings/transitions inspected. All torque caps hold and both whole-FR goal failures are shown. One sampled penetration remains **8.090 mm > 8 mm**, so physical video QA retains that failed check. Video SHA-256 `6e2f700370598970b571266c9ebf210e86ac6059a3e39606d9b50baf974e08e6`; official v1 checkpoint/video hashes verified unchanged.
- The longer CPU control completed all 40 rounds / 5,120 Adam steps / 3,932,160 transitions in **301.874 s**, versus the existing Warp **90.406 s** (**3.339×**). CPU **60/72**, Warp **59/72**; all upright with allowed support and all gait-retention comparisons pass. Initial weights, learning settings and implementation hashes match. This supports shared longer-training drift; the different per-body failures and single-seed scope remain explicit. No further training or policy promotion followed.
- Total this goal: **15 comparison training runs, 869.284 seconds (14 min 29 s)**, plus **1.066 seconds** for two actual maintained-CLI smoke runs. All failed attempts and all final checkpoints are retained. Training ancestry, setup, validation, recording and elapsed implementation time remain separate.
- Verified implementation/results/media milestone **2026-09-11 19:24:53 UTC**, elapsed **57 min 09 s**, before final documentation commit/push/GPU synchronization. Ruff and formatting checks pass for all 76 Python files; Git LFS integrity passes. Full recipe, matched and equal-time results, numerical limits, provenance and viewer command: `docs/locomotion/WARP_TRAINING.md`.

### Adaptive standing and walking on physical supports

- Goal start **2026-09-11 23:18:31 UTC**. A new `feature/adaptive-standing` branch preserves main, the accepted walking weights/video and the `adaptive-dog-v1` tag. User extension retained all gentle scenes and added high/extreme pads, 12/18/24° slopes on both axes and 12/20/28 cm steps: **19 surfaces** total. No additional IMU model; existing body-state feedback remains alongside real downward rays and terminal-contact measurements.
- **23:41:35 UTC**: gentle/aggressive static previews, physical reset/range checks and shared-policy training interface implemented. Frozen walking v1 passes **0/18** initial gentle standing/transition checks; every failure remains recorded. Reset-only IK makes initial support poses feasible; there is no live pose correction or hidden support.
- Seven candidate rounds on the RTX 4090 used **536.734 s (8 min 57 s)** of new learning and **26,836,992 transitions**, in nominal 60/60/60/60/60/120/120-second budgets. All attempted checkpoints/configurations/results are preserved. Selected ancestry is **2,459.460 s (40 min 59 s)**, including the inherited **1,922.725 s** walker. Setup, rehearsal collection, diagnostic one-update tests, evaluation and rendering are excluded from the new-training claim.
- Initial healthy-only learning drifted away from damaged walking. Mixed nine-body flat training, 18 healthy terrain groups and training-only v1 walking rehearsal restored every walking-retention gate. A contact-window kernel then accumulated reactions at every 2 ms physics step inside Warp graphs so the reward could detect brief unintended support; the final round restored longer standing episodes. The deployed actor remains one 86→128→128→12 network, with a separate 90→128→128→1 training critic and 20 ms commands.
- Checkpoint selected before final seed 9307 and video seed 9311. Four trials per case: CPU **48/80** healthy standing/transition and **46/64** damaged flat checks; Warp **48/80** and **48/64**. All **144 trials per backend** survived and respected torque caps, but strict support/drift/tilt gates retain failures. Some damaged transitions exceed 8 mm sampled penetration: maxima **10.318 mm CPU / 9.492 mm Warp**. Both front missing-support cases maintain their designated foot unsupported for the full measured hold. Rear gaps and extreme surfaces remain limited.
- Independent original walking gates at 0.5 ms: **36/36** tasks plus every visible-step, stride/stance, speed and body-motion retention gate pass. Mac suite **73 passed / 24 CUDA-only skipped**, NVIDIA suite **97 passed**. Ruff lint/format passes across 86 files. Selected checkpoint passed the native Mac viewer. Existing upstream struct-deprecation and capsule–cylinder multicontact warnings remain documented.
- Full video captures **17 predetermined cases**, with no resets and all failures retained, at **1×, 1920×1080, 25 fps, 45 s**. Original physical capture/save **45.154 s**, initial render **33.059 s**. Visual QA caught head-camera occlusion in the preliminary smoke (fixed before full capture) and overlapping failure labels in the first full render. Final v2 replays the exact hashed trajectories: load/verify/save **16.711 s**, render/encode **33.637 s**, zero new simulation or learning. **1,125 frames** decoded, declared chapter/interaction/end samples inspected. The far-side lower-FL damage marker remains partly occluded; status labels and other damage markers are readable. **10/17** recorded task cases pass; torque caps hold and maximum sampled penetration is **3.495 mm**.
- A slow direct Git LFS upload was stopped and retried through a temporary loopback-only proxy over the existing authenticated connection. The retry succeeded; no media was copied outside Git/LFS and no permanent network settings changed.
- Verified implementation/results/video milestone **2026-09-12 00:41:58 UTC**, elapsed **1 h 23 min 27 s**, before final documentation commit/push and Mac/GPU synchronization. Video SHA-256 `9a748238a2456d6eab6176832883847f1c91d05a4c83441575394d645103711d`. Local Git LFS integrity and frozen v1 hashes pass. Run instructions, masses/joint limits/torques, raw metrics, assumptions and known failures: `docs/locomotion/standing/README.md`.
- Completion audit **2026-09-12 00:44:57 UTC**, elapsed **1 h 26 min 26 s**: implementation commit `347f2fa` is synced on both machines, working trees clean, final checkpoint/video and frozen v1 hashes match, Git LFS integrity passes on both hosts, and optimized replay verifies all 17 models/trajectories (8,600 control states) without simulation. Temporary upload tunnel closed. Only this audit-note commit follows.

### Diagnostic video of standing failures

- Follow-up started **2026-09-12 01:35:54 UTC** at the user's request to inspect failures before deciding whether they need correction. Kept the frozen standing checkpoint, original videos, dynamics, rewards and acceptance thresholds. **Zero new training.**
- Selected the first failed trial in every failing CPU holdout condition: **15 conditions**. Repeated the original four-world batches and seed 9307, recording the selected world. Every selected result row matches the original exactly; all 15 remain upright and respect torque caps. Support/drift/tilt/airborne-foot/penetration failures remain explicitly flagged. Maximum sampled penetration remains **10.318 mm**, exceeding the original 8 mm gate.
- Capture/save took **10.966 s** for all batches. Exact model binaries and trajectories are cached separately from presentation. Full trial sequences total **160 simulated seconds** in the selected worlds. Final video adds eleven labeled two-second peak-inspection holds: **182 s, 1920×1080, 25 fps**, following/overhead/detail views. Final render/encode took **186.962 s**. Early diagnostic stills and implementation/inspection time are separate from these timings.
- Every **4,550 encoded frames** decoded; all 15 chapter inspection samples plus full-resolution opening, settled drift, brief link contact, penetration peak and ending were reviewed. Measured thresholds remain readable, peak pauses are labeled, and markers identify removed limbs and flagged links. Ruff lint/format and diff checks pass. Logging-only evaluator changes were checked through exact agreement with all selected original holdout rows; no training was repeated.
- Verified video/docs milestone **2026-09-12 01:50:54 UTC**, elapsed **15 min 00 s**, before final commit/push and GPU worktree synchronization. Video SHA-256 `d172fb839cff21403da04d3be6105165421cd4e6bd8ece571ee9ad40e4e74fc9`. Viewer, chapter list and reproducible recording commands: `docs/locomotion/standing/FAILURE_REVIEW.md`.
- Completion audit **2026-09-12 01:52:33 UTC**, elapsed **16 min 39 s**: media commit `b22a433` pushed and synchronized through Git/LFS; both worktrees clean, video/checkpoint hashes match, LFS integrity passes on both hosts, frozen v1 and prior standing artifacts unchanged. Temporary transfer tunnel closed. Only this timing-audit commit follows.

### Accepted terrain footage and matched friction comparison

- Follow-up started **2026-09-12 02:01:40 UTC**. The user accepted the eight terrain conditions before the damaged section of the failure review, described them as natural/organic, and requested their inclusion in the results film. They suggested greater foot friction and then explicitly included terrain/support friction. Visual acceptance is recorded separately from unchanged original quantitative gates.
- New v3 is **67 s, 1920×1080, 25 fps**. It keeps all 42 s of original v2 physical footage, inserts two full ten-second four-panel sections from the exact accepted terrain traces, and uses a five-second results card. **25/25** recorded trials remain upright; **10/25** pass every original gate. These are 25 trials across 19 distinct conditions, with reviewed cases explicitly selected after evaluation. No new physics or learning for the edit. Render/encode **33.001 s**; all **1,675 frames** decode. Encoded opening, both added terrain groups, transition back to damaged bodies and final card were inspected. Prior checkpoint/video hashes remain unchanged.
- Replay kinematics measured **0.25–1.25 cm/s** mean loaded-contact-point slip across the eight terrain cases. Both original feet and terrain use sliding friction **0.8**. A separate frozen-policy CPU MuJoCo/mjbatch comparison set both to **0.8 / 1.0 / 1.2**, keeping other coefficients, solver, state initialization, commands, weights and thresholds unchanged. Eight reviewed conditions plus three passing controls, four paired trials each: **132 total trials**, **1,344 aggregate simulated world-seconds**, **23.742 s** final comparison/capture wall time. **Zero new training.**
- All 44 baseline result rows exactly reproduce the original CPU report. Reviewed-terrain strict passes **0/32 → 0/32 → 1/32**; average peak drift **12.33 → 12.43 → 12.09 cm**. All 132 trials remain upright and respect torques; all passing controls stay **12/12** for each setting. No higher-friction default is promoted; original friction is retained in the accepted video.
- A baseline assertion caught an omitted transition argument in the first comparison script: its flat control stood instead of executing walk–stop–walk. Fixed in `8c12c70`, then repeated the full matched sweep. That discarded setup used **368 additional aggregate simulated world-seconds**; its separate wall time was not recorded. All final comparison rows and hashes are retained.
- Focused standing checks **22 passed / 5 CUDA-only skipped / one existing training smoke deselected**. New test verifies foot/terrain friction inside the batch and actual contact records while reset states, torque caps and other friction dimensions stay identical. Ruff and diff checks pass. Video/report/source milestone **2026-09-12 02:15:16 UTC**, elapsed **13 min 36 s**, before commit/push and GPU sync. V3 SHA-256 `5ba3253ebaef8e265196bd6198444c9deddb13227434626d21dcbabaab59a986`. Details: `docs/locomotion/standing/RESULTS_VIDEO_V3.md` and `TERRAIN_FRICTION.md`.
- Completion audit **2026-09-12 02:17:38 UTC**, elapsed **15 min 58 s**: media/results commit `ed8a0a0` pushed and synchronized through Git/LFS, matching video/checkpoint hashes and clean worktrees on both machines; LFS integrity passes. Main and archived artifacts remain unchanged. Temporary transfer tunnel closed. Only this timing-audit commit follows.

### Clearer shared-policy caption

- **2026-09-12 02:22:31–02:24:47 UTC**, **2 min 16 s** to verified render, before media commit/push/sync. Replaced the closing card's backend/footage sentence with the user-approved wording: “One trained policy across different bodies and terrain. No retraining between scenes.” Updated the existing v3 video path; earlier bytes remain in Git LFS history.
- Re-render/encode **33.109 s**, **zero new simulation or learning**. All scene/trajectory hashes, 25 trial results, chapter timing, dimensions, frame rate and duration match the previous v3. All **1,675 frames** decode; the last physical frame, card transition and final encoded card were inspected. Ruff/diff checks pass; no physics or training tests rerun for this caption-only edit. Video SHA-256 `61c2c33b734273427ea34c41d321f5434f3352306a061aef67a835c6aa2ee070`.

### Standing release and moving-environment goal

- Work began **2026-09-12 02:37:25 UTC**; explicit moving-environment goal registered **02:37:55 UTC**. Preserve accepted static-standing/Warp work on main before a new experiment branch. Release manifest pins unchanged checkpoint and latest user-approved 67-second video. Merge verification: **74 Mac tests passed / 24 CUDA-only skipped; 98 NVIDIA tests passed**. Existing upstream warnings retained.
- **02:43:41 UTC**: accepted standing merge commit created, then pushed as `d652f8f`; `adaptive-standing-v1` release tag pushed. GPU main worktree synchronized. New branch `feature/moving-supports` starts from that release. Static moving-deck preview compiled and inspected before live motion; first CPU/Warp checks and native Mac viewer pass.
- Moving development seed 9401: frozen original/relative-input policies each **18/24** strict passes, **24/24** upright. First GPU round **59.201 s**, **3,145,728 transitions**, improves moving checks to **24/24** but loses upright behavior on two omitted static conditions. All original walking gates still pass. Preserve this failed retention candidate.
- Corrective GPU round **59.353 s**, **2,359,296 transitions**: include all original static surfaces and training-only frozen-standing rehearsal, weight 50, while keeping walking rehearsal weight 15. New learning so far **118.554 s**, excluding CPU one-update interface probes and dataset/setup/compilation. Candidate 2 restores **80/80** healthy static upright trials and **48/80** strict passes, matching the accepted parent; moving development **23/24** strict, all upright. Further retention and holdout checks follow.
- Candidate selected **2026-09-12 03:06:56 UTC** before new seeds. All **36/36 walking tasks and original gait gates** pass. Matched static development: **144/144 upright**, strict **100/144** versus parent **98/144**. Held-out seed 9407 agrees on CPU/Warp: frozen original/relative-only **16/24**, trained **22/24**, all healthy trials upright. Two trained combined-motion trials use unintended link support. Untrained damaged moving transfer deliberately remains **0/8 strict, 6/8 surviving** per backend; no further tuning after this holdout.
- Video verified **2026-09-12 03:17:26 UTC**, elapsed **40 min 01 s** from work start, before final docs/commit/sync. **64 s, 1920×1080, 25 fps, 1,600 frames, 1×**; five three-way motion comparisons plus explicitly labeled synchronized three-view replay and result card. Warp capture/save **50.806 s** (600 aggregate simulated world-seconds); render/encode **40.214 s**. Video-seed strict results **15/20 original, 14/20 relative-input-only, 20/20 trained**, all 60 upright. Encoded chapter/interaction/camera/end samples inspected; complete decode passes. Video SHA-256 `aa4290c5691e2938cecf18d47037e7ce7aa2d59567340293b729d5574d1f1c15`.
- Final suites: **78 Mac passes / 25 CUDA-only skips; 103 NVIDIA passes**. Existing upstream warnings retained. Native Mac viewer and lint/format pass. Two one-update CPU pipeline probes used **0.304 s and 0.588 s** optimization time; these are not part of selected training ancestry. Frozen walking and standing checkpoint/video hashes remain unchanged.
- Final physics/media audit milestone **2026-09-12 03:21:39 UTC**, elapsed **44 min 14 s**: all **15 saved models and trajectories (7,500 control states)** verify on the capture host with finite states, complete ten-second timelines and matching hashes; no new simulation. Git LFS integrity passes on both hosts and release hashes remain unchanged. Source/results/video commit `414e773` and trace audit `4608956` are pushed. Final timing-note commit and branch synchronization follow; accepted main remains `d652f8f`.

### Accepted moving-support merge and paused generative styling trial

- Merge follow-up began **2026-09-12 03:30:30 UTC**. Accepted moving-support work merged to main as `3526047`, tagged `adaptive-moving-v1`, pushed and synchronized to GPU main. Release manifest pins unchanged weights and the 64-second comparison. Previous walking/standing hashes and Git LFS integrity verified; existing validation reused.
- Generative-styling work began **03:34:01 UTC**, on `feature/cinematic-video-styling`. Added isolated Python 3.14.7 / WaveSpeed SDK 2.0.2 workflow, current model/schema research, durable submission records and a timestamp-matched comparison. Only the required API credential was installed in an ignored local environment file. Four submission-safety checks and lint/format passed.
- Exactly one five-second source clip submitted **03:40:34.855 UTC**; completion observed **03:44:12.068 UTC**. Provider inference **215.120 s**, upload **0.684 s**, verified debit **$1.98**. Returned 113 frames / 24 fps / 4.708 s; common-interval comparison 117 frames / 25 fps / 4.68 s. Both decode completely and opening/intermediate/end samples were inspected. Improved material/environment appearance accompanies fine geometry/motion drift and a shorter duration. No new physics or learning.
- At **04:00:44 UTC**, user requested pausing generative work and improving native MuJoCo presentation. Existing trial saved without another generation. The previously requested combined standing/moving edit remains deferred.

### Native graphite presentation after pausing generative styling

- Follow-up started **2026-09-12 04:00:44 UTC**. The accepted moving-support result had been merged to main as `3526047` and tagged `adaptive-moving-v1`. Generative styling was paused and its single reviewed trial archived on `feature/cinematic-video-styling` at `c1f4e82`; no additional generation was submitted. The combined standing/moving compilation remains deferred. Native presentation starts separately on `feature/graphite-presentation` from accepted main.
- Found that visible branding is raised/recessed mesh geometry; hiding the white letters alone left letter-shaped recesses. Initial rectangular cover prototypes protruded from the shell, so the final thin observer-only covers follow the body's curvature and cover both sides plus forward wordmarks. Vendor files, physical geometry and policy inputs stay unchanged.
- Added neutral metals, graphite floor, focused teal/orange accents, 4096-pixel shadows, 8× multisampling and restrained overlays. Static preview inspected before re-rendering motion. Opt-in `--theme graphite` works in walking, standing and moving live viewers; all three native Mac smoke checks completed four seconds each.
- Two invariance tests compare flat/moving models for 500 physics steps each: physical parameters, every pose, velocity and sensor reading match exactly. Both pass on Mac and Linux/NVIDIA. Mac full suite **80 passed / 25 CUDA-only skipped**, with existing upstream warnings; lint/format passes across 105 files.
- Ten-second film replays the exact accepted `combined_2` trajectory and model after verifying hashes. **Zero new physics steps or learning for recording**; GPU-host EGL render/encode **5.564 s**, separate from test/viewer simulation. **1920×1080, 25 fps, 250 frames, 1×** with three synchronized views. Full decode passes; opening/intermediate/camera/end samples inspected. Video SHA-256 `54e33d6a006d2a6da06adff476bd4cf7360fad710f4c2f071a127709f7c22f19`.
- Verified media/source milestone **2026-09-12 04:15:39 UTC**, elapsed **14 min 55 s**, before final documentation commit/push and branch synchronization. All three accepted checkpoint/video pairs remain unchanged, and Git LFS integrity passes. Run and playback paths: `docs/locomotion/graphite/README.md`.

### Complete adaptive-dog film in graphite

- First retained clock checkpoint for this follow-up: **2026-09-12 04:22:57 UTC**. The user requested one combined film, then explicitly requested **2× playback** with a visible label. The compilation covers the adaptive-dog learning progression; generative styling remains paused.
- Re-rendered **39 original recorded trials**: nine walking bodies, 17 standing cases, all eight previously accepted terrain-review trials and five trained platform motions. Each trace and original physical model passed hash verification before presentation changes. No new physical rollout or training was needed. Three successive frozen checkpoints are identified as three stages, with one shared actor within each stage.
- Final film: **65 s, 1920×1080, 25 fps, 1,625 frames**. The first **59 s** show **118 s** of source timelines at labeled 2×; a six-second results card follows. Actual simulated clocks, force-based contact dots, orange cut markers and prior quantitative misses remain visible. Observer, overhead and head cameras are synchronized on the main sequences.
- Final render/encode times: Mac walking **20.764 s**, Mac accepted terrain **13.414 s**, NVIDIA-host standing **18.157 s**, NVIDIA-host moving **7.825 s**; final Mac assembly/encode **7.652 s**. These are separate compute measurements. An initial walking render was replaced so the final render provenance correctly pins the committed script. Original films/checkpoints remain unchanged.
- Media/QA/docs milestone **2026-09-12 04:33:30 UTC**, **10 min 33 s after the first retained clock checkpoint**. Every encoded frame decodes; all 36 chapter start/mid/end samples were inspected, including all camera views, contacts, branding covers, transitions, playback labels and final card. Ruff/format checks and Git LFS integrity pass; three accepted checkpoint hashes and four prior movie hashes are unchanged. No dynamics tests repeated for this presentation-only edit.
- Final video SHA-256: `63c1792fe09663ea9a3124ee48706e49a2e6619a39bc6373fb42d1ed656012d3`. The combined movie, chapter media and reproduction/QA scripts are saved separately on `feature/graphite-presentation`; main and release tags remain unchanged. Playback and chapter guide: `docs/locomotion/graphite/COMPLETE_VIDEO.md`. Final commit/push and Mac/GPU synchronization follow this milestone.
- Completion audit **2026-09-12 04:35:29 UTC**, **12 min 32 s after the first retained clock checkpoint**: final media commit `4adff6d` is pushed; Mac and NVIDIA worktrees are clean and synchronized, the movie hash matches on both, and Git LFS integrity passes on both. The final movie was opened on the Mac. Only this timing-note commit and its synchronization follow.

### Restore normal-speed playback

- Follow-up started **2026-09-12 04:36:15 UTC**. User preferred 1× after reviewing the 2× compilation. Restored normal speed as the recorder default, added an explicit `--speed 1|2` option, and kept the 2× movie/chapters unchanged under their original `v1` names. New normal-speed media uses `v2`.
- Re-rendered from the same hashed physical states at **25 fps**, sampling each 40 ms instead of each 80 ms. All **39 source case records are identical** to the earlier edit. Same weights, scenes, cameras, graphite appearance and measured outcomes; **zero new training or physical rollout**. No stretching or frame duplication of the prior movie.
- New movie: **124 s (2 min 04 s), 1920×1080, 3,100 frames**; 118 s of physical footage at labeled 1×, plus the unchanged six-second results card. Render/encode: walking **38.956 s**, standing **31.811 s**, terrain **25.858 s**, moving **13.773 s**; assembly/encode **14.311 s**. These are separate wall-time measurements.
- Verification milestone **2026-09-12 04:40:09 UTC**, elapsed **3 min 54 s**, before final media commit/push and synchronization. Complete decode and format/coverage checks pass; all 36 chapter start/mid/end samples inspected. Ruff/format and LFS checks pass. README now points to normal speed, and the movie was opened on the Mac. Prior 2× movie hash is unchanged; all case, model, checkpoint and trajectory provenance matches the prior edit.
- New video SHA-256: `1878e7253818475a867f411f8ddccaf4c1423d507fcf077b58c4b0ae00aad89e`. Playback: `open previews/locomotion/graphite/adaptive_dog_complete_v2.mp4`.
- Completion audit **2026-09-12 04:41:53 UTC**, elapsed **5 min 38 s**: final media commit `7bba359` pushed and synchronized; Mac/GPU worktrees clean, matching movie hashes, and LFS integrity passes on both. Only this timing-note commit and its synchronization follow.

### Consolidate main and remove development branches

- Follow-up started **2026-09-12 04:44:21 UTC**. User requested merging the accepted work, pushing main, and removing all local/remote development branches. They then explicitly included the paused generative experiment in main for future use.
- Native presentation and the complete normal-speed film merged as `f9d3f3a`. The generative branch adds the isolated WaveSpeed SDK project, scripts, docs and reviewed media. Resolved only additive conflicts in simulation guidelines and the chronological time log, retaining both. Generative styling remains paused; no API submission or generation was made.
- Verification milestone **2026-09-12 04:48:00 UTC**: Mac adaptive suite **80 passed / 25 CUDA skips / two existing upstream warnings**; NVIDIA presentation invariance checks **2 passed**; generative workflow **4 offline tests passed**, with lint/format clean. Git LFS integrity passes and the final native/generated movie identities are unchanged. Credentials remain in ignored local `.env` files.
- Every branch tip will be verified reachable from main before removal. Former GPU experiment worktrees will remain detached at their original commits, retaining ignored raw traces, caches and environments. The three existing release tags remain unchanged. Branch history and future-work guidance are documented in `docs/BRANCH_ARCHIVE.md`; cleanup and final synchronization follow.
- Cleanup verified **2026-09-12 04:50:36 UTC**, elapsed **6 min 15 s**: generative merge `5b0671c` is pushed to origin/main and synchronized to GPU main. All six former origin branches and six Mac branches were removed after ancestry checks; five GPU branches were removed after their worktrees were detached without changing commits. Only main remains on both machines and origin. Six GPU worktree directories, all ignored caches and all three release tags are preserved. Both primary worktrees are clean; native/generated movie hashes and LFS integrity agree. Only this audit-note commit and final sync follow.

### Consolidated RTX 4090 training report

- First retained clock checkpoint **2026-09-12 05:05:29 UTC**; report verification **05:09:52 UTC**, **4 min 23 s** between those checkpoints, before commit/synchronization. User requested all adaptive-dog GPU statistics together and clarified that earlier Mac training must remain separate.
- Added `docs/locomotion/GPU_REPORT.md` and an exact JSON ledger, linked from the final adaptive-dog section of the README. Nine standing/moving GPU rounds total **655.288 s**, **32,342,016 transitions**, **179.678 aggregate simulation hours**, **49,355 transitions/s**, **329 recorded rollouts**, and **41,819 optimizer steps**. Separate walking benchmarks preserve the **3.23×** matched speedup, CPU evaluation provenance and unsuccessful longer continuations.
- Verified all nine source-report hashes, 4,096-world configurations, aggregate arithmetic, actor/critic parameter counts and local documentation links. Diff whitespace check passes. **Zero new simulation, training, rendering or generation**; existing acceptance reports are cited without rerunning physics tests. Earlier Mac ancestry, setup time, recorded-versus-discarded experience and failed task gates remain explicit.

### Simplify the GPU walking summary

- **2026-09-12 05:12:00–05:12:58 UTC**, **58 s** to verified documentation, before commit/synchronization. User requested ordinary totals instead of unexplained per-seed benchmark jargon. Walking now shows **42 seconds / 1.77 million experiences**, summed across the three independent benchmark runs; the other summary rows also use rounded totals. The detailed table uses run numbers, while original random seeds remain in the exact JSON/source records.
- Verified **1,769,472 experiences / 42.321 seconds / 9.83 aggregate simulation hours** from the existing measurements. Standing/moving totals remain separate; no new simulation, training or media changes. Diff check passes.

### Measured GPU acquisition and complete shared-policy curriculum

- Follow-up began **2026-09-12 05:14:54 UTC**. The prior 42-second row was misleading as walking acquisition: it summed three independent pretrained continuation benchmarks. User requested an actual timed replay of the successful curriculum on the RTX 4090, using **4,096 worlds throughout** and one final checkpoint for all commands. Work is isolated on `feature/gpu-walking-from-scratch`; accepted weights and videos remain unchanged.
- Two incorrect starts are retained separately. An improvised three-round healthy curriculum used **328.928 s** of learning; round 1 failed goal completion, round 2 completed tasks but failed established healthy gait gates, and round 3 was not adopted. A faithful-reward replay incorrectly used 512 worlds; it was stopped at **05:37:38 UTC** during stage 5. Its four completed stages used **157.969 s**, excluding the interrupted stage's partial work. Neither attempt supplies weights or references to the actual replay.
- The corrected run began **05:41:16.029 UTC**, source `a84b3f5`, clean checkout. Thirty stages replay the recorded 21-stage walking ancestry, seven standing stages and two moving-support stages. Every stage uses 4,096 worlds, 24-step rollouts, four epochs and minibatches of 3,072. Walking sample counts are rounded up to complete GPU rollouts; planned total **71,761,920 transitions**. Earlier GPU-trained checkpoints in this same run supply all training references and rehearsal data. No historical trained weights are imported.
- Healthy walking milestone, stage 5: **113.797 s cumulative GPU learning**, **8/8** development tasks completed with allowed support, all original healthy gait gates passed (mean stride **29.49 cm**, speed **0.602 m/s**, duty gap **7.15 percentage points**, height variation **2.14 mm**). The later all-foot visible-lift targets did not pass yet and were not reclassified as healthy acquisition gates. Evaluation used CPU MuJoCo at 0.5 ms, seed 9501, and **4.020 s** evaluation wall time.
- User requested avoiding unnecessary checks. Planned remaining checks are one four-trial-per-body walking-retention audit after walking and after standing, then one combined final audit with the same final weights across all command families. The larger evaluation counts in the initial plan are superseded. CPU checks of copied checkpoints can run on the Mac while GPU learning continues without changing the training checkout or curriculum.
- GPU run completed **06:07:52.847 UTC**: all 30 stages, **1,325.012506 s (22m 05s) learning**, **71,761,920 transitions**, **730 rollouts**, **93,440 optimizer steps** and **4,096 worlds throughout**. Setup **187.827 s** is separate; process span **1,596.818 s (26m 37s)**. Source and recipe hash remained unchanged. Device-synchronized training time includes the archived rehearsal losses; standalone walking checks and offline reference collection do not enter that timer.
- Final audit results observed by **06:09:51 UTC**: the same final checkpoint passes **36/36 walking tasks**, all original healthy gait gates, **136/144 strict static checks**, and **24/24 moving-support checks**; all **204/204** task trials remain upright. CPU audit wall time **50.103 s**, with zero training. The eight static failures are retained: four steep-slope unintended-support cases, one damaged transition with unintended support, and three damaged transitions exceeding the drift threshold. Both intermediate walking audits pass **36/36** tasks and healthy gait gates. No further tuning follows these results.
- Saved initial, healthy, walking, standing and final checkpoints, all 30 raw training records, raw evaluation reports and a verified timing ledger. The final shared checkpoint hash is `076099ef8fc47bad53cb0cbeb9cc311e64ea47accacfa8ca4080488934b10241`. The GPU report now describes measured acquisition; the prior continuation report is archived separately. Existing accepted policy/video artifacts remain unchanged. Final artifact commit and synchronization follow alongside the user's subsequent presentation request.

### Ember machinery palette study

- First retained clock **2026-09-12 06:11:29 UTC**. User supplied the Golden Ember charcoal/steel/ivory/yellow/orange palette and requested a compatible red or green. Added restrained forest green `#3F6B53` for contact indicators and oxide red `#C64B3C` for damage markers. This is optional presentation work after the completed GPU replay, with no training changes.
- Static **1920×1200** preview rendered and inspected: yellow cosmetic shell panels, steel leg members, dark chassis/ground, red cut markers and a neutral support deck with yellow edge decoration. All decorations remain in the observer scene. The classic/graphite defaults and original vendor assets remain preserved. Four focused flat/moving presentation checks pass with identical physical states and sensor readings; no policy retraining or broad evaluation rerun.
- Presentation checkpoint **06:23:47 UTC**, **12m 18s** after the first palette clock: preview and theme committed as `4d2b307`; the measured GPU result is separately committed as `6a51653`. The native Mac viewer completes a three-second check with the final unified actor. Source lint/format and local Git LFS integrity pass. Training, reports and palette are pushed on the experiment branch; accepted main remains unchanged. Final timing-note synchronization follows.

### Complete Ember film and concise measured GPU summary

- Follow-up began **2026-09-12 06:27:00 UTC**. User requested the earlier summary/report style with useful scale statistics and a final film in the new machinery palette. The brief fixes all 51 conditions and demonstration seed 9511 before capture. Every run uses final actor `076099ef8fc47bad53cb0cbeb9cc311e64ea47accacfa8ca4080488934b10241`; no intermediate policy substitution, training, reward or curriculum changes.
- Fresh native CPU MuJoCo/mjbatch capture/save takes **33.769 s**, recording **546 aggregate world-seconds / 27,300 control states**. All **51/51 trials stay upright**, with **49/51 strict/task passes**. Retained misses: unintended link support on the 24° fore/aft slope and 17.2 cm drift in a rear-left lower-leg transition hold. These demonstration trials are separate from the final 204-trial held-out audit. Walking uses 0.5 ms physics, balance uses 2 ms; control remains 20 ms.
- Final native Ember render/encode takes **184.404 s**. The film is **174 s / 1920×1080 / 25 fps / 4,350 frames / 1×**, covering every condition once, plus training-scale and held-out-results cards. Source renderer `6cf3a30`; capture source `be0ee27`. Video SHA-256 `229eb37952477abcbe960376c2fcf033a152d91b108ca3e6849bc9253345b9ca`. The previous graphite film and accepted releases remain unchanged.
- Verified media milestone **06:44:32 UTC**, **17m 32s** from follow-up start, before final documentation/commit/synchronization. Full decode and model/trace/checkpoint/timeline/case-coverage checks pass in **12.713 s**. Inspected 63 encoded samples spanning opening/midpoint/end, command transitions, terrain contact, moving platforms and result cards, plus full-resolution examples. No additional simulation or broad policy audit for visual QA. Shareable cards and `GPU_SUMMARY.md` use the measured **22m 05s / 4,096 worlds / 71.76 million experiences**; production time stays separate.

### Industrial floor and platform detail study

- First retained clock **2026-09-12 06:46:21 UTC**. User liked the Ember film and requested richer, consistent hazard/detail treatment for the environment, then explicitly requested a static floor/platform preview before continuing. Video and live theme stay unchanged while a separate `ember_environment` candidate is reviewed.
- Added flush hazard paint, inset steel plates, corner fasteners, dark side panels, floor seams, lane/distance markings and a ribbed background shell. Six environment-only views cover flat floor, moving deck, extreme pads, a 24° slope, 20 cm steps and missing support. The robot is hidden only in the preview. **Zero physics steps, recapture or training**; all six original physical-model/sensor configurations remain identical. Static imagery rendered and visually inspected before handoff. Complete film/report commit `45b80b1` is already pushed and synchronized to the GPU checkout.

### Approved industrial environment film

- Follow-up began **2026-09-12 06:54:50 UTC**, after the user approved the six-view static environment study and requested a new video. Added the recorder's explicit `--environment industrial` selection while preserving `simple` and the original v1 film. Source renderer commit `ad474c3` uses the unchanged saved 51-run capture and final shared actor; **zero new training or physical capture**. Robot-in-scene previews of walking, pads/steps and the moving platform were inspected before full rendering.
- Verified media milestone **07:00:31 UTC**, **5m 41s** after approval follow-up started, before final commit/synchronization. Industrial v2 render/encode **222.843 s**; full decode/trace/coverage/preservation QA **13.195 s**. Film remains **174 s / 1920×1080 / 25 fps / 4,350 frames / 1×**. All 63 encoded chapter/transition samples inspected, with unchanged 51 trial outcomes and exact prior physical-run identities. Video SHA-256 `6dc252963ccb3bc58553c79e30d83bae4dd2ffbecba907aafbe4ce1e346afd6d`. The original movie and stats images remain unchanged. User subsequently requested a separate half-length edit at normal playback speed.

### Half-length edit at normal speed

- First retained editing clock **2026-09-12 07:02:30 UTC**; verified media/documentation milestone **07:09:29 UTC**, **6m 59s** between those checkpoints, before final commit/synchronization. The user requested halving every segment while preserving 1× motion and the complete prior film. New alternate `adaptive_dog_short_v1.mp4` is **87 s / 1920×1080 / 25 fps / 2,175 frames**, with all 17 chapters halved and all 51 conditions represented.
- Trimmed continuous intervals from the encoded industrial v2 film. Static/walking chapters keep their first half; moving-platform chapters keep their middle half. Walk–hold–walk excerpts use source 2.52–8.52 s to show both command changes at 3 s and 8 s. Both result cards last three seconds. Outcome labels still describe the original complete trials. **Zero new training, physical capture or MuJoCo rendering**; same frozen actor, trajectories and camera synchronization.
- Final edit/encode **6.319 s**, complete decode/format/coverage QA **7.160 s**. An earlier draft (**6.322 s** encoding) began the transition excerpts after the stop command; visual review caught this and the retained final edit corrects the trim. All 51 final opening/middle/end samples were inspected across six contact sheets. Video SHA-256 `69f7a93c151cdf637c4110734576bc4b13bd430e23b273966d535474415428ca`. Full industrial v2 and older movies remain unchanged; reproduction and exact source selections are in `docs/locomotion/ember/SHORT_VIDEO.md`.

### Consolidate GPU curriculum and Ember films on main

- Follow-up began **2026-09-12 08:15:23 UTC**. User requested merging to main, pushing origin and removing local/remote development branches. Merged `feature/gpu-walking-from-scratch` tip `8c131ca` without conflicts as `897456e`, then updated the branch archive and film links in `0150c3d`. All implementation, checkpoint and preview files match the finished branch exactly; no new simulation, training or rendering.
- Cleanup verified **08:17:55 UTC**, **2m 32s** after the first retained clock. Main is pushed and synchronized to the primary GPU checkout. Only `main` remains on the Mac, GPU machine and origin; development branch ancestry was checked before deletion. The GPU capture worktree is detached at its original `8c131ca`, preserving its ignored runs, caches and environment. All previous detached worktrees and three release tags remain unchanged.
- Both primary worktrees are clean, Git LFS integrity passes on both machines, and the final shared actor plus full/short film hashes match their verified originals. This timing-note commit and synchronization follow the cleanup milestone; no repeated policy evaluation was needed for a conflict-free merge of the already verified artifacts.

## October 5: surgical robot and PufferLib 5.0 project research

- Observed research/documentation interval **17:16:01–17:23:48 UTC**:
  **7 minutes 47 seconds elapsed**. Simulation **0 s**, training **0 s**,
  rendering **0 s**; this interval is research and proposal preparation.
- Inspected repository scene/viewer/batch patterns and primary upstream sources.
  Recommended a dVRK Si dual-arm MuJoCo port, then needle handoff/ring passage,
  with thread and compliant-pad closure as later stages.
- Verified Puffer 5.0 native C/CUDA interfaces, continuous actions and NVIDIA
  training requirement from documentation/source. No runtime compatibility or
  performance claims were tested; no dependencies installed or GPU jobs started.
- Saved the [brief](surgical/BRIEF.md) and [proposal](surgical/README.md), including
  source revisions, setup, effort estimates, physics limitations and proposed gates.
- Repository fast-forward check completed using a per-command rebase override;
  origin was already current. Preserved pre-existing local changes. Documentation
  whitespace checks passed; simulation tests were unnecessary for this report.
- Next: convert and statically inspect one Si PSM with the 420006 needle driver,
  then validate its physical mechanism before adding the second arm or training.

## October 5: Neuralink robot comparison follow-up

- Observed follow-up interval **17:34:30–17:35:23 UTC**:
  **53 seconds elapsed**. Simulation, training and rendering **0 s**.
- Checked Neuralink's public technology description and PRIME progress update,
  and the ROSA ONE Brain manufacturer's description. R1's linear positioning
  stage, optical head and specialized thread inserter are a closer design target
  for neural implantation than the proposed bimanual dVRK scene.
- A custom R1-inspired MuJoCo micromanipulator is a conditional alternative if
  neural insertion is the intended task. No project pivot or model implementation
  was performed. No public validated R1 MuJoCo asset was established by this search.

## October 5: neural insertion refocus, interrupted for workspace correction

- Observed interval **17:39:18–17:43:50 UTC**, **4 minutes 32 seconds elapsed**.
  Researched the thread-insertion workflow and inspected the existing scene and
  viewer code. No scene was written, compiled or rendered; no training ran.
- A separate insertion worktree was created, then the user requested that the
  primary checkout use `main` and the fly work move to its own workspace. The empty
  worktree was subsequently reused for the fly; no insertion branch was retained.

## October 5: preserve fly workspace and return primary checkout to main

- Observed interval **17:43:50–17:51:24 UTC**, **7 minutes 34 seconds elapsed**.
  Simulation **0 s**, training **0 s**, rendering **0 s**.
- Saved 184 changed/new fly files in local commit `5a218e1` on
  `feature/fly-brain`. Reused the empty worktree as the dedicated fly workspace.
  Compared all 183 preserved content hashes in the migration manifest; they match.
- Moved roughly 49 GB of generated fly outputs and the fly environments/caches by
  same-filesystem rename, checking unchanged inode, size and modification time.
  Retained migration manifests and pre-move changes in ignored local backups.
- Git LFS integrity passes. Rebuilt relocated editable packages/native binding
  from locked projects; repaired generated launcher paths with backups. Both fly
  package import checks and both `pytest --version` entry points pass. The fly
  working tree is clean. Training remains stopped; no remote push was performed.
- Primary checkout is now `main`, fast-forward checked against origin. Preserved
  the surgical research here, archived the earlier dVRK proposal, and saved the
  active [neural insertion plan](neural_insertion/README.md) and
  [brief](neural_insertion/BRIEF.md). Repository guidance records the user's main
  workspace preference. No robot implementation is claimed at this checkpoint.

## October 5: first surgical insertion scene

- Observed work interval **18:11:16–18:22:49 UTC**, **11 minutes 33 seconds elapsed**.
  Simulation **0 s**, training **0 s**. Three preview batches took **2.38 s**
  of rendering/image-write wall time combined; the final batch took **0.735 s**.
- Built `static_v1` on `main`: five constrained slides, needle/retainer, microscope,
  supported tissue phantom, six targets, vessels and a four-thread cassette.
  Added the dedicated package, CLI, six cameras and review sheets. Simplified the
  brief to focus on our own robot and tasks.
- Corrected carriage/beam and retracted-tool interference found during scratch
  configuration checks. Disabled shadow maps after inspecting microgeometry
  rendering artifacts; final views were visually inspected and all eight PNGs
  decoded/hash-verified. Saved scene/image hashes and static check results with
  the previews.
- Static checks pass at 120 sampled poses: no penetrations, no warnings, enabled
  command/force limits, 12.925 mm initial needle clearance and 4.925 mm sampled
  minimum. Both static package tests pass, including a deliberate solid-phantom
  collision probe. No physical hold or insertion cycle was run.
- Native macOS viewer completed its bounded 15-second smoke check with exit 0;
  automated native screenshot access was unavailable. Offscreen camera renders
  supplied visual inspection. CLI help, source compilation, Git LFS integrity
  and whitespace checks pass. Unrelated dynamic demo suites were not rerun.
- Next: user visual inspection, then bounded hold and approach validation.

## October 5: flexible-thread SI attempt, interrupted for resolution review

- Observed interval **19:04:20–19:10:00 UTC**, **5 minutes 40 seconds elapsed**,
  including interruption/recovery. No training or rendering was recorded.
- A 1 s SI point-mass flex diagnostic ran with approximately 60 µm support
  penetration; the articulated elastic cable failed minimum-inertia compilation.
  These were different formulations/settings, not a controlled unit comparison.
  Physics wall time was not separately recorded for this initial diagnostic.
- Stopped at the user's resolution concern. The initial flexible-scene/native
  drafts were never built or used for a robot mission; they are now archived in
  ignored build storage. User subsequently authorized consistent unit-rescaling
  experiments and requested documentation of requirements, pros and cons.

## October 5: microscale unit-rescaling and accuracy study

- Observed interval **19:12:31–19:43:07 UTC**, **30 minutes 36 seconds elapsed**
  to the verification checkpoint. User explicitly prioritized accuracy over
  performance. Work stayed on `main`; static workcell and fly workspace preserved.
- Ran **36 saved exploratory cases** and a **nine-case final suite** using
  standard CPU MuJoCo 3.12.0. Aggregate saved simulation time **23.27 s**; summed
  trial wall time **684.61 s**, including compilation/Python stepping/telemetry.
  Some exploratory trials overlapped, so this sum is not elapsed project time or
  measured CPU compute time. Additional short diagnostic/test calls are excluded.
- A separate native `mj_step(nstep=60000)` timing covered **0.15 s simulation**
  in **32.8165 s wall time** while another trial ran; this contended timing is not
  used as a standalone throughput claim. Training **0 s**.
- Final validation: **7.02 s aggregate simulation**, **136.80 s summed trial wall**,
  **137.04 s suite elapsed**. Unit equivalence and preliminary bending checks pass;
  contact penetration reaches 1.739 µm, but finest timestep refinement differs by
  34.979 µm and fails the 1 µm accuracy gate. Suite correctly exits 1 and retains
  failures. Contact quality remains unresolved; no robot integration or RL began.
- Added explicit dimensional conversions, reusable benchmarks, run provenance,
  all pilot records, measured results and [requirements/tradeoffs](neural_insertion/RESCALING.md).
  Identified generated-composite contact defaults and vertex rounding as special
  cases. Kept physical masses/radii; no inertia floors or added armature.
- Final plots and recorded-state render took **0.812 s** rendering/plotting wall
  time, with **0 s additional simulation**. Inspected both actual PNGs; preserved
  an earlier static bench image whose rendering time was not separately captured.
  No video was produced for this failed-contact diagnostic.
- All **six surgical-scene/rescaling tests pass**; static collision inspection
  still passes all 120 sampled poses. CLI help, whitespace checks and Git LFS
  integrity pass. Unrelated wind/locomotion suites were not rerun; the native
  viewer implementation was unchanged from its prior smoke test. No commit/push.

## October 5: document solver alternatives and Isaac Sim 6.1 integration

- Retained documentation/verification clock window **19:55:51–20:03:32 UTC**,
  **7 minutes 41 seconds elapsed**. Earlier research in this follow-up is outside
  this measured window; no earlier start is reconstructed. Simulation **0 s**,
  training **0 s**; no dependencies installed or remote jobs launched.
- Saved the [solver decision](neural_insertion/SOLVER_DECISION.md), retaining the
  failed contact results and numerical/physical-validation distinction. Inspected
  pinned MuJoCo 3.12.0 plugin/integration source and official SOFA, PhysX, Newton
  and Isaac documentation. User supplied the Isaac Sim 6.1 engine integration page;
  recorded its one-active-engine rule, rigid-only Newton tensor coverage and
  metre-based USD restriction from the companion backend guide.
- Recommendation: benchmark standalone Newton VBD rods against an independently
  refined SOFA beam model before a custom module or engine fork. An Isaac backend
  is a proposed integration destination, not evidence of microscale accuracy.
  No replacement solver has been accepted or implemented.
- Used Archify to produce the proposed architecture as standalone HTML plus a
  selected PNG. Updated the proposal for the user's Isaac 6.1 steering. Two
  finalization passes took **2.620 s** combined including delivery/browser checks;
  diagram rendering/capture time was not isolated from validation overhead.
  Final **9/9 showcase checks**, zero errors/warnings, browser checks passed.
  Inspected the final light/dark 2048×1320 captures; automated evidence also covers
  1440×900. Portable hashes/review receipt accompany the diagram; machine-local
  provenance stays ignored.
- Documentation links, whitespace, portable-path/privacy checks and Git LFS
  integrity pass. No dynamics tests rerun for this documentation-only follow-up;
  previous physics failures remain unchanged. No commit/push.

## October 5: prioritize MuJoCo DER after literature review

- Observed research/documentation window **20:17:25–20:24:00 UTC**,
  **6 minutes 35 seconds elapsed** to the verification checkpoint. Simulation,
  training and rendering **0 s**; no dependencies installed or engine changes.
- Checked the published MuJoCo DER paper and upstream revision
  `71d2d504ef8e6408ccabcba2e05ec3a2b19e0fad`: capsule/joint representation,
  quasistatic twist, and plugin instructions targeting 3.3.2 with custom composite
  source changes. Checked Newton unified VBD/coupling documentation, DeformX paper
  and setup, the historical inertia issue, and the inspected 3.12.0 cable source.
- Updated brief, README and solver decision to the user's order: MuJoCo DER,
  Isaac/Newton, coupled solvers, standalone alternatives. Retained accuracy gates
  and the earlier diagram as an optional later integration route. Contact remains
  an unisolated cause; no replacement solver has passed or been implemented.
- Fast-forward synchronization confirmed main is up to date, with existing local
  work preserved. Documentation local links, portable paths and whitespace pass.
  No dynamics tests rerun for these documentation-only changes; no commit/push.

## October 5: MuJoCo DER port and minimum acceptance checks

- Observed interval **20:26:25–20:59:22 UTC**, **32 minutes 57 seconds elapsed**
  to the verification checkpoint. Stayed on `main`; no worktree, installed-engine
  patch, robot integration or RL training. Training **0 s**.
- Pinned and preserved the upstream DER C++ source and MIT license. Built a local
  plugin against MuJoCo 3.12.0 with hash-checked Eigen headers. Added explicit model
  construction, lifecycle/state handling, and a separate direct-projection variant
  with torsional endpoint moments. Native mass/inertia/contact geometry preserved.
- The retained published force mapping returned zero torque for a straight
  twisted rod. The corrected variant passes virtual-work checks at maximum
  relative error 2.85e−9, repeated evaluation/copy/reset/restore, and force unit
  equivalence. These are numerical consistency checks, not physical calibration.
- Complete suite: **6.200002 s aggregate simulation**, **206.225 s elapsed wall**,
  including model creation/Python stepping/telemetry. Cantilever error 2.400%;
  free-relaxation shape refinement 0.0998 µm and energy drift 0.0380%. Contact
  refinement differences 58.573 and 14.379 µm, failing the 1 µm gate.
- One further matched contact refinement to **39.0625 ns**: **0.06 s simulation**,
  **179.681 s elapsed wall**, with **41.762 µm** trajectory disagreement. Stopped
  further timestep reduction at the failed, nonmonotonic convergence result.
  Both accuracy commands exit 1 and preserve failed results.
- An initial validation attempt completed three 2 s bending cases and one 0.01 s
  relaxation before a JSON reporting error; those trajectories and the interrupted
  manifest remain saved. Its three recorded bending trial walls total **47.98 s**;
  relaxation wall was lost at reporting failure and is not reconstructed. Separate
  force audit saved another 2 µs simulation. Regression-test/diagnostic simulation
  time is not included in these benchmark totals. Some runs overlapped tests, so
  reported wall times are not isolated physics compute or throughput measurements.
- Two scientific-figure passes took **0.620 s** and **0.583 s** plotting wall with
  **0 s additional simulation**. Static-preview rendering/build times were not
  isolated. Inspected initial scenes and both actual result figures. Preserved
  original and corrected preview variants; no video produced for the failed gate.
- Focused **10 tests pass**; complete optional-stack suite **42 tests pass in
  163.24 s**. Static robot still passes 120 sampled clearance poses. Corrected a
  viewer-handle access error; the subsequent native static viewer completed a
  clean 5 s smoke test. Whitespace, documentation links, source hashes, portable
  result paths and Git LFS integrity pass. No commit/push.
- Saved [acceptance report](neural_insertion/DER_VALIDATION.md), full result
  manifests, repeatable benchmark/refinement commands and selected review images.
  Contact accuracy, settled contact, actual grasp/release and physical calibration
  remain unvalidated. The corrected model is not accepted for surgery tasks or RL.

## October 5: isolate contact and explicitly recheck unit rescaling

- Observed interval **21:11:50–21:32:03 UTC**, **20 minutes 13 seconds elapsed**
  to the verification checkpoint. Stayed on `main`; preserved existing work,
  synchronized with fast-forward only, and did not patch the installed engine.
  No robot dynamics or RL added; training **0 s**. No commit/push.
- Added normal and angled/sliding rigid-segment controls, the same articulated
  chain with elastic forces removed, and a matched flexible DER continuation.
  Saved a portable pre-contact state with source/trajectory provenance. Documented
  why each control exists: rigid fixtures isolate contact, not replace the flexible
  surgical thread. Static contact scenes were rendered and inspected before motion.
- Checked each fixture in mm–g–s, mm–µg–s and 0.1 mm–g–s, separately changing
  mass and length units. Compiled physical quantities agree to normalized error
  4.65e−16. Preserved the original 0.01 µm unit-agreement and 1 µm timestep gates.
- Full 24-case matrix: **1.08 s aggregate simulation**, **351.913 s elapsed wall**.
  All cases complete without warnings; 11 of 28 comparison gates pass. Rigid
  normal-drop refinement differs by 2.234 µm; angled/sliding by 0.345 µm. The
  no-elasticity chain fails timestep and unit checks. Fine-step flexible DER differs
  by 3.502 µm under mass-unit change and 3.055 µm under length-unit change. Its
  matched-checkpoint timestep difference is 52.031 µm. Failed cases are retained.
- Two explicitly recorded follow-up normal-drop halvings: **0.12 s simulation**,
  **7.435 s elapsed wall**; differences 4.502 and 1.135 µm remain failures. Stopped
  further timestep refinement. Wall times include model creation, native stepping
  and telemetry and overlap repository tests; they are not isolated compute time.
- New C++ telemetry loop uses unchanged MuJoCo split stepping. Rigid and DER
  regression cases match Python stepping bit for bit. Fine DER restart reproduces
  the original saved trajectory exactly. Four focused tests pass; full optional
  stack suite **46 tests pass in 158.75 s**. Regression-test simulation is excluded
  from the benchmark totals above. Native static viewer completes a clean 5 s run.
- Scientific result plotting: **0.995 s wall**, **0 s additional simulation**.
  Static rendering/build time was not isolated. Inspected both the actual initial
  sheet and eight-panel result figure. Source/checkpoint/trajectory hashes,
  documentation links, portable paths, whitespace and Git LFS integrity pass.
- Saved [rationale, protocol and results](neural_insertion/CONTACT_ISOLATION.md),
  complete manifests and PNG/SVG plots. The evidence isolates failures without
  rod elasticity and demonstrates dynamic unit sensitivity, but does not identify
  a specific engine defect. Contact accuracy remains unaccepted; next proposed
  work is a focused contact-solve/reference audit before any engine patch.

## October 5: run the focused contact-solver audit

- Observed interval **21:37:23–21:58:18 UTC**, **20 minutes 55 seconds elapsed**
  to the verification checkpoint. Stayed on `main`, synchronized fast-forward
  only, preserved existing work and the installed MuJoCo engine. No commit/push.
  Robot integration and training **0 s**; no scene/viewer behavior changed.
- Inspected and hashed MuJoCo 3.12.0 constraint, solver, collision and integration
  sources and the installed library. Implemented an independent SI normal-impact
  law for the symmetric rigid capsule, with adaptive continuous-time integration
  and a separate fixed-step recurrence. The recurrence reproduces MuJoCo within
  **0.000170 µm**; the continuous reference exposes the existing timestep error.
  Tightening the reference changes positions by **2.65e−8 µm** on the saved grid.
- Frozen evaluations: 12 rigid states and seven states for each articulated model
  across three unit systems, with default and zero solver tolerance for chains.
  Maximum rigid formula error **1.16e−11** relative; flexible-chain same-state
  generalized contact-force unit difference **9.49e−10**. No physical integration
  time is assigned to these scratch state/force queries.
- Two paired-unit DER traces inspect every physics step: **0.04 s aggregate
  MuJoCo simulation**, **59.233 s wall**. Saved the first differing contact sets
  and replayed both states across all units. Picometre-scale signed-gap differences
  put the evolved trajectories on opposite sides of contact activation; either
  identical state gives matching contact sets across units.
- Three predefined planar 1e−12 rad perturbations (seeds 7, 17, 29) and one
  unperturbed zero-tolerance control: **0.12 s aggregate MuJoCo simulation**,
  **324.663 s elapsed wall**. Perturbations produce maximum trajectory differences
  **3.623, 16.466 and 11.761 µm**. Zero tolerance changes the trajectory by
  **2.332 µm**, takes **186.874 s**, and reaches the 100-iteration cap; it is not
  treated as a reference or a demonstrated cure.
- Initial saved reference audit: **0.12 s aggregate independent-reference time**,
  **4.428 s wall**. Completed reference audit including four independent discrete
  recurrences: **0.36 s reference time**, **4.535 s wall**. A preceding recurrence
  pilot covered another **0.24 s reference time**, approximately **1.079 s summed
  loop wall**. These are scalar reference calculations, not MuJoCo simulation.
  Wall times include setup/telemetry and overlap other jobs; no isolated compute
  or RL-throughput claim is made. Regression-test simulation is excluded.
- Four focused reference tests pass; full optional-stack suite **50 tests pass
  in 161.33 s**. Existing accuracy failures remain failures. New numerical tests
  check formula agreement, exact discrete free flight, complete rigid impact and
  refinement of the independent continuous reference.
- Two plot/scratch-replay passes: **0.750 s** and **0.747 s wall**, **0 s additional
  simulation**. Corrected a receipt extension filter, then inspected the actual
  four-panel PNG. Source/trajectory hashes, plot receipts, documentation links,
  portable paths, whitespace and Git LFS integrity pass. No video or native-viewer
  rerun was needed because the inspected fixtures and viewer are unchanged.
- Saved [audit explanation and decision](neural_insertion/CONTACT_AUDIT.md),
  independent reference, paired-event and sensitivity manifests, frozen replay,
  source provenance and PNG/SVG review plots. Contact-event localization/adaptive
  substepping is the next proposed integration experiment. No specific engine
  defect or accepted fix is claimed; physical accuracy and release remain gated.

## October 5: find a route through the failed contact gates

- Observed interval **22:06:11–23:35 UTC** to this progress checkpoint, with the
  recorded matrix still running (its manifest and plots follow separately).
  Stayed on `main`, preserved existing work and the installed MuJoCo engine.
  Robot integration and training **0 s**; no scene/viewer behavior changed.
- Independent SI study of the rigid symmetric drop under the baseline and
  penetration-ramped laws, Euler and RK4 recurrences, with and without exact
  entry localization: **1.44 s aggregate reference time** per law across four
  timesteps, about **157 s wall** including six continuous references. Showed
  two separable causes: the entry force step (fixed by a ramp or localization)
  and first-order impact error (fixed by RK4).
- Development MuJoCo matrix in scratch: 96 rigid cases (**5.76 s simulation**),
  22 chain cases and 3 perturbation seeds (**1.02 s simulation**), a third DER
  timestep level, a friction-free control and nine low-velocity settling cases
  (**0.57 s simulation**). About **45 min elapsed wall** with up to eight
  parallel processes; not isolated compute. One scratch naming bug overwrote
  early files and was rerun before any result was used.
- Implemented `contact_laws.py`, an RK4-capable path in `contact_kernel.cc`
  (implicitfast path unchanged), `contact_smoothing.py`, the ramped-law and RK4
  extensions of the independent reference, a report script and five regression
  tests. Focused neural-contact tests: **13 pass**. Full suite not yet rerun at
  this checkpoint.
- Measured: rigid timestep gates pass at **0.025–0.077 µm** with the candidate;
  DER unit disagreement **0.003–0.051 µm** (from 3–26 µm); DER timestep
  disagreement **8.0 µm**, then **6.5 µm** at a third halving; seeds
  **0.55 µm** (from 3.6–16.5 µm); settling fixture **6.6e−5 µm** timestep and
  about **1e−11 µm** unit disagreement with **0.05 µm** settled penetration.
  Saved [the explanation and decision](neural_insertion/CONTACT_SMOOTHING.md).
  Accuracy for the robot remains unaccepted; calibration and clock coupling open.

## October 6: close the recorded contact-smoothing matrix

- Recorded run **22:43:09–00:07:52 UTC**, **5082 s elapsed wall** with eight
  parallel workers, **4.44 s aggregate MuJoCo simulation** over 88 cases
  (rigid cases 119 s summed wall; chain and settling cases 19306 s summed wall,
  overlapping). Waited for the complete matrix at the user's request rather
  than stopping the slowest control. Full optional-stack suite **55 tests pass
  in 170.23 s** during the run.
- Every recorded comparison reproduced the development numbers. Candidate
  gates: rigid timestep **0.077/0.025 µm**, rigid units ≤ 3.1e−7 µm; DER drop
  timestep **8.03 µm** and **6.52 µm** at the third halving (fail), DER units
  **0.0031–0.051 µm** (three of four pass); settling fixture **6.6e−5 µm**
  timestep, **≤ 4.9e−11 µm** units, **0.046 µm** settled penetration (pass).
  Exit status 1 because the gated chain drop fails; `ready_for_robot_or_rl`
  remains false.
- Plots **1.50 s wall**, **0 s additional simulation**; inspected the actual
  eight-panel PNG. Copied the manifest to
  [the results file](neural_insertion/CONTACT_SMOOTHING_RESULTS.json) and
  finalized [the write-up](neural_insertion/CONTACT_SMOOTHING.md). Second
  checkpoint commit follows the first progress commit `4a3760c`.

## October 6: task-derived gates and the thread clock

- Observed interval **00:17:55–00:43 UTC** to this checkpoint, on `main` after
  confirming origin was behind local. User direction: complete all next steps,
  focus on delivery, document the process, commit logical successes often.
- Wrote [task-derived acceptance](neural_insertion/ACCEPTANCE.md) and the
  [process account](neural_insertion/PROCESS.md). Added material presets
  (illustrative 100 MPa; polyimide 2.5 GPa from typical film data).
- Clock study: 80 settling cases, **4.8 s aggregate simulation**, **547 s
  elapsed wall** with 10 workers. Largest passing step 10–20 µs; unit
  disagreement ≤ 4.7e−10 µm.
- Probe fixtures (drag, press, release): 24 cases at 10/5 µs, **100 s wall**;
  release rerun at 5/2.5 µs, 8 cases, **94 s wall**; **4.32 s aggregate
  simulation**. Release failed at 10 µs (1.58 µm) and passed at 5 µs (0.19 µm);
  all other gates passed. A first drag design jammed and was replaced before
  the recorded run. Plots **< 2 s**, no additional simulation.
- No training. Wall times overlap other work on this machine; not isolated compute.

## October 6: first robot motion

- Implemented the programmed servo, hold and approach primitives, the
  six-target tour, video recording, CLI commands and a live viewer path, plus
  the physics-only learning scene. Two first-run issues (Z travel, needle sag at
  its stop) were fixed before the recorded run.
- Recorded suite: hold, six single approaches and the tour, **36.2 s aggregate
  simulation**, **3.7 s wall**. Video render **36.0 s wall**, 0 s additional
  simulation; decoded all 613 frames (1920×1080, 30 fps, 20.43 s) and inspected
  the opening, first hover, a transit and the results card. Native viewer
  5 s smoke test passed. Physics-only scene reproduces the full tour bit for bit.
- Follow-up: drag and press at 5/2.5 µs, 16 cases, **2.24 s aggregate
  simulation**, **145 s wall**; all gates pass (≤ 5.1e−7 µm timestep), so every
  fixture now has a direct 5 µs comparison.

## October 6: alignment learning environment

- Built the shared C core, the PufferLib 5.0 adapter and installer, and the
  local evaluation library. Puffer's CPU evaluation build compiled and ran 20
  untrained episodes on the Mac; two portability shims (bash 3.2, framework
  rpath) live in the installer, the pinned Puffer source is unchanged.
- Scripted-reference checks exposed and fixed two task-design errors before
  training. Baselines on 200 evaluation seeds: scripted 100%, zero 0%, random
  0% success; about **17 s wall**, **~260 s aggregate simulation** across the
  three policies. No training yet; the GPU host became reachable through the
  user's tunnel at about 00:53 UTC.

## October 6: alignment training on the RTX 4090

- The user rejected a CPU training fallback before any was written and opened
  the RustDesk tunnel. Synchronized commits by Git bundle,
  without pushing; the GPU checkout was clean and fast-forwarded. A plain
  `uv sync` removed the GPU machine's optional extras; restored at once with the
  documented `--extra wind --extra reticulum` sync and verified imports.
- Built the CUDA trainer after four installer/adapter fixes; made the model file
  byte-identical across machines. Smoke run: **2.9 M steps**, **~72 s**.
- `align_v1` (commit `35b67e2`, 01:08:47 UTC): diverged from a reward bug
  (penalty on unclipped actions); stopped at about **22 M steps**, about **6 min
  training wall**, log retained. MuJoCo physics on CPU, network on GPU.
- Fixed with a regression test (commit `2f851cc`); `align_v2` started
  01:16:11 UTC, 100 M steps planned at about 60 K steps/s.
- `align_v2` diverged from oversized updates (KL 0.16 to ~300 between 12.7 M and
  15.9 M steps); stopped at about 21 M steps, about **5.5 min training wall**.
  `align_v3` lowers only the learning rate, 0.015 to 0.003.
- `align_v3` (commit `f1c7bb3`, 01:22:22 UTC): **99.9 M steps in 29 min training
  wall**; physics on 28 CPU threads, network on the GPU. Evaluation of both
  checkpoints on 200 seeds: **2.4–3.7 s wall** each on the Mac CPU. Final
  checkpoint: 100% success, 0% collisions, 1.40 s mean, worst 8.6 µm lateral.
  Video render **11 s wall**, 0 s additional simulation; decoded all 260 frames.
  Native viewer ran the final checkpoint live (two episodes before the window
  was closed).

## October 6: website pilot

- User direction: a journal web page with an interactive raylib replay instead
  of video, realistic materials in the given palette, the exact task
  specification and a network diagram; build a pilot to review. Compared with
  the AlienWars Gym web pipeline first and adopted its shader approach.
- Built locally only: raylib 6.0 WebAssembly replay viewer, scene and replay
  exporter, journal page, build script. Recorded 400 episodes (200 learned,
  200 scripted) for replay; no training, no new physics claims.
- Checked in Chrome and fixed four issues before committing: episodes requested
  before the C `main` loaded them, canvas sizing, raylib overwriting the page
  title, and diagram text overflow. Nothing published.

## October 6: brief v2

- Started 03:06:38 UTC. With the user, rewrote the brief around mission metrics
  and robustness: learn every phase under a shared disturbance layer, scripted
  controllers as yardsticks only. Original kept as `BRIEF_V1.md`.
- Disturbance layer in the shared C core, committed 03:13:52 UTC. Level 0
  reproduces all 200 recorded evaluation episodes of `align_v3` and the scripted
  controller exactly. At full level both collapse (1.5% and 1.0% success).

## October 6: robust alignment, overlays and compute records

- `align_v4` (commit `3e8b2a9`, 03:15:00 UTC): first run under the disturbance
  layer, level drawn per episode in [0, 1], learning rate 0.003. Closed to about
  50 µm mean final error by 35 M steps, then diverged (KL 0.1 to 14); stopped at
  **48.5 M steps, 14.8 min training wall** (03:29:55 UTC). Physics on 28 CPU
  threads, network on the GPU; GPU near 11% use and 59 W, trainer about 1 GB host
  memory. Log and per-minute resource samples retained.
- `align_v5` (commit `1ab03f5`, 03:30:36 UTC): only the learning rate changes,
  to 0.001. 100 M steps planned at about 57 K steps/s.
- User direction during the runs: make disturbances visible as 3D overlays with
  data in the scene, and record GPU resources per run. Built in the replay
  viewer (committed 03:28:55 UTC, with fixes to callout sizing, layout, caching
  and panning through 03:40:48 UTC after the user's checks in the browser), plus
  a dashboard log parser and per-run compute summary. Site rebuilds took about
  1 min wall each; no new simulation beyond recording replays.
- `align_v5` completed **99.9 M steps in 31.7 min training wall** (52 K steps/s;
  GPU near 10% use and 59 W, trainer under 1.4 GB GPU memory; 28.8 CPU threads
  busy). Evaluation on the Mac CPU, all policies at five levels: **about 2 min
  wall**. Not yet robust (62.5%, 26.5%, 0.5% at levels 0, 0.5, 1).
- `align_v6` (commit `041071a`, 04:05:37 UTC): only the step budget changes,
  to 300 M; about 1 h 45 min expected.

## October 6: first end-to-end cycle

- About 04:28 UTC, user direction: validate the whole cycle (pick, align,
  insert, release, withdraw) as soon as possible before refining any phase.
  `align_v6` keeps training on the GPU meanwhile.
- Built a separate end-to-end scene variant in millimetre-gram units: the
  workcell, the free 32-segment DER thread with a pickup eyelet, a slotted
  needle, a keeper on the needle carriage and a trenched cassette bed; plus a
  provisional tissue force model and a phase runner with measured checks.
  Static geometry rendered and checked for penetration by about 04:36 UTC. Physics
  cost on the Mac CPU: 3.0 ms per 5 µs step (about 600 s per simulated second).
- `align_v6` completed **299.9 M steps in 1 h 31 min training wall** (05:37 UTC).
  Evaluated on the current core (all policies, five levels, about 3 min wall on
  the Mac): 95.0% at none and nominal, 73.5% at stress, with 5% collisions.
- With the user: disturbance scale v2 (nominal quality workcell, stress 2x),
  drive gain 4 and capped start heights, each tested (hold drift at nominal
  under 1 µm on all 200 seeds after the start fix). `align_v7` (commit
  `3c480f1`, 05:44:04 UTC) trains on them, 300 M steps.
- End-to-end pass, mechanism iterations on the Mac CPU (each pick-to-lift test
  about 10 min wall for 1.15 s simulated; full-thread physics about 600 s per
  simulated second): passive hold, clamp pads, squeeze, cage, 5 mN and 100 µN
  force clamps all lost or ejected the eyelet; a latch (one bar closing the
  slot into an eye) held it through the 44 mm lift (06:10 UTC). Full cycle with
  the latch running from the lift checkpoint. Review videos of three
  instructive failures and the latch kept locally; parts gallery added.
- About 06:25 UTC, user direction: a modern interpretation of the machine
  rather than the 2019 design. Design v2 (anchored short thread, ledge needle,
  rotary pincher) built from about 06:29 UTC; seven cycle runs between 06:31 and
  06:37 UTC (each one to two minutes wall on the Mac CPU) found the pinch,
  contact-pair, slack and puncture-track problems. **First complete cycle at
  06:38 UTC**: 0.9 s simulated in 38 s wall (about 40 s per simulated second,
  against 600 for the first-pass scene). Video re-rendered at 06:47 UTC (25 s).
- Old-design run v7A (latch, still air, 1.6 s lowering): no instability through
  transport and lowering, but the eyelet ended 0.30 mm off the tip and failed
  the alignment check. Superseded by design v2.
- `align_v7` at 192 M steps (about 06:42 UTC): peaked at 84.6% training success
  near 114 M steps, then a KL spike (1.05) at 118 M left 20-30% collisions;
  still training.
- After the user found the pincher floating (about 06:50 UTC), design v3: a
  cannula on a bracket, a spring-closed side jaw, then a fork latch; cycles of
  0.9 s simulated in about 35 s wall each. Fast thread settings (50 µs step)
  first tried at 07:54 UTC on that design: about 4 times slower than real time,
  but the loop was lost in the peel.
- With the user (about 07:55 to 08:00 UTC): abstract the thread handling. Thread
  tube static review at 08:01 UTC, ring removed on request. Live viewer runs:
  08:06 (needle missed the settled thread), jam at 45°, wrap at 45° with 18
  segments, the 15° tube, then the tissue grip as constraints. **First complete
  thread-tube cycle at 08:13 UTC**, 0.43 s simulated in about 5 s wall.
- With the user (about 08:17 UTC): three sites before improving. Three-thread
  scene, reloads, moves between sites; release diverged until the thread got
  internal damping and 30 ms to settle. **Three sites completed headless at
  08:21 UTC and live in the viewer at 08:23 UTC** (1.8 s simulated
  in about 41 s). The first live attempt stopped when the state was reset from
  outside the runner mid-run; the runner now stops with a clear message on
  that.
- With the user (about 08:27 UTC): lime thread, disturbances, journal. Lime
  thread at 08:29; disturbance layer in the cycle, seed-1 runs at levels 0, 1, 2
  at 08:33; yardstick baseline on evaluation seeds 1001-1010 at 08:38 (21 runs,
  10 in parallel, 159 s wall); journal rebuilt at 08:39 and checked in the
  browser at 08:40 UTC.
- With the user (about 08:45 UTC): the fresh journal fell short of the previous
  one. Restored the previous journal as the base and extended it: an Insertion
  mode in the WebAssembly viewer (thread-tube replays with rotating thread
  segments, translucent tube and tissue, bond, puncture and release effects,
  insertion callouts and HUD) beside the Alignment mode; a parts gallery for the
  thread-tube workcell; chapters 15-19 for the insertion and an updated chapter
  20. Seven replay runs re-recorded with display fields by 08:57 (identical
  placements to the baseline); first native viewer screenshots at 09:01; full
  site built and checked in the browser by 09:08 UTC.
- With the user (09:12 UTC): move to RL training. Option 1 chosen at 09:20
  (fast C environment; random sites, phantom rotation, threads already placed).
  Thread-end statistics from 60 full-simulation runs and the C yardstick
  baseline on 200 evaluation seeds by 09:31; committed at 09:32. **insert_v1
  launched on the RTX 4090 at 09:33:22 UTC** (55K steps/s; CPU MuJoCo for the
  environments, CUDA for the network).
- 09:40-09:50 UTC: full-simulation harness for the policy (policy_cycle.py).
  The C environment's compensating yardstick, on the policy's 24 inputs built
  from the full simulation, placed 3 of 3 threads at 7.8, 6.2 and 6.3 µm
  (level 1, seed 1001; 4.5 s simulated in about 70 s wall). insert_v1 showed a
  flaw in the C environment: the stages kept moving after the stroke started.
  insert_v2 environment: stages held during the stroke, stroke timed as the
  full cycle; C yardstick re-evaluated (16 s wall).
- 09:54 UTC: insert_v1 stopped by hand after its first checkpoint (65.5M
  steps, about 20 min of training). **insert_v2 launched at 09:55:13 UTC.**
  The v1 checkpoint, sampled as trained, in the C environment: 17.5% within
  10 µm, median 18.5 µm (level 1, 200 evaluation seeds); its deterministic mean
  never starts the stroke. In the full simulation (level 1, seed 1001) it
  placed 3 of 3 threads at 13.3, 24.3 and 7.2 µm (about 90 s wall). Videos
  rendered and committed by 10:05 UTC.
- 10:17 UTC: insert_v2's first checkpoint (65.5M steps). Evaluation in both
  simulations (C, 1200 episodes; full simulation, 21 three-site runs, 10 in
  parallel) took 280 s wall. Full simulation: on par with the approved
  yardstick (level 1 median 22.8 µm, 27 of 30 placed); one run stopped on a
  MuJoCo instability while the thread was held at depth. 10:25-10:40 UTC:
  traced it to jitter of the gripped thread segments during the hold, present
  in the yardstick runs too (and at level 0); not yet fixed.
- 10:41 UTC: insert_v2 at 131M steps (about 46 min of training). Evaluated in
  both simulations by 10:47 (about 5 min wall); compensating yardstick on the
  same 21 full-simulation runs by 10:55. Full simulation, level 1: learned 19
  of 30 within 10 µm (median 9.6 µm), compensating yardstick 19 of 30
  (7.7 µm), approved yardstick 11 of 30 (23 µm); 60 of 60 placed, all runs
  complete.
- 11:00-11:25 UTC: insert_v2 at 196.6M steps evaluated in both simulations,
  sampled and deterministic (about 5 min wall each). Split of the
  full-simulation error: the deterministic policy has the thread end 5.6 µm
  (median) from the target when it starts the stroke; the programmed stroke and
  release bring the 90th percentile from about 11 to 20 µm.
- 11:34 UTC: insert_v2 finished (299.9M steps; 1 h 39 min training on the RTX
  4090, 09:55-11:34). Final evaluations in both simulations, sampled and
  deterministic, about 15 min wall; video, checkpoint asset and results
  document by 11:50 UTC (docs/neural_insertion/INSERT_TRAINING.md).
- With the user (14:38 UTC): publish the insertion results to the
  journal. Results summary (scripts/summarize_insert_training.py), learned
  runs in the 3D viewer beside the scripted ones (replay format NIT2),
  chapters 20-24, the corrected chapter-19 explanation, data fetches versioned
  against caching; checked in the browser at desktop and phone width by
  14:53 UTC. Not pushed or deployed.
- With the user (journal viewer feedback, 15:13 UTC): zoom rebuilt (page-side wheel and pinch
  input proportional to the scroll distance, a clamped zoom target with soft ends, the camera easing
  toward it) and callout leader lines in one neutral colour so the lime thread stands alone.
- With the user (viewer feedback, 15:24 UTC): disturbance overlays drawn at screen scale.
  Force arrows 1 mN = 7 px and vibration 1 mm/s^2 = 15 px at any zoom; micrometre offsets (measured
  tip, target estimate, tissue-motion trail) magnified about their reference by a 1-2-5 factor
  chosen so 10 um spans about 80 px; the legend states both scales.
- With the user (viewer controls, 15:35 UTC): controls regrouped under labels (replay,
  controller, disturbance, evaluation run; camera, overlays; playback and speed); fixed the mode
  buttons losing their highlight on any click in the viewer.
- With the user (default camera, 15:42 UTC): the viewer opens on the Workcell camera, also
  after switching between insertion and alignment; offset magnification capped at x2000.
- With the user (publish, 15:53 UTC): the surgical journal approved for publishing. Removed
  a machine connection detail from the unpushed history, pushed main (54 commits, 109 LFS objects,
  146 MB), added scripts/publish_pages.sh and deployed the journal to GitHub Pages under surgical/.
