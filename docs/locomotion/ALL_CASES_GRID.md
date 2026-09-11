# All trained bodies in one synchronized video

Started September 10, 2026 at **20:24:00 UTC**. User request: “can we show all trained cases at once in a video?” Follow-up question: “are we still using a single policy?”

Use the single selected checkpoint `paired_selected_545s_seed2.pt` in every panel. No training, policy selection, policy switching or teacher inference is needed for this video. The two reference policies were training-only. Preserve all earlier media.

Show fourteen conditions together at 3840×2160, 25 fps, twelve seconds and 1× playback: one healthy body; a representative FR-thigh torque reduction to 60% at 3 seconds; four single shortened calves at 70%; four mild pairs at 90% / 85%; and four stronger pairs at 75% / 65%. Thus all thirteen geometry variants in the selected checkpoint's two-stage curriculum are shown. Random motor joints/strengths, velocity commands and initial conditions form a distribution, so fourteen panels do not exhaust every randomized training state. Held-out bodies are not included in this grid.

Rows align the four single-leg positions and four paired combinations. Two explanatory cards describe the shared controller and leg labels. Each panel has an identically configured following camera, damage percentages, measured distance, foot-support indicators and completion/failure status. These are independent physical rollouts presented at the same simulation timestamp, not fourteen robots interacting in one world. RGB and annotations are observer output; lane velocity commands remain scripted.

Freeze seed 9137, trial 0, and use the already selected checkpoint for every case before inspecting outcomes. Reuse matching saved healthy and target-pair trajectories after checking checkpoint/model hashes and duration. Capture the remaining cases with the existing strict physics-substep support checks, then render recorded states. Keep unsuccessful results if any occur. Save a hashed capture manifest so layout revisions can replay the same runs.

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog grid
```

The command defaults to the selected checkpoint and writes `previews/locomotion/all_trained_cases.mp4`. Its sidecar records the shared checkpoint hash, cases, physical outcomes, trajectory hashes, timings and rendering source commit. Subsequent calls reuse this grid's saved capture when the checkpoint, seed and duration match. Use another output name for a different capture.

## Delivered result

[Watch the 4K video](../../previews/locomotion/all_trained_cases.mp4) · [Encoded preview](../../previews/locomotion/all_trained_cases.png) · [Capture and outcomes](../../previews/locomotion/all_trained_cases.json) · [Delivery checks](ALL_CASES_GRID_DELIVERY.json)

All fourteen fixed demonstration trials reached 5 m, survived twelve seconds and used only allowed foot/stump support, checked every 2 ms physics step. Measured torques stayed within the physical caps. All recorded timestamps and trajectory hashes matched. Two existing trajectories were reused and twelve were newly simulated. Capture took 5.438 seconds; rendering/export took 51.925 seconds, excluding renderer/context setup. The selected policy retains its 540.622-second cumulative training ancestry; this recording required zero training.

All 300 encoded frames decoded at 3840×2160, 25 fps and twelve seconds. Encoded opening, motor-fault, middle-stride and ending frames were visually inspected. All panels begin moving promptly and remain framed. This is a fixed demonstration, not a new robustness study: the broader [paired-damage evaluation](PAIRED_DAMAGE.md) still includes one failed front-pair trial out of 32.

Open on macOS from the repository root:

```sh
open previews/locomotion/all_trained_cases.mp4
```
