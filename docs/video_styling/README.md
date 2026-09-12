# Cinematic video styling

A separate presentation workflow for existing MuJoCo footage. It uses the official
WaveSpeed Python SDK, with source hashes, a saved task ID and a side-by-side review.
No simulation, policy or accepted video is modified. Generated pixels are an artistic
interpretation and cannot replace the original physics footage or its measurements.

**Status:** paused, with the complete SDK workflow and reviewed trial preserved
on `main`. Merging this project does not submit or resume a generation.

## Setup

From the repository root:

```sh
uv tool run --from uv==0.12.12 uv sync --project experiments/video_styling --locked
```

Set `WAVESPEED_API_KEY` in the root `.env` or process environment. The `.env` is
ignored by Git; `experiments/video_styling/.env.example` contains a placeholder.
Existing process variables take priority. Do not put keys in prompts, commands,
committed logs or generated manifests. The isolated project pins Python 3.14.7,
WaveSpeed SDK 2.0.2, python-dotenv 1.2.3, Pillow 12.3.0 and bundled FFmpeg 0.6.0.
It runs on macOS and Linux without CUDA. Generation happens on WaveSpeed's service;
the local RTX 4090 is not used for this editing step.

## Model choice

Model catalog and pricing checked **2026-09-12 UTC**. The authenticated live catalog
confirmed all three editing endpoints; their request schemas are saved in
[MODEL_SCHEMAS.json](MODEL_SCHEMAS.json). This is a shortlist, not a quality benchmark.

| Model | Why consider it | Published estimate for five seconds |
| --- | --- | --- |
| [Seedance 2.5 Video Edit](https://wavespeed.ai/models/bytedance/seedance-2.5/video-edit) | First trial: current Seedance editor supports material, environment and lighting changes with source-video guidance; 720p–4K options | $2.20 at 720p, input plus output billing |
| [Kling O3 Pro Video Edit](https://wavespeed.ai/models/kwaivgi/kling-video-o3-pro/video-edit) | Alternative for short edits and optional visual element references | $0.84 |
| [Wan 2.7 Video Edit](https://wavespeed.ai/models/alibaba/wan-2.7/video-edit) | Alternative with negative prompts, fixed seed and explicit 720p/1080p selection | $1.00 at 720p, input plus output billing |

These are estimates from the providers' published tables, not verified invoices.
Live schemas can differ from prose: for example, Wan's live reference-image limit
is three, and Kling's live automatic-shot enum is `intelligence`. This trial sends
neither reference images nor automatic-shot settings. Seedance exposes no seed in
the inspected schema, so an identical prompt is not a promise of identical output.

## Reproduce the trial

The prepared input contains seconds **52–57** of the accepted moving-support film,
cropped to the unobstructed third-person camera at **1280×900 / 25 fps**. This is
five seconds of already validated physical motion, including ongoing combined
platform motion. The saved source manifest identifies the original video,
checkpoint, crop and interval. The full source is preserved.

```sh
# This extracts locally. Choose a fresh output path if the source already exists.
uv tool run --from uv==0.12.12 uv run --project experiments/video_styling --locked style-video prepare --output previews/video_styling/new_source_5s.mp4

# This uploads that clip and creates ONE paid generation.
uv tool run --from uv==0.12.12 uv run --project experiments/video_styling --locked style-video submit --video previews/video_styling/new_source_5s.mp4 --job-dir outputs/video_styling/new_trial

# Resume/poll the same task, downloading when complete. This does not submit again.
uv tool run --from uv==0.12.12 uv run --project experiments/video_styling --locked style-video collect --job-dir outputs/video_styling/new_trial --output previews/video_styling/new_generated.mp4 --wait-seconds 45

# After completion, make a comparison at matching elapsed times.
uv tool run --from uv==0.12.12 uv run --project experiments/video_styling --locked python -m video_styling.review --source previews/video_styling/new_source_5s.mp4 --generated previews/video_styling/new_generated.mp4 --output previews/video_styling/new_comparison.mp4
```

The default [prompt](brushed_metal.txt) asks for brushed steel/aluminum panels,
graphite joints, rubber feet, a charcoal lab and soft studio lighting. It explicitly
asks to retain geometry, individual foot placements, platform motion and the camera.
These are editing instructions, not physical constraints enforced by the model.

The SDK handles upload and result polling. SDK 2.0.2's `run()` exposes output URLs
without the successful task ID, so one documented REST submission persists that ID
immediately for recovery. A job directory can be submitted only once. There are no
automatic paid retries. If a connection drops before an ID arrives, inspect provider
history before deciding whether to create another task. Raw provider replies and
temporary CDN URLs remain in ignored `outputs/`; selected source/generated videos
and sanitized provenance use Git LFS. See the official [SDK](https://wavespeed.ai/docs/python-sdk),
[submission](https://wavespeed.ai/docs/docs-api) and
[catalog](https://wavespeed.ai/docs/list-models) documentation.

## Review standard

Both source and result must decode completely. Inspect the opening, every elapsed
second and last frame for brushed-metal appearance, background quality, flicker,
changed limb anatomy, contact drift, added objects and camera motion. The comparison
samples frames by elapsed timestamp at 25 fps, preserves aspect ratios and never
stretches the generated motion to fit. Different output lengths remain reported.
Visual inspection can reveal deviations; it cannot certify exact physical fidelity.

The implementation's tests cover duplicate submission prevention, unknown-model and
long-clip rejection, and resuming an ambiguous job without creating another charge.
No physics or learning tests are rerun for this presentation-only workflow.

## First result — paused after review

[Watch the comparison](../../previews/video_styling/moving_brushed_metal_comparison_v1.mp4)
· [Generated clip](../../previews/video_styling/moving_brushed_metal_seedance25_v1.mp4)
· [Original crop](../../previews/video_styling/moving_source_5s.mp4).

The single Seedance 2.5 trial produced a metallic robot and a softly lit industrial
lab. Fine limb/contact details and platform pose diverge from the original; motion
is recognizable but not exact. The provider returned **1148×808, 24 fps, 113 frames
(4.708 s)** from the five-second source. The comparison uses the common interval
without stretching: **1920×900, 25 fps, 117 frames (4.68 s)**. All frames decode;
opening, intermediate and ending samples were inspected.

Verified completed-task debit: **$1.98**. Provider inference timing: **215,120 ms**;
observed submission-to-completion polling interval: **217.213 s**. No local GPU
training or new physics. User paused generative styling to improve native MuJoCo
presentation instead. No further generation is queued.
