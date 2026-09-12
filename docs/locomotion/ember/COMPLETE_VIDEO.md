# One final policy in the Ember scene

[Watch the film](../../../previews/locomotion/ember/adaptive_dog_complete_v2.mp4) · [Training summary](../GPU_SUMMARY.md) · [Training card](../../../previews/locomotion/ember/training_stats_v1.png) · [Results card](../../../previews/locomotion/ember/results_v1.png)

**2m 54s, 1920×1080, 25 fps, 1×.** Every scene uses the same frozen final actor from
the measured 4,096-world RTX 4090 curriculum. The Ember palette uses charcoal and
steel, machinery yellow, oxide-red damage markers and forest-green contact dots.
Cosmetic covers hide embedded branding; vendor assets and physical models remain
unchanged. The approved industrial environment adds hazard borders to every
support family, inset steel plates, corner fasteners, floor seams, numbered lane
markings and a dark ribbed background. This is native simulation rendering.

Version 2 re-renders the **exact same 51 physical runs** as the
[first Ember film](../../../previews/locomotion/ember/adaptive_dog_complete_v1.mp4).
Only presentation changes; the checkpoints, outcomes, chapter order, timestamps
and normal playback speed remain identical. There is no new physical capture or
training for v2. The original video, manifest and QA report remain preserved.

Watch immediately on macOS, from the repository root:

```sh
open previews/locomotion/ember/adaptive_dog_complete_v2.mp4
```

## Chapters

| Video time | Demonstration |
| --- | --- |
| 0:00–0:12 | Healthy walking, observer/overhead/head views |
| 0:12–0:24 | Any one lower leg removed, four simultaneous bodies |
| 0:24–0:36 | Any one entire leg removed, four simultaneous bodies |
| 0:36–0:48 | Healthy walk–hold–walk, three synchronized cameras |
| 0:48–0:58 | Flat ground, individual pads and 6° slopes |
| 0:58–1:08 | One foot without support, all four positions |
| 1:08–1:18 | High/extreme pads and 12/20 cm steps |
| 1:18–1:28 | Fore/aft and sideways slopes at 12°/18° |
| 1:28–1:38 | 24° slopes, 28 cm steps and stationary deck control |
| 1:38–1:48 | Static holds with each lower-leg removal |
| 1:48–1:58 | Static holds with each entire-leg removal |
| 1:58–2:10 | Walk–hold–walk with each lower-leg removal |
| 2:10–2:22 | Walk–hold–walk with each entire-leg removal |
| 2:22–2:32 | Translating, rotating, heaving and rocking platforms |
| 2:32–2:42 | Combined platform motion, three synchronized views |
| 2:42–2:48 | Actual GPU training time and scale |
| 2:48–2:54 | Separate final held-out evaluation results |

The three-camera sequences show the same physical run and timestamp from each
view. Four-panel chapters show independent bodies/conditions with synchronized
simulation clocks. Cameras are observer output; the actor does not receive pixels.
High-level velocity commands and platform actuator targets are supplied by the
demo. The learned actor supplies the dog's joint commands through bounded servos.

## Exactly which policy and runs

All 51 conditions use
`assets/locomotion/checkpoints/gpu_from_scratch/unified.pt`, SHA-256
`076099ef8fc47bad53cb0cbeb9cc311e64ea47accacfa8ca4080488934b10241`.
No intermediate checkpoint is substituted for walking or balancing.

Each condition was specified in the [brief](VIDEO_BRIEF.md) before capture: one
trial, seed **9511**, with every outcome retained. The nine walking trials last
12 seconds each, 33 static/transition trials last 10 or 12 seconds each, and six
platform trials last 10 seconds each. All **51/51 remain upright** and **49/51 meet
their strict/task targets**. The two misses are labeled in the movie:

- The 24° fore/aft slope uses an unintended link for support, reaching **10.95 N**.
- Rear-left lower-leg removal drifts **17.2 cm** during the transition hold,
  exceeding the unchanged **15 cm** threshold.

The unchanged final card reports the separate **204-trial held-out audit**, with
36/36 walking completions, 136/144 strict static passes, 24/24 moving passes and 204/204
upright. Its eight static misses are not the denominator for this 51-trial film.
The [full training report](../GPU_REPORT.md) records both scope and limitations.

Training used MuJoCo Warp on the RTX 4090. The original demonstration runs executed
the frozen actor in **native CPU MuJoCo/mjbatch**, with **0.5 ms walking physics**,
**2 ms balance physics** and **20 ms control**. No new learning or reward changes
were made for this film. Saved physical models, states, actions, torques, contacts
and timestamps precede rendering; replay advances through those recorded states.

The [video manifest](../../../previews/locomotion/ember/adaptive_dog_complete_v2.json)
records each trial's measurements, checkpoint/model/trace hashes, capture and
render time, chapter boundaries and video hash. The
[QA report](../../../previews/locomotion/ember/adaptive_dog_complete_v2.qa.json)
records complete decoding, frame/timeline/coverage checks and visual inspection.
Raw trajectory caches remain under ignored `outputs/`.

## Reproduce

From the repository root, with assets available through Git LFS:

```sh
git lfs pull
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/record_unified_showcase.py --part capture
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/record_unified_showcase.py --part preview --environment industrial
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/record_unified_showcase.py --part render --environment industrial --output previews/locomotion/ember/adaptive_dog_complete_replay.mp4
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/qa_unified_showcase.py previews/locomotion/ember/adaptive_dog_complete_replay.mp4
```

Existing capture caches are reused only after their hashes/specifications match.
Rendering refuses to overwrite an existing movie. Inspect the actual preview and
encoded QA samples after reproducing. Fonts/rendering may vary between hosts.
The approved environment is selected explicitly with `--environment industrial`;
`--environment simple` reproduces the first Ember environment. Both use the same
saved trajectories. No new capture is needed when that cache is already present.
The [previous graphite film](../../../previews/locomotion/graphite/adaptive_dog_complete_v2.mp4)
and all accepted release artifacts remain available.
