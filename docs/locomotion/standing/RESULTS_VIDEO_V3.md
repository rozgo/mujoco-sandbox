# Results video with accepted terrain behavior

[Watch v3 — 67 seconds](../../../previews/locomotion/standing/adaptive_standing_v3.mp4).

After reviewing the diagnostic film, the user accepted all eight healthy terrain
conditions before the damaged-body section, describing them as natural and
organic. These are the cases at 0:00–1:28 in the failure review: both rear gaps,
extreme pads, 18° X slope, both 24° slopes, and 20/28 cm steps. That is a visual
acceptance for the demonstration. Their original quantitative flags remain in
the reports and are summarized in the film; none were retrospectively converted
to passing trials.

The new cut preserves all 42 seconds of physical footage from v2 and adds the
eight exact reviewed CPU trajectories as two ten-second four-panel sections.
It replaces the closing card with a five-second summary. There are **25 recorded
trials across 19 distinct conditions**: the original 17 video-seed trials plus
eight selected evaluation-seed trials, six of which revisit conditions already
in the original film. All 25 remain upright; 10 satisfy every original strict
gate. This selected demonstration is not an independent success-rate estimate.

| Video time | Content |
| --- | --- |
| 0:00–0:12 | Walk, stop to balance, resume walking; following/head/overhead views |
| 0:12–0:22 | Original ordinary supports |
| 0:22–0:32 | Original aggressive terrain |
| 0:32–0:42 | Reviewed rear gaps, extreme pads and 18° slope |
| 0:42–0:52 | Reviewed 24° slopes and 20/28 cm steps |
| 0:52–1:02 | Original damaged-body holds |
| 1:02–1:07 | Results and training provenance |

Every physical sequence runs at 1×. The added panels show complete ten-second
trials without diagnostic pauses. The original footage used MuJoCo Warp; added
terrain states were captured with CPU MuJoCo/mjbatch. Both use the same frozen
standing weights. RGB cameras are observer output. **No new learning or physical
simulation was needed for the edit, and the original friction remains in v3.**
The later friction comparison is a separate diagnostic, documented in
[the friction review](TERRAIN_FRICTION.md).

## Watch or reproduce

From the repository root on macOS:

```sh
open previews/locomotion/standing/adaptive_standing_v3.mp4
```

On the capture machine, the following replays the hashed terrain traces and
models. A clean checkout must first run `record_standing_failures.py` to recreate
its ignored capture cache using the settings in the failure report.

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked \
  python experiments/adaptive_locomotion/scripts/record_standing_results.py \
  --output previews/locomotion/standing/adaptive_standing_v3_reproduction.mp4
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked \
  python experiments/adaptive_locomotion/scripts/qa_standing_video.py \
  previews/locomotion/standing/adaptive_standing_v3_reproduction.mp4
```

The original MP4 sections are decoded and re-encoded once. The additional terrain
panels are rendered from recorded physical states after checking model and trace
hashes. Changes affect cameras, layout and the closing card only. Prior videos and
policy files remain byte-identical.

Final render/encode: **33.001 seconds**. Encoding: **1920×1080, H.264, 25 fps,
1,675 frames**. Every frame decoded, and encoded opening, both new terrain groups,
the return to damaged bodies and the final card were inspected. The original
far-side lower-FL marker is still partly occluded; the existing labels identify
that body. All torque limits hold. Maximum sampled penetration among the 25
included trials is **5.426 mm**. This does not remove penetration failures in other
holdout trials.

[Recording provenance](../../../previews/locomotion/standing/adaptive_standing_v3.json) ·
[Encoded-video QA](../../../previews/locomotion/standing/adaptive_standing_v3.qa.json).
