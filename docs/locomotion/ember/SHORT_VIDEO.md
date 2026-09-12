# A shorter story at normal playback speed

[Watch the 1m 27s cut](../../../previews/locomotion/ember/adaptive_dog_short_v1.mp4) · [Full 2m 54s film](../../../previews/locomotion/ember/adaptive_dog_complete_v2.mp4)

This alternative edit halves each of the full film's 17 segments. Motion remains
**1×**, at **1920×1080 / 25 fps**. All 51 conditions, synchronized camera panels,
original result labels and both end cards are retained. The full version remains
the primary film while the shorter pacing is reviewed.

Each segment uses one continuous interval of source frames. Walking and static
terrain retain their first half, preserving initial support adjustments.
Walk–hold–walk chapters use the six seconds starting 2.52 seconds into the source
chapter, showing walking before the 3-second stop, the hold, and walking after the
8-second restart. Moving-platform chapters use the middle five seconds, skipping
the initial ramp-up. Each results card lasts three seconds instead of six.

The displayed simulation clocks retain their original values. There is no motion
acceleration, frame interpolation, new physics, learning or MuJoCo rendering.
The edit re-encodes selected frames from the approved industrial v2 movie.
Results labels refer to the complete original 10/12-second trials; the final
evaluation card still refers to the separate 204-trial audit. Shortened excerpts
are presentation, not a new validation run.

The [edit manifest](../../../previews/locomotion/ember/adaptive_dog_short_v1.json)
records every source interval, parent-video hash and exact frame count. The
[QA report](../../../previews/locomotion/ember/adaptive_dog_short_v1.qa.json)
records complete decoding, half-duration checks, preserved command changes and
visual inspection. The first draft accidentally began the transition excerpts
just after stopping; it is retained under ignored `outputs/` and was corrected
before delivery by checking the recorded command timestamps.

```sh
open previews/locomotion/ember/adaptive_dog_short_v1.mp4
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/cut_unified_showcase.py --output previews/locomotion/ember/adaptive_dog_short_replay.mp4
```

The cutter refuses to overwrite existing movies and verifies the parent hash
before editing. No trajectory cache or GPU is needed to reproduce this edit.
