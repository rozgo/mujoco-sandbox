# Industrial test-bay environment preview

[Open the static study](../../../previews/locomotion/ember/environment_v2.png)

The user requested more consistent environment detail, reviewed the static floor
and platforms, and explicitly approved rendering a new video. The environment is
used in the [complete Ember v2 film](COMPLETE_VIDEO.md); the original v1 movie and
live theme remain available unchanged.

The six views show a marked floor, moving deck, modular pads, 24° slope, 20 cm
steps and a missing-support gap. All support families receive the same diagonal
yellow/black border, inset steel skin, flush fasteners and dark side panels.
Fixed floor seams, metre stencils and lane markings provide scale and movement
reference. A dark ribbed background shell completes the visual test bay.

The robot is hidden only in these static review images. No geometry was deleted
from the physical model. The candidate lives in `ember_environment.py`, separate
from the accepted Ember presentation. Added details exist only in `MjvScene` and
are not collision objects, support surfaces, sensors or changes to robot mass.
The background shell is outside the demonstration envelope; it is scenery.

All six models retain identical masses, inertias, joint limits, geom dimensions,
poses, friction, collision masks, actuator parameters and sensor definitions after
configuration. Static previews were rendered and inspected with **zero physics
steps and zero training**. Moving-deck and floor paint from the older scene are
hidden so the palette is consistent across nested environment bodies as well.

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/preview_ember_environment.py
```

The approved video reuses the existing saved 51-run traces. Training and physical
recapture are unnecessary. Use the recorder's `--environment industrial` option;
the default `simple` option preserves the first Ember film's environment.
