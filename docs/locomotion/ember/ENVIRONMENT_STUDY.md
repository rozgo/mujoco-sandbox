# Industrial test-bay environment preview

[Open the static study](../../../previews/locomotion/ember/environment_v2.png)

The user requested more consistent environment detail, then explicitly asked to
see the platforms and floor before continuing. This candidate is **preview-only**;
the completed Ember v1 video and live theme remain unchanged pending review.

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

After visual approval, the existing saved 51-run traces can be re-rendered with
this environment. Training and physical recapture are unnecessary.
