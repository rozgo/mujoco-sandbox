# Standing does not yet switch into walking

The preserved position_sustain_retention01 checkpoint remains stable in all
three continuous eight-second worlds, but it does not start walking after
standing. Stand and stop pass their full gates in all worlds; walk and resume
fail forward tracking. Each world travels less than 0.3 mm in x over the entire
sequence. Zero warnings or numerical failures; capture takes 48.559113 s.

Commands are verified in both recorded requests and actual actor observations:
0, 1, 0, 1 cm/s, two seconds each. All 78 outputs are applied. Physical state,
previous-action feedback and recurrent memory persist across the boundaries.
The neural latent maps remain active; no teacher, action mask or reset controls
the run. Frozen-start walking success therefore does not establish command
switching. Later ground training needs transitions within episodes.

The video includes all three worlds at 1x with phase labels, eye cameras and
anatomical latent-state projection. All 1,200 frames decode, 12 keyframes were
inspected, and it was opened locally. It deliberately retains the failed walking
requests. This is a diagnostic, not a new learned checkpoint.

[Complete continuous-command review](../../../../previews/embodied_fly/position_sequence_01_all_worlds_v1.mp4).
