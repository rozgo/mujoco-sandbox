# Continuous wing-angle candidate — stable ground review

This candidate extends the same actor to 395 inputs with measured wing angles
and velocities. Neutral full-graph migration preserved prior outputs within the
declared tolerances. A **60.494686-second** mixed imitation pilot used 32 neural
sequences, 82 updates and 83,968 supervised examples; setup took 6.911252 s and
validation 1.585699 s. No live physics worlds ran during offline imitation.

All six fixed ground development cases retained stable permitted support, while
their raw tracking gates still failed. The uninterrupted walk/stop/resume capture
also remained stable. Both 0.3-second flight probes fell. This is a useful ground
candidate, not an accepted full locomotion/flight/survival controller.

The [six-second review](../../../../previews/embodied_fly/student_angle01_walk_stop_walk_v1.mp4)
shows a new actual rollout with synchronized signed neural state, utility scores
and observer eye cameras. Its walk/resume phase gates pass; stopping still has
0.8723 mm late drift and fails its gate. The complete 300-frame film is 1600×900,
50 fps and 1×. Render/encode took 52.338373 s; full local decode and visual QA
covered walking, stopping, resuming and the final frame. Eye pixels are not yet
policy inputs, and the neural inset depicts modeled latent state rather than
biological spikes or calibrated voltage.

```sh
open previews/embodied_fly/student_angle01_walk_stop_walk_v1.mp4
```
