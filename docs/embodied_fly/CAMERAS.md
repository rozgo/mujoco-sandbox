# Smoother motor review camera

Motor reviews now default to a damped overview camera. The previous exact
body-locked framing remains available with `--camera-profile locked` on both
`embodied_fly.record` and `embodied_fly.montage`.

The overview distance increases from 0.95 to 1.30 cm. Its target follows the
measured thorax through exponential smoothing: 0.12-second time constants in
the horizontal directions and 0.40 seconds vertically. A 0.25/0.25/0.30 cm
lag bound keeps the fly visible during fast movement or a fall. Filtering uses
the encoded 50 Hz clock; angle, playback speed and source timestamps remain
unchanged. The body-mounted observer eye views retain their original motion.

This affects rendering only. It changes no captured pose, controller input,
action, model, force or success metric. Video metadata records the profile,
camera-target hash and measured lag/motion statistics.

The first [damped review](../../previews/embodied_fly/position_sustain_retention_01_all_tasks_v2_damped.mp4)
reuses the exact successful retention01 captures. All 750 frames decode at
1600x900, 50 fps, 1x. During hover, RMS vertical target displacement per encoded
frame is 0.0350 cm versus 0.1603 cm for direct body tracking (78.2% lower).
That is a camera-motion measurement, not an improvement in physical stability.
Opening, middle and final frames of each task are inspected; the fly remains
visible. The earlier locked video is preserved for comparison.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.montage \
  outputs/embodied_fly/position_sustain_retention_01_evaluation \
  previews/embodied_fly/new_review.mp4 --camera-profile damped
```
