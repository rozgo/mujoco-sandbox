# Six physical round trips return to the starting point

[Watch the synchronized reference video](../../../../previews/embodied_fly/round_trip_reference_v1.mp4).

The accepted PID reference completes left/right, right/left, forward/backward,
backward/forward, up/down and down/up target sequences. Each goes to +/-1.5 mm,
returns home, and holds for four seconds. A seventh world performs stationary
hover. **No learned fly actor or RL updates are present in this reference.**

All six paths reach both intermediate targets and return. During the last second,
maximum position error is below 0.08 mm in every case. The largest speed measured
from 100 ms displacements is 0.0307 mm/s. These observations support directional
control authority on the unchanged plant, not learned navigation or recovery.

The original combined gate nevertheless fails in all seven worlds: its raw-speed
limit of 1.5 mm/s rejects the approximately 13.54 mm/s instantaneous vertical RMS
velocity caused by tiny, rapid wingbeat vibration. Even the stationary PID control
fails it. Preserve that original failed gate in reference.json. The supplementary
motion_check.json separates sustained displacement from raw velocity; it does not
change the physics, filter forces, replace the old result or claim new RL success.
The previously accepted PID hover criterion allowed 15 mm/s raw vertical RMS.

## Same body and measured time

The Linux reference model matches the accepted compiled MJB hash exactly:
214cd5830379849d44e790647116b05e873496747bd8619e91a47dd558161899.
Seven CPU physics worlds, seven threads, 1,000 Hz physics and 500 Hz bounded wing
commands collect 42,000 action transitions: 12 seconds per world, 84 aggregate
simulated seconds. Capture takes 24.180077 seconds, total wall time 31.244537
seconds including setup/export. Training and actor updates are zero.

The first Mac compilation was rejected before stepping by the exact physical
fingerprint check. Array comparison shows compiler floating-point differences:
maximum mass delta 5.42e-20 in model units and quaternion delta 2.61e-14.
The check was kept intact; physical capture ran on the validated training machine.
Mac rendering replays that machine's saved binary model and physical states.
See mac_compile_difference.json. The first output directory/log remain preserved.

A slow first video attempt repeatedly decompressed NPZ arrays per panel/frame.
It was interrupted after roughly five minutes; the partial file and log remain in
ignored outputs. The corrected renderer loads each needed array once. Both use
the same captured trajectories; the interruption did not rerun or alter physics.
Final encoding and visual checks are recorded separately in video_qa.json.

Two target-sequence tests pass in 0.80 seconds. Six critic tests pass in 1.30
seconds; the full critic-change package passed 185 tests in 155.06 seconds.
The physical seven-world probe provides integration evidence for the new paths.

## Reproduce

On the validated physics machine, from the repository root:

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.round_trip --output outputs/embodied_fly/round_trip_reproduction
```

The command returns exit 2 for the preserved original raw-speed gate failure.
It still saves the full capture and report. Transfer them to the Mac for rendering:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.round_trip_video outputs/embodied_fly/round_trip_reproduction previews/embodied_fly/round_trip_reproduction.mp4
open previews/embodied_fly/round_trip_reference_v1.mp4
```

The [proposed learned curriculum](../../CLOSED_FLIGHT_CURRICULUM.md) keeps one
brain and mixes stationary hover with these closed trips. Random distances and
timing will test target response; prescribed target paths do not prescribe the
wing actions. No learned round-trip training has run yet.
