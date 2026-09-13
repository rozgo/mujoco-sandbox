# Walking and sustained flight, with altitude and yaw control incomplete

The same unassisted actor now walks forward and stays airborne/upright through
the complete five-second flight review. Standing still passes. Walking yaw and
hover position fail the preserved gates; this is not a complete motor release.

Source fe7a865; parent position_fullbody01. The wing_position body, 397-input /
78-output actor, fixed measured graph and utility/intentions are unchanged.
Motor encoder/decoder parameters and modeled neuron dynamics learn. The saved
parameter, graph, normalization and physical audits are provided alongside this report.

The new walking teacher uses an initial-path preview capped at 0.15 cm error.
That future path/heading is privileged training information, absent from the
student's causal observation. The student must still prove its own tracking.
Walking collection is fully teacher-driven; standing and hover collection are
fully student-driven. Hover's first 50 ms receives 10x supervision weight,
applied to 4,800 existing samples. No runtime phase input or controller is added.

| Measurement | Actual result |
| --- | ---: |
| Setup | 12.127465 s |
| Training | 180.972660 s |
| Collection / forward | 139.260770 s |
| Backward / optimization | 41.696225 s |
| World/action transitions | 165,888 |
| Aggregate simulated experience | 331.776 s |
| Optimizer updates | 162 |
| Peak CUDA allocation | 9,420,270,080 bytes |

32 worlds: 11 stand, 11 walk, 10 hover. Native CPU MuJoCo/mjbatch with 16 threads,
5 kHz physics and 500 Hz actions; RTX 4090 neural work. 2,336,540 motor/cell
dynamics parameters learn. All 152 training failures are hover cases and their
traces are verified. The 30 two-second hover timeouts are not the acceptance test.

The predeclared seed96013 review runs every command for five complete seconds
with no teacher, resets or runtime action substitutions. It contains 7,500
physical world/action transitions. Setup 7.928524 s; capture 31.287544 s;
zero numerical warnings. All states, actions and causal feedback are verified.

| Command | Stable | Full gate | Result |
| --- | --- | --- | --- |
| Stand | yes | PASS | full initial form held; root RMSE 0.151 mm |
| Walk | yes | FAIL | advances about 4.9 cm; root RMSE 1.56 mm; yaw RMS 2.70 rad/s |
| Flight/hover | yes | FAIL | remains airborne but climbs/drifts; root RMSE 89.06 mm |

Ground wing peaks remain 0.176 degrees standing and 0.359 degrees walking.
The flight case starts at 1.84 cm altitude, never drops below 1.52 cm, reaches
14.74 cm and ends at 13.16 cm (last recorded frame 4.998 s). It also drifts
backward about 3.4 cm. Being airborne is not accurate hovering. This covers one
complete predetermined trial, not general robustness or an isolated causal comparison.

[Complete 15-second review](../../../../previews/embodied_fly/position_motion_01_all_tasks_v1.mp4).
The file includes every command at 1x, 1600x900, 50 fps, 750 frames. It was fully
decoded, visually inspected and opened. Earlier candidates and videos are intact.

Next: train sustained altitude/velocity correction on this same body while
rehearsing standing and walking. Longer physical hover episodes can expose the
late climb; the heavy startup weight can return to normal now that the actor
starts and sustains its strokes. Recheck every command after continuation.
The broader utility, takeoff/landing and multi-agent survival goal remains active.

Render/encode: 58.985950 s. Final full suite: 126 passed
in 89.01 s, with 33 known dependency warnings.
