# Full motor learning: standing passes; walking and hover remain incomplete

Source 4f97fe5, parent position_feedback01. This pilot keeps exactly the same
wing_position body and 397-input / 78-output architecture. It learns motor
encoders/decoders and modeled cell dynamics while the measured graph edges and
utility/intentions remain fixed. Parameter and physical audits are archived.

Standing now receives all initial actuator targets. Walking retains the parent's
body targets. Ground execution is entirely student-driven; hover training uses
the complete measured-state teacher. No teacher is used during evaluation.

| Measurement | Actual result |
| --- | ---: |
| Setup | 11.526114 s |
| Training | 180.165843 s |
| Collection / forward | 144.110337 s |
| Backward / optimization | 36.042687 s |
| World/action transitions | 144,384 |
| Aggregate simulated experience | 288.768 s |
| Optimizer updates | 141 |
| Trainable parameters | 2,336,540 |
| Total actor parameters | 2,516,146 |
| Peak CUDA allocation | 9,450,888,192 bytes |

32 physical worlds (11 stand, 11 walk, 10 hover), 16 native CPU MuJoCo/mjbatch
threads; RTX 4090 neural work; 5 kHz physics / 500 Hz control. Training has 132
completed episodes and 27 failures: 12 stand, 15 walk, zero assisted hover.
All 40 hover timeouts use a teacher and are not autonomous flight successes.
All failure traces were independently checked.

The predetermined seed95033 review retains all 5 seconds per command. Setup
6.902338 s, stepping/capture 31.738248 s, 7,500 world/action transitions,
zero numerical warnings. All recorded actions, states and causal feedback pass
the capture audit.

| Command | Stable | Full task gate | Observed result |
| --- | --- | --- | --- |
| Stand | yes | PASS | initial form held; root RMSE 0.155 mm; no body-height loss |
| Walk | yes | FAIL | wing pose held, little commanded forward movement |
| Hover | no | FAIL | falls; root RMSE 21.40 mm |

Resting-wing maximum deviations are 0.144 degrees standing and 0.192 degrees
walking. Standing leg-angle RMS is 0.0731 rad; body-angle RMS is 0.0527 rad.
The standing result covers this single complete test, not general robustness.

[Complete review video](../../../../previews/embodied_fly/position_fullbody_01_all_tasks_v1.mp4).
One checkpoint supplies every command. This is a motor-learning candidate,
not a complete motor release or the completed survival simulator. Preserve the
old torque-model checkpoint and both position-pilot videos.

Next work must improve commanded movement and autonomous wing-stroke feedback,
while rehearsing the now-successful standing behavior on this same body. Copying
a ground reference that barely moves cannot establish walking; assisted flight
histories cannot establish closed-loop hover. Keep evaluation teacher-free and
retain the current gates. Utility, takeoff/landing and multi-agent survival remain
later requirements of the unchanged overall goal.

The full video contains all 750 frames at 1600x900 / 50 fps / 1x (15 seconds).
It was decoded, inspected and opened on the local Mac. Render/encode took
58.193610 seconds. Post-action peak wing torque reached the declared 0.03 CGS cap
in hover; no numerical warnings occurred. Final full suite: 124 passed in
85.40 s, with 29 known dependency warnings.
