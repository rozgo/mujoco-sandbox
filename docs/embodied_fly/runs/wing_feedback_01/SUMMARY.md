# Wing feedback learning: encoder learned, physical gates still fail

This stage trains the existing 1,792 sensor-extension weights and 105,222 nonlinear wing-readout parameters through fixed MaleCNS dynamics. The original sensory encoder, cell gains/leaks/biases, base motor decoder, normalization and body mechanics remain frozen. There is no raw-sensor-to-motor bypass or runtime teacher. One checkpoint commands all 78 actuators.

The three-minute online imitation pilot uses 32 worlds: 11 stand, 11 walk, 10 hover. Native CPU MuJoCo/mjbatch runs at 5 kHz on 16 threads; the RTX 4090 runs the brain and learning at 500 Hz control.

| Measurement | Result |
| --- | --- |
| Setup | 10.227206 s |
| Training | 180.484524 s |
| Collection/forward | 146.945133 s |
| Backward/optimization | 33.526860 s |
| Physical transitions | 148,480 |
| Aggregate simulated experience | 296.96 s |
| Updates | 145 |
| Peak allocated CUDA memory | 3,825,026,560 bytes |

All 1,792 sensor weights changed, with an L2 change of 4.000338; their first recorded gradient norm was 0.046946. Independent checks prove every unselected state scalar unchanged, including core parameters and graph/body hashes. The maximum non-wing output difference from the frozen parent on identical histories was 1.6705 across all tasks; this includes airborne states. Ground non-wing distillation error is recorded separately. A changed encoder does not guarantee unchanged physical body behavior.

Training saves 231 completed episodes and 144 failure traces. Twelve hover episodes reached their two-second timeout without leaving the loose training envelope. These are not accepted hover successes or five-second tests.

The unassisted seed72013 review runs five seconds per command: setup 6.308202 s, capture 32.135790 s, 7,500 transitions and 15 aggregate simulated seconds. Standing falls, walking stays upright with valid support but fails wing posture, and hover falls. All strict task gates fail with zero numerical warnings. **Retention02 remains selected.**

The [complete video](../../../../previews/embodied_fly/wing_feedback_01_all_tasks_v1.mp4) preserves every case and failure: 15 s, 750 frames, 1600×900, 50 fps, 1×. Rendering/encoding took 59.680450 s. It was fully decoded, visually inspected and opened. All 144 failure traces and full evaluation captures were verified for hashes, finite values, bounded actions and causal feedback.

The next declared test freezes this learned sensor encoding and refines only the decoder on its resulting physical states. This tests a staged learning recipe; it does not establish that representation changes caused the failures. The full motor, utility and multi-fly survival goal remains incomplete.
