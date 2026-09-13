# Position feedback pilot: resting wings improve, motor tasks incomplete

Source da0997c; parent is the explicit position_initial01 migration of preserved
state_hover_retention02. Same 397-input / 78-output actor controls all three tasks
on the new shared wing_position body. Graph wiring and all original state outside
the wing sensory extension and wing residual are independently verified unchanged.

32 worlds / 16 CPU physics threads; native MuJoCo/mjbatch at 5 kHz, actions at
500 Hz. RTX 4090 neural forward/backward. 107,014 parameters trained.

| Measurement | Actual result |
| --- | ---: |
| Setup | 10.136163 s |
| Training | 180.174339 s |
| Collection / forward | 147.231872 s |
| Backward / optimization | 32.929403 s |
| World/action transitions | 145,408 |
| Aggregate simulated experience | 290.816 s |
| Optimizer updates | 142 |
| Peak CUDA allocation | 3,825,026,560 bytes |

Ground actions are 100% student; hover training is 80% teacher. There are 311
completed training episodes, 193 failures (192 hover, one walk), and 30 hover
2-second timeouts. Timeouts and assisted trajectories do not establish success.
All 193 failure traces and all 7,500 evaluation frames were independently checked.

The seed95013 review runs 5 seconds per command, without assistance or resets.
Setup 7.029955 s; stepping/capture 34.310978 s; zero numerical warnings.

| Task | Upright/stable | Resting wing peak error | Outcome |
| --- | --- | ---: | --- |
| Stand | yes | 0.353 degrees | yaw/posture gates fail; body still crouches/drifts |
| Walk | yes | 0.571 degrees | resting wings pass; commanded motion fails |
| Hover | no | not a ground-pose task | falls; root RMSE 21.47 mm |

Post-action maximum wing torques are 0.001121, 0.001204 and 0.014569 CGS,
below the physical 0.03 cap. These are action-boundary measurements; MuJoCo
applies the actual torque limit every physics substep.

The [complete review video](../../../../previews/embodied_fly/position_feedback_01_all_tasks_v1.mp4)
is 15 s / 750 frames / 1600x900 / 50 fps / 1x. It was fully decoded, visually
inspected and opened. All failures remain visible. No checkpoint promotion.

Next: full initial-form standing labels and full motor-parameter learning,
keeping the same physical profile. The prior retained ground labels could
preserve a crouch, and the wing-only restriction cannot directly correct other
motor outputs. Full hover teacher collection supplies sustained stroke histories;
only a subsequent autonomous review can show whether the student learned them.
