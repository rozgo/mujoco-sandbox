# Learned command switching, with remaining yaw and hover failures

One shared actor now starts walking after standing, stops, and resumes in all
three continuous test worlds. Each walking phase covers about 1.9 cm in two
seconds. The prior checkpoint stayed almost motionless when asked to walk after
standing. Neither physical state nor neural memory resets at a command change;
all 78 motor outputs are the learned actor's. This is a practical motor gain,
not yet a completed three-skill solution.

| Continuous phase | Observed result across three worlds |
| --- | --- |
| Stand | All full gates pass |
| Walk | Mean forward speed 0.976-0.981 cm/s; stable; yaw gate fails |
| Stop | All full gates pass; total braking displacement 0.31-0.51 mm |
| Resume | Mean forward speed 0.973-0.978 cm/s; stable; yaw gate fails |

Requested forward speed is 1 cm/s while moving, with zero requested lateral
speed and yaw. Moving mean yaw is 0.484-0.577 rad/s, above the unchanged
0.35 rad/s gate. All phases have valid support according to the recorded
contact-force sensors, and resting wings remain within 0.0033 rad. No numerical
warnings occur. Do not call the full command sequence a gate pass.

The separate five-second motor review retains standing but shows larger walking
path error and lost hover: root RMSE 0.135 mm stand, 20.278 mm walk, 18.234 mm
hover. Walking yaw RMS remains failed at 2.820 rad/s. Thus this candidate does
not replace position_sustain_retention01 as the combined development baseline.

Training changes requests inside episodes. Twelve worlds switch each second,
five stand, five walk, ten hover. Switching worlds receive current-state
walking-reference or full initial-pose stand labels; fixed ground worlds retain
the frozen parent. Ground task and non-wing target weights remain 4, ground
wing loss 10. Every physical action is the student action. The graph, body,
normalization, utility freeze and deployed architecture are unchanged.

| Measurement | Result |
| --- | ---: |
| Training | 181.291996 s |
| Setup | 11.449804 s |
| Collection/forward | 144.621090 s |
| Backward/optimization | 36.658112 s |
| Physical transitions | 143,360 |
| Aggregate simulated experience | 286.720 s |
| Optimizer updates | 140 |
| Observed within-episode command changes | 84 |
| Peak CUDA allocation | 9,451,158,528 bytes |

32 worlds, 16 native CPU MuJoCo/mjbatch threads, RTX 4090 neural work, 5 kHz
physics / 500 Hz actions. The 366 failed training trace windows, all three motor
captures, and all three continuous captures are hash/causality/finite-state
verified. All command boundaries carry nonzero neural memory. Physical identity
and frozen parameter boundaries match the parent. Later recovery work is a
separate pilot with its own declared plan.

[Continuous commands, all three worlds](../../../../previews/embodied_fly/position_commands_01_continuous_v1.mp4).
[Full motor review, including hover failure](../../../../previews/embodied_fly/position_commands_01_all_tasks_v1.mp4).
Both videos are fully decoded, visually inspected and opened. They retain every
case at 1x: 24 seconds/1,200 frames and 15 seconds/750 frames, 1600x900 at 50 fps.
