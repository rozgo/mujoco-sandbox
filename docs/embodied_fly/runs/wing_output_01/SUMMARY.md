# Wing output 01: ground stability retained, posture and hover not improved

Parent: `state_hover_retention_02`. Source: `0d767e7`. Only the six existing wing-output linear rows and biases were eligible for learning (1,542 parameters). All upstream parameters, persistent buffers and 72 other output rows remained bitwise unchanged. The independent checkpoint audit verified 2,423,543 frozen state scalars. Same-history non-wing actions differed by at most 1.386e-6 across 174,080 world/actions. This stage was output-decoder imitation; internal cell parameters were frozen.

The 180.428-second pilot used 32 physical worlds (11 stand, 11 walk, 10 hover), 16 native CPU MuJoCo threads, and RTX 4090 neural inference and learning. Physics remained 5 kHz; actions 500 Hz. Fresh Adam used learning rate 1e-4. The actor supplied all executed actions; ground labels came from the frozen parent plus resting-wing corrections, and hover labels from the measured-state reference. No teacher action was executed.

Training collected 174,080 transitions / 348.160 aggregate simulated seconds, with 170 updates. Setup took 10.317 s; collection/forward 175.684 s; optimization 4.732 s. Peak allocated CUDA memory was 891,770,368 bytes. All 495 failure traces (31,680 frames) and causal bounded-action mixtures were verified and retained.

Independent full five-second command evaluations had no numerical warnings. Stand and walk stayed upright with permitted support, but resting-wing accuracy regressed: stand RMS 0.1968 rad / peak 0.7888 rad; walk RMS 0.1088 rad / peak 0.3197 rad. Hover fell, with 18.938 mm root tracking RMS. Evaluation setup took 7.122 s and capture 31.549 s. All full task gates still fail.

The candidate is archived, **not promoted**. Retention02 remains the combined development parent. Restricting parameter updates succeeded technically, but this larger imitation step did not improve behavior.

The [complete video](../../../../previews/embodied_fly/wing_output_01_all_tasks_v1.mp4) preserves all three outcomes. It is 15 s, 750 frames, 1600×900, 50 fps, 1× playback, fully decoded, visually inspected and opened on the Mac. Rendering/encoding took 60.325 s.
