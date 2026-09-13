# First unfiltered hover reference — retained, visually rejected

The first per-tick force reference passes the former ten-second hover gate,
with 2.881 mm root RMSE and 0.726 mm height span after the first second. It has
no numerical warnings or prohibited ground support. Its predecessor, also
saved here, remains airborne but fails the position gate with 8.602 mm RMSE.
The added fore-aft reference feedback changes wing commands only.

The user rejected the resulting video as too jerky. Passing the old numerical
gate did not establish acceptable visual quality. The subsequent
[PID reference](../pid_hover_reference_01/SUMMARY.md) uses tighter criteria and
1,000 Hz physics. No RL policy was trained in either reference effort.

The final first-version capture uses committed source `ccbd172`: 3.620305 s
setup and 21.275804 s stepping for three simultaneous ten-second reference
worlds. All three captures in this diagnostic round total 63.393316 s stepping.
Hover passes; the accompanying walking reference still fails yaw tracking.
No claim that all three motor commands pass is made.

[Preserved video](../../../../previews/embodied_fly/instant_hover_reference_v1.mp4):
10 seconds, 500 frames, 50 fps, 1600x900, 1x; 53.127069 s rendering. Fully decoded,
eight frames visually inspected, and opened on the Mac. The camera follows
with damping; reported physical errors come from the underlying trajectory.

The new force version, tests, contract marker and explicit checkpoint-transfer
tool are in source. The prior filtered checkpoints have not been changed or
relabelled. See [measured statistics](MEASURED_STATS.json).
