# Aggregated wing-output fit (02)

**Not selected.** Add physical histories reached by readout01 to the original
motor06 examples. This recovers upright standing but loses walking stability;
the wing-pose and flight tasks remain failed. Keep motor06 preferred and stop
this six-output-row calibration path.

- Source51aea32; readout01 acts in32worlds,16CPU MuJoCo threads,seed73010.
- Same canonical body,5kHz physics,500Hz control,RTX4090 neural inference.
- Collection31.473841s plus5.141391s setup;32,000new physical transitions,
  64aggregate simulated seconds. No teacher assistance or episode resets.
- Pool112,000examples after proving identical full upstream/hidden feature maps
  and preserving complete-world train/validation splits. Pooling9.079576s;
  reused examples are not newly simulated experience.
- Fit0.212640s plus2.513395s setup; selectedridgealpha0.1 from the same five values.
- 101,800trainingexamples/10,200validationexamples. Balanced held-out action
  MSE0.0563897→0.0284017 on the expanded corpus; this differs from01's denominator.
- Six existing output rows only;1,542parameters,zero new deployed parameters.
  No physical or upstream network changes; all other output rows unchanged.

Development evaluation uses the same seed72001 and three full five-second cases.
Standing remains upright with permitted support, but wing RMS is0.4342rad and
speed/yaw errors fail. Walking falls; wing RMS1.0188rad. Hover loses altitude and
does not satisfy airborne stability, even though it lands upright in this case.
All full gates fail. Zero numerical warnings; capture33.873253s/setup4.674592s.

Source/cache/checkpoint/model hashes, finite and bounded captures, causal
previous-action feedback, and unchanged upstream/non-wing parameters are verified.
Every failed case remains in the data. The failed follow-up does not justify
claiming a better policy from a lower supervised loss.

[Complete review](../../../../previews/embodied_fly/ground_readout_02_all_tasks_v1.mp4).
15s,750frames,1600×900,50fps,1×. Render/encode/full decode60.252703s;
sampled frames inspected and video automatically opened on the Mac.

Windowed capture analysis confirms that this is not just a startup transient:
standing wing RMS remains0.4226rad over2–5s, and walking remains1.0988rad over
that window. Standing's peak occurs at1.606s, walking's at3.874s. The exact
capture-linked measurements are in `wing_windows.json`.
