# Nonlinear motor readout 01: better recorded fit, failed physical control

The optional 815→128 tanh→6 motor-cell readout adds 105,222 parameters inside the same actor. Every original actor weight and the current wing_motion body/force law are unchanged. Feature normalization uses fitting frames only. No new recurrent memory, sensor bypass or runtime teacher.

Fitting reuses the five recorded histories, 6,758 fitting frames and 1,680 temporally related validation frames. AdamW fits bounded wing-action error; both the nonlinear architecture/feature normalization and optimization objective differ from the previous linear ridge fit. No new integrated physics or neural replay is counted. The 18,000-update cap is reached before the two-minute allowance: setup1.527266s, fitting17.907381s, total19.509097s. Peak allocated CUDA165,577,728bytes. Selection chooses update10,500.

Validation action MSE is0.00000521 /0.0001855 /0.00000799 /0.006849 /0.005710 across stand/walk/reference-hover/two correction histories. These are related recorded histories, not independent physical episodes.

The unassisted seed72009 test runs5seconds per command. Setup6.300667s, stepping/capture32.082729s, three worlds,7,500transitions/15aggregate simulated seconds; no numerical warnings. Standing stays upright, but its maximum wing deviation reaches1.005rad. Walking and hover fall. All strict task gates fail. Retention02 remains selected; this candidate is not promoted.

The [full15second video](../../../../previews/embodied_fly/nonlinear_motor_readout_01_all_tasks_v1.mp4) retains every task and failure:750frames,1600×900,50fps,1×. Render/encode60.336619s. It was fully decoded, visually inspected and opened automatically.

Independent verification proves all original actor state bitwise unchanged, graph/model/capture hashes consistent, causal bounded actions, unchanged physical contract, training-only normalization and exported predictions matching across GPU/Mac to3.28e-7 maximum absolute difference. This establishes the implementation boundary, not physical success.

Next: online corrective imitation using this decoder on its own physical states, including current-state ground-wing restoration. Existing recorded-history fit alone is insufficient. The broader motor, utility and multi-fly survival goal remains incomplete.
