# State readout 02: corrections improve fitting, physical tasks still fail

Source `684e273`, parent `state_hover_retention_02`. The two actual actor histories contributed 116 and 822 pre-fall frames; post-fall frames were excluded. Fitting used 6,758 frames and 1,680 temporally related validation frames. Startup examples received the declared fourfold weight. Only 1,542 existing wing-output parameters were eligible; all upstream and non-wing weights remain unchanged.

The GPU job took 76.765293 s total: 61.235897 s setup/current-state labeling, 13.663314 s frozen neural replay, 1.866078 s fitting/packing/save. Five neural histories, 12,500 processed observation frames including excluded history, zero new integrated physical transitions. Peak allocated CUDA memory was 664,091,648 bytes. Selected ridge regularization: 1e-5; parameter change L2 norm: 0.9103.

Reference-history and correction-history errors improve, but correction validation MSE remains 0.04362 and 0.01664. A recorded near-limit example still receives predicted sweep commands -0.108/-0.072 instead of the positive 0.471/0.479 target. This supports investigating the readout's feature representation; it does not establish a unique cause.

Independent five-second-per-command evaluation, seed 72005: 7.076312 s setup and 32.581429 s capture. Standing remains upright; walking and hover fail. Standing wing RMS is 0.1189 rad and peak 0.3968 rad. All strict gates fail with zero numerical warnings. The candidate is not promoted; retention02 remains the development parent.

The [complete video](../../../../previews/embodied_fly/state_readout_02_all_tasks_v1.mp4) retains all outcomes: 15 s, 750 frames, 1600×900, 50 fps, 1×. Fully decoded, visually inspected and opened automatically.

Independent verification checks masks, causal bounded actions, frozen state, model/capture/checkpoint hashes and the fitted coefficients. Cross-host ridge recomputation differs by only a few billionths; the helper's original 1e-9 absolute tolerance was too tight for the float32 inverse-tanh input followed by cross-platform linear algebra. A declared 1e-8 coefficient tolerance passes; physical gates were unchanged. Exact source delta coefficients are used to verify checkpoint export.

A subsequent I/O-only fix materializes each NPZ array once instead of decompressing it per frame. The two full label extractions on Mac take 0.104 s and 0.194 s and produce labels bitwise equal to those used in this GPU fit. Five focused tests pass in 3.94 s. This is not a matched cross-machine speed benchmark.
