# Nonlinear motor calibration improves fitting, not autonomous flight

Source `d6cb676`, seed 58001, readout01 parent. Only the existing nonlinear motor
decoder changes: **230,572 parameters**, no new deployed parameters. Every
upstream actor parameter and buffer remains bitwise identical. Cache inputs are
the actual 815 motor-cell states; raw sensors and labels never bypass the graph.
Whole-episode exclusions, complete variable-length histories, model/state hashes
and parent-decoder reconstruction were verified.

Ground/original-flight/startup feature replay totals: **22.067460 / 13.419874 /
4.457373 s**, including actual graph replay **16.809685 / 8.674531 / 1.218329 s**.
The ground cache retains both complete 3,000-frame stop/resume histories.
Parallel neural sequence counts are **8 / 8 / 64**. No physics is newly integrated
during this replay. Full corpora have 30,400 frames; training uses 21,800.

RTX 4090 fitting takes **60.000718 s**, setup **4.956973 s**, evaluation/save
**0.020001 s**. It makes **49,201 updates**, split 24,565 ground / 12,396 original
flight / 12,240 startup, with 1,024 examples per update. **50,381,824 sampled
examples reuse 21,800 distinct training frames**; zero live physics worlds.
Peak CUDA allocation is **156,993,024 bytes**, separate from graph replay.

Excluded original-flight wing MSE improves **0.006163→0.002702**; startup-corpus
MSE **0.034882→0.004972**. Ground non-wing retention MSE ends at **0.000366**, so
leg outputs are not exactly preserved. Both unassisted flights fall (hover/forward
root RMSE **7.982/14.840 mm**). First-30-ms sampled upward passive force is only
**0.1608 body weight**, versus 0.9630 for the expert; wing angles remain excessive.
These are sampled control-boundary forces, not substep-averaged lift estimates.

Five fixed ground cases stay stable; slow walking falls. Continuous walk/stop/
resume stays upright with permitted support; walk/resume gates pass and stopping
fails, with **0.842 mm** late displacement. All evaluations have zero warnings.
Do not promote this checkpoint. It is a better supervised fit and a starting
point for further diagnostics, not learned flight or established robust walking.
