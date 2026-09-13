# More startup phases improve prediction, but flight still fails

The original corpus has only six training reset phases. Added **64 × 20 ms**
expert-controlled starts with independent random wing phase, seed 55001. All
64 meet the demonstration envelope: **6,400 frames / 1.28 simulated seconds**,
**4.150386 s** setup, **10.873399 s** collection. This is inherited-expert data,
not student flight. The whole-episode split excludes 16 of the new starts.

Frozen readout01 replay on **64 neural sequences** takes **4.049921 s** total,
including **1.198871 s** actual graph replay. The same 1,542 existing wing output
parameters are calibrated using equal original/new corpus sampling, 40% startup
draws, training-only axis scaling, learning rate 0.001 and seed 56001. Source is
`4a35d6c`. No sensor bypass, module, neural timing or physics change is introduced.

RTX 4090 optimization takes **10.000448 s**, setup **1.359881 s**, evaluation/save
**0.040237 s**. **16,517 updates / 16,913,408 reused samples / 13,800 distinct
frames**, zero live physics worlds during fitting, **37,548,544 bytes** peak CUDA
allocation. New-corpus excluded-episode MSE falls **0.034882→0.010548**; original
full-window MSE worsens **0.006163→0.009098**. Original excluded startup pitch
MSE improves from 0.1312/0.1416 to **0.0276/0.0293**, with roll improving too.

Both declared unassisted flights still fail (hover/forward root RMSE
**10.012/11.124 mm**), with ground contact and zero numerical warnings. All
upstream actor parameters and non-wing output rows are verified identical to
readout01. No additional ground rollout is claimed for this fit. Preserve it as
evidence that phase coverage helps supervised prediction; do not promote it as
flight. The subsequent paired memory handover shows why startup alone is not
the remaining explanation.
