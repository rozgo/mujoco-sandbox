# Startup weighting alone does not establish flight

Source `164c7f3`, parent readout01, seed 54001. Only the existing 1,542 wing
output parameters change; every other parameter and non-wing output row is
verified identical. Forty percent of each 1,024-example minibatch is drawn from
the first 25 frames (5 ms) of training episodes; the remainder is uniform over
all training frames. Per-axis loss uses training-target standard deviation,
with a 0.1 floor. Excluded whole episodes never contribute to normalization.

The RTX 4090 fit took **10.000220 s**, setup **1.385319 s**, evaluation/save
**0.035597 s**: **16,875 updates**, **17,280,000 reused samples**, **9,000 distinct
training frames**, zero live physics worlds, **31,005,184 bytes** peak CUDA
allocation. Overall excluded-episode MSE worsens **0.006163→0.007537**. Startup
pitch errors decrease, while startup roll errors worsen. The report's generic
corpus-sampling sentence is stale; the explicit startup fields accurately report
the implemented mixture. Later source corrects the generic sentence.

Both unassisted original-physics flights fall (hover/forward root RMSE
**12.296/11.991 mm**). Six fixed ground cases stay stable with permitted support,
but fail raw tracking gates. Continuous walk/stop remain stable; resume topples.
The stage masks wing outputs and all non-wing parameters remain identical, so
these ground differences cannot establish a learned ground change. Preserve all
trajectories and the earlier selected ground review; do not promote this fit.
