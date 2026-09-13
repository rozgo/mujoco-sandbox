# Faster sensory adaptation — rejected

From angle01, preserve Adam moments and use 3e-5 actor learning rate with 3e-3
only for the existing 12→128 sensory extension. Same 395-input MaleCNS actor,
three corpora, 32 offline sequences, 64 burn-in / 32 supervised steps, seed 49001.

Measured learning: **60.607377 seconds**, 83 updates, 84,992 supervised examples;
6.295566 seconds setup and 1.587431 seconds validation. Peak allocated CUDA memory
9,586,650,112 bytes. Offline flight MSE improved 0.046076→0.035415; this did not
establish better physical control.

Both 0.3-second airborne tests fell (hover/forward root RMSE 8.310/15.625 mm).
Five of six fixed ground cases stayed stable; ordinary walking fell. All three
continuous walk/stop/resume phases were unstable and support-invalid. No MuJoCo
numerical warnings occurred. Retain the full failure and checkpoint; keep angle01
and online01 references. Next learning uses physical outcomes, not another
identical imitation continuation.
