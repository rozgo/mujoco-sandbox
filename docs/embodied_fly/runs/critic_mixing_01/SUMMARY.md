# Shuffling learns the shared return profile, but misses the full gate

The fixed-data diagnostic finishes in 23.105900 seconds on the Mac CPU, without
new physical collection or actor updates. Three matched seeds compare sample
mixing with and without input standardization. Each fit has 960 updates and the
same per-cohort sample count. Test streams are held out completely.

For the safe-noise cohort with original inputs, shuffled transition batches
reduce median held-out RMSE from 0.6660 to 0.1022 and improve explained variance
from 0.0001 to 0.9739. Thus the old contiguous-time minibatches were a significant
obstacle to learning the common return profile. However, after removing the
shared time profile, RMSE rises from 0.0626 to 0.0733. This misses the predeclared
5% regression limit. **The complete diagnostic progress gate fails.**

Input standardization does not help this comparison. On the safe cohort, its
shuffled median explained variance is -0.8454. For the cohort including failures,
original-input shuffling improves explained variance 0.0192 to 0.3335, below the
required 0.5. All results, including each initialization, remain in diagnostic.json.
These are measured four-second returns, not infinite-horizon PPO value accuracy.

The result supports shuffled critic batches while preserving the claim's limits:
it does not yet establish reliable discrimination among nearly identical flies
at the same moment. Future action noise also contributes irreducible uncertainty
to these individual realized returns. No actor is promoted and no hover gate is
relaxed. The user's new paired-direction/return-to-origin curriculum motivates
more varied physical experience; it is a separate curriculum hypothesis, not a
reinterpretation of this failed gate.

Six focused critic tests pass in 1.30 seconds. The full package passes 185 tests
in 155.06 seconds, with 45 dependency warnings. Source commit and reproducible
configuration are in diagnostic.json. Original batching remains the default;
`--critic-shuffle-transitions` opts into the tested time/world mixing.
