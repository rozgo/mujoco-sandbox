# Hover is sensitive to initial conditions as well as exploration

The fixed parent is position_sustain_retention_01. Three predetermined new
starts, each paired across noise 0, .003 and .01, produce 27 five-second cases.
Standing and walking remain stable in all 18 corresponding cases. Hover fails
in all nine, including all deterministic controls. No parameter updates, critic,
teacher, live resets, or physical changes occur. This establishes a limitation
of the current parent beyond PPO, not proof that action noise caused every fall.

The companion recorded-start test restores the successful video's exact three
initial physical states and repeats the matched comparison. Hover remains
airborne at noise 0 (6.105 mm root RMSE) and .003 (6.735 mm), but fails at
2.768 seconds under .01 noise. This is evidence for reducing exploration while
retaining initial-state fragility as a separate open issue. One noisy trial per
condition does not estimate a population success probability.

Every recorded state, action, causal feedback channel, initial-state match and
checkpoint/parameter hash is verified. The slight deterministic trajectory
difference from the original review occurs with nine instead of three parallel
worlds; the recorded initial physical states are identical. These are diagnostic
comparisons, not new policy acceptance claims.
