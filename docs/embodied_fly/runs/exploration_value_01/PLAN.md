# Exploration and critic diagnostic before the next PPO trial

Continue the active fly goal's motor-learning stage. Previous goal turn is
progress: the longer-credit trial, failures, video review and preferred parent
were measured, archived and synced. No current training job is live. Observed
start2026-09-14 00:30:01UTC. Accurate hover and the later motor/survival stages
remain incomplete; this diagnostic does not redefine the full goal.

Freeze the preferred imitation actor315ef3a... and accepted body/clocks. Run64
independent ten-second worlds:16each at latent action std0,.001,.003,.006.
Use16independent random streams repeated across noise levels for a paired sweep.
No actor training, no teacher and no live resets. Preserve the first physical
failure and absorb rewards thereafter while recording the full trajectory.
Use the current20mm/s vertical-speed reward scale, unlike the older hover_audit
default of50mm/s; record the complete recipe. This is an explicit diagnostic
and does not change earlier reports or any training recipe.

Save causal critic features every50ms, measured rewards every2ms and wing/state
traces. Compare noise survival, returns, position/altitude error and wing motion.
For a candidate increase prefer.003 if all16worlds survive and mean return is
at least90% of the.001 group. The.006 group is a further sensitivity check,
not a reason to maximize noise. No candidate is guaranteed safe beyond this probe.

Then fit two identical fresh critic networks to four-second discounted measured
reward windows (five-second discount decay), with no bootstrap.1000Adam updates,
LR1e-4,batches512,gradient clip1, same minibatch order and initialization. One
fits physical targets; the other fits training-mean/std normalized targets and
folds that affine transform back into the last layer. The latter does not alter
architecture or actor. Hold out complete random streams12–15 across nonzero
noise groups; exclude deterministic control duplicates from the test set.
These are diagnostic finite-horizon fits, not deployed PPO critics. Report both
fit times separately from physical collection and from actor training (zero).

The saved critic from hover_credit_01 may also be evaluated on these causal
features for descriptive correlation/activation saturation. It estimates a
different policy/horizon, so do not call its absolute finite-window RMSE PPO
value accuracy. This check asks about learnable variation, not generalization
to new tasks. Choose the next bounded PPO change from the evidence; record any
change before executing it. Preserve the preferred parent and all failures.
