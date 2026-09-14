# Compare critic minibatch mixing on recorded experience

Start observed 2026-09-14 01:07:57 UTC. Previous goal turn made progress:
it completed, evaluated, filmed and archived hover_explore_01. That actor fails
hover improvement criteria and the preferred imitation checkpoint stays fixed.
Mac and GPU are synced at cff1f0d; no live training process was found.
The complete motor/survival objective remains active and incomplete.

Use the existing frozen exploration capture, not new physical rollouts. Fit the
same critic MLP to complete four-second measured discounted reward windows,
gamma exp(-.002/5), no bootstrap. Keep first 120 anchors (0 through 5.95 seconds).
Hold out complete random streams 12–15 within each noise group. Training uses
streams 0–11. Test streams are never used for normalization or fitting.

Compare contiguous-time batches with batches shuffled across individual times
and worlds. Both visit every training sample once per epoch, with identical
batch size (8 anchor times times training worlds), 64 epochs, 960 optimizer
updates, Adam LR 1e-4, eps 1e-5, gradient clip 1. Compare original inputs and fixed
standardization separately. Standardization uses training anchors 0–4.05s only.
Three fixed initialization/shuffling seeds: 120701, 120702, 120703. Targets remain
in physical reward units. No actor changes, no new physical experience.

Run two cohorts: .001/.003 noise (safe-noise probe) and .001/.003/.006 (including
failures). Report held-out total RMSE/explained variance and residuals after
removing each time's common world mean; the latter exposes inter-world differences.
These are same-start development trajectories, not new-task generalization.

A useful safe-cohort result requires median held-out explained variance >= .5,
plus either a .1 absolute improvement in explained variance or a 20% RMSE
reduction against the matching input-scale/blocked-batch condition. Within-time
RMSE must not worsen by more than 5%. This gate chooses the next fitting recipe;
it does not establish accurate infinite-horizon PPO values or better flight.
If it fails, do not extend a flight run merely because this diagnostic was added.

User steering during implementation: consider gentle directed flight to teach
hover corrections. The proposed next motor curriculum is move then settle,
using the same actor/body and existing target-error inputs. Preserve stationary
hover examples; validate directional control on the accepted plant before any
new physical-training recipe. This diagnostic still comes first.
