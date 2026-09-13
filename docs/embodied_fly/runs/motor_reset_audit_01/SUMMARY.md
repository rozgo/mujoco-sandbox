# Frozen checkpoint fails with the training batch size too

Initial observations exactly match the saved evaluation. Constructing the
training reference changes neither physical state nor actor inputs. First-action
differences across load/mode/batch comparisons are below 4.18e-7; subsequent
contact dynamics can amplify roundoff, so trajectories are not identical.

All 11 copies of the standing start fail at 0.494 seconds; all 10 hovering copies
fail at 0.332 seconds. All 11 walking copies remain upright for this one-second
probe. These are duplicate starts, not 32 independent generalization trials.
The physical failure persists in the training batch size with frozen final
weights. Successful episodes under changing training weights do not establish
fresh-start stability of the saved checkpoint.

12.548267 seconds setup; 11.961960 seconds frozen rollout. 16,000 physical action
transitions, 32 aggregate simulated seconds, zero learning updates, zero warnings.
Same canonical body and graph; no teacher actions are executed. Exact recorded
states, action feedback and hashes are verified. The audit does not identify a
unique training cause; it rules out a changed initial observation or a failure
that exists only in the smaller evaluation batch.
