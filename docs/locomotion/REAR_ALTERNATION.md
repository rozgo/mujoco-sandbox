# Rear-leg phase diagnostic after visible-step review

The user called the new visible-step video almost perfect, then identified that
front-left removal alternates the rear feet more naturally than front-right
removal. A second screenshot identifies BOTH lower-FR and whole-FR. This is a
request to diagnose and recommend the smallest next change; no new training in
this diagnostic follow-up.

Measured the actual saved final-video poses at 20 ms, after 1 s, with 0.5 ms
MuJoCo physics. Use upward crossing of 1 cm physical foot-bottom clearance to
measure RR swing onset within consecutive RL cycles. Zero/one is synchronous;
half a cycle is alternating. One preselected demonstration per body:

| Case | Rear swing phase separation | Both rear feet >1 cm clear |
| --- | ---: | ---: |
| lower-FL | 58.2% cycle | 0.0% time |
| whole-FL | 58.7% cycle | 0.0% time |
| lower-FR | 10.6% cycle | 6.9% time |
| whole-FR | 6.5% cycle | 7.8% time |

Circular phase concentration is 0.990–0.996, so the repeated difference is
consistent within these recordings. This confirms the user's observation;
it does not prove the same fractions for arbitrary starts. The full report
with trajectory hashes is [REAR_ALTERNATION_DIAGNOSTIC.json](REAR_ALTERNATION_DIAGNOSTIC.json).
Reproduce with `scripts/diagnose_rear_phase.py` after regenerating the named
recording directory.

The visible-step objective scores each foot independently. The no-support cost
only applies when all allowed supports lose force; both rear feet can swing
together while the remaining front foot supports the body. Equal mean stride or
duty durations also do not establish alternating phase.

## Smallest proposed next experiment

Keep the current selected checkpoint/video. Add one small soft cost for
simultaneous swing of two intact rear feet on front-damaged bodies, identically
for left/right and lower/whole removal. Both rear feet planted together remains
allowed. Keep the successful lift, stride and support objectives. Exclude bodies
with a missing rear foot from this pair-specific cost. Do not prescribe a clock,
force exact 50% phase or mirror joint poses within one damaged body.

A 60–90 second fine-tune would test this single change, with all nine bodies still
in the batch and all existing lift/task/healthy-gait gates retained. Require the
FR rear-swing overlap to decrease without sacrificing any intact foot's lift or
regressing the already alternating FL cases. This is a proposal, not an executed
training round or a guarantee of improvement.

For context, [Isaac Lab's Spot gait reward](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/rewards.py)
separately scores synchronized and asynchronous foot contact timing. Its full
reward explicitly requires two pairs of feet, so it cannot be copied unchanged
onto a three-legged body. The proposed overlap cost is a smaller local change.
