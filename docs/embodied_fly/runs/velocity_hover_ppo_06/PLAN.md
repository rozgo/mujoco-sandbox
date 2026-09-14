# Unchanged PPO continuation focused on vertical regulation

User approved another ten-minute continuation on September 14. Goal/start recorded
at **14:52:05 UTC**. Begin from run 05 final, SHA256
`3106cb363aa75dab78322181ad9f46c229a8a29ca8f1f0599bc18e8d59763a32`.
Preserve this parent and all earlier captures/checkpoints.

No recipe changes: 32 worlds, 16 CPU MuJoCo/mjbatch physics threads, RTX 4090
neural inference/learning, 1,000 Hz physics, 500 Hz actor, four internal neural
updates. Same 391 observations, four zero velocity commands, 78 actor outputs,
FlyBody, measured graph and wing-motion force law. Train only the existing
105,222-parameter wing readout; retain frozen upstream weights. PPO horizon 512,
sequence 64, two actor epochs, four critic epochs, proposed LR 3e-6, analytic
minibatch KL ceiling .02 with rollback, fixed .003 exploration. Same reward and
light PID-label anchor. Restore actor Adam, trained critic/Adam and actor sampling
RNG. As in run 05, physical episodes reset and critic sample shuffle restarts
from the declared seed. No new warmup or smoke is needed for this unchanged,
already-verified continuation path; normal first-rollout replay audit still runs.

Budget: 600 measured training seconds, finishing the current rollout. Setup,
replay audit, checkpoint IO, evaluation, rendering and transfer are separate.
Reuse the parent's final physical capture and existing PID reference, checking
checkpoint correspondence and capture hashes. Evaluate the same four starts
(0, 1, 8, 9) at halfway and completion, without exploration or teacher assistance.

Primary outcome: retain four ten-second flights and reduce climb toward **50 mm
or less** in each capture. Report vertical RMS and horizontal RMS separately;
sideways velocity should remain near the parent's **9.3 mm/s** or improve.
A horizontal RMS above 10 mm/s is a material regression for this trial. Report
both midpoint and final, including failures; final is the primary evaluation.
If vertical regulation plateaus, the agreed next candidate is a modest increase
of the existing vertical reward weight with a brief critic adaptation. Do not
change any settings in this running trial.

Record and open a 1x PID / parent / final comparison. Archive timings, provenance,
weights and metrics; update the learning journal and report without promoting
incomplete hover as a solved motor skill.
