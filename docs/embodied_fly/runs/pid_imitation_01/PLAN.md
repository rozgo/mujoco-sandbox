# Teach the accepted PID wing pattern, then return to PPO

User-approved September13: use PID as teacher, imitation as training, followed
by PPO and an independent actor-only review. Observed work start23:20:04 UTC.
Preserve pilot06 and its9.25 Hz/2.80 mm settled oscillation as the before case.

Use the same399-input,78-output MaleCNS actor and accepted instantaneous1 kHz
physical fly. Both teacher and student act at500 Hz. Call the existing HoverPID
implementation on read-only scratch states, with separate integral memory per
world. Its30 Hz clock/integral are training-only and never become actor inputs.
No new network layer, motor mask, body-force controller or physics change.

Start from hover_only_06. Run300 requested seconds of online imitation with
32 physical worlds,16 CPU physics threads,RTX4090 neural work,sequence64,
activation recomputation,Adam LR1e-4 and gradient clip1. Loss is mean squared
six-wing command error plus0.1 times mean squared other-joint error against
the PID's initial-pose commands. All existing motor and neural-core parameters
learn; graph topology, routing, utilities and observation normalization stay fixed.

Teacher action share is1 for the first half of measured training time, decreases
linearly to0 by80%, and remains0 for the last20%. Only bounded actuator commands
are mixed. Targets always come from PID on the actual physical state. The final
teacher-free section still receives supervised labels, so it is imitation, not
PPO. Exact schedule and every executed/teacher/student action are recorded.
Save the end-of-full-teaching checkpoint and final handoff checkpoint separately.

Half the worlds use exact nominal cold-wing starts; half have reset-only height
offsets +/-0.2 mm and vertical speeds +/-5 mm/s. Episode lengths3.2–4 seconds
desynchronize wing phases. Reset physical/neural/teacher memory together only at
declared failure/episode end. Record complete episodes' actual teacher share and
retain all failed traces, including post-action failure states. Seed120401.

The imitation optimizer is fresh. Remove stale PPO/critic optimizer states from
the new actor checkpoint; save imitation Adam under its own key. Next PPO starts
fresh optimizers/critic while retaining learned actor weights. Do not present
this as uninterrupted pure RL or train-from-scratch learning.

Evaluate final actor on the same three ten-second starts plus accepted PID.
If handoff collapses, evaluate the retained teaching-stage checkpoint before
choosing a starting point for PPO. These are development/model-selection cases.
Run one300-second PPO continuation on64 hover worlds with the existing bounded
reward (vertical-speed scale20 mm/s),LR1e-7,criticLR1e-4,std/floor.001,
horizon512,sequence128,two epochs,gamma.999,GAE.995,KL.03,entropy0,
four critic-fitting rollouts andfive-second episodes. No teacher in PPO.
Preserve a failed continuation as evidence; do not promote survival as precision.

Render/open actor-only PID comparisons after imitation and PPO, and before/after
when useful. Keep1x,matched cameras/scales,source/checkpoint/body hashes,all
physical failures and measured compute times. Utility and next motor tasks wait.
