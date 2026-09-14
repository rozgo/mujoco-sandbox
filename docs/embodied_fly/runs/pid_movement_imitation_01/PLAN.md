# PID movement imitation 01 — predeclared

User-approved experiment: teach the same fly brain using the accepted PID's
directional movement, braking, reversal and return-to-hover. Earlier imitation
mostly demonstrated stationary hover with small vertical reset perturbations.
Moving targets have been used in PPO, but not yet in PID imitation. This trial
tests the richer-demonstration hypothesis; it does not assume success.

Start from the preferred `pid_imitation_01_teacher_stage.pt` (SHA256
315ef3a259ed0e74757163bf58ab22feabbc2049d24992a5028d6802805df1d3).
Preserve every earlier checkpoint and result. Same 399-input /78-output actor,
graph routing, body, force model, 1,000 Hz physics /500 Hz action interface.
Encoder, decoder and cell dynamics train; utility/intention remain frozen.

## Training

- 64 worlds, 16 CPU MuJoCo/mjbatch physics threads, RTX 4090 neural learning.
- 32 stationary hover worlds and 32 worlds cycling through six directional
  round trips. Requested excursion 1.2–1.8 mm; timing scale 0.85–1.15.
  Full 12-second episodes retain return and settling. Small vertical initial
  offsets/speeds on half the worlds are preserved from the earlier imitation.
- Accepted PID computes labels from each current physical state and current
  target. It executes 100% of training actions throughout this initial pass.
  No elapsed-time handoff. Successful training trajectories describe the PID,
  not the student's independent ability. No PPO, critic, or reward changes.
- Same imitation loss: wing-command MSE weight 1, other-joint posture MSE 0.1.
  Same Adam LR 1e-4 and 64-step recurrent training sequences. Fresh Adam state.
- Requested 600 s training allowance, finishing the active sequence/update.
  Save one intermediate checkpoint after the first update crossing 300 s.
  Record setup, physical/neural collection, optimization and trace overhead.
  Retain all trace hashes and actual supervised presentations/physical steps.

No future route, PID phase clock or integral is added to actor observations.
Measured wing angles/speeds and recurrent neural state remain available. The
teacher's hidden phase/integral and small correction-label magnitudes remain
possible learning limitations; this experiment changes demonstration diversity
and preserves assistance rather than claiming those issues are solved.

## Independent review and handoff gate

Evaluate the intermediate and final actors separately on the same seven nominal
12-second cases, teacher absent, deterministic actions, no live resets. Reuse
the unchanged preferred parent's frozen capture. These are declared development
cases, not held-out generalization evidence. Save both results, including failures.

Keep the existing progress gate: >=10% lower mean six-route position RMS than
the parent, <=5% stationary-hover regression, no new physical failure. Complete
acceptance still requires all intermediate/return windows <0.5 mm position error
and final 100 ms displacement drift <1.5 mm/s, with no physical failures.
Select a new preferred candidate only if it passes progress; otherwise retain
the parent. Do not automatically remove teacher assistance during this run.

Render the final and intermediate checkpoints in the matched seven-case plus
PID-hover layout, inspect/full-decode and open both videos. Document the outcome
before any new training stage. The longer-term movement curriculum stays one
brain; no trained utility or new ground/flight command capability is claimed.
