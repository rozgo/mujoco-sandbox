# Wingbeat phase, imitation and physical rewards

Discussion audit, 2026-09-14 17:51 UTC. This is a proposed experiment, not a
trained improvement. The user asked whether phase mismatch could punish a
working wing pattern. No training objective or deployed controller is changed
by this note.

## What the implementation actually does

`velocity_hover_ppo.py` compares predicted wing commands against the matching
frames of stored PID demonstrations. During wing-readout PPO, frozen full-core
histories produce the cached features used by that auxiliary squared-error
loss. It is teacher-history supervision, not an independent PID clock used to
score the live student's wing angles. The observation includes measured wing
angles and speeds. Consequently, this is not proof that current labels are
inconsistent or that phase mismatch caused the observed flight regressions.
The constraint may nevertheless oppose a useful change to the wing pattern.

The separate physical reward scores survival, averaged body velocity, averaged
angular velocity and uprightness. It has no teacher phase comparison. Its
100 ms velocity average can hide rapid back-and-forth body motion; mean error
and body-motion variation must be inspected separately. This averaging is for
scoring only. Wing forces still read measured wing angles and speeds every
1 ms physics step.

`VelocityPID` has a useful existing separation: velocity feedback produces a
desired acceleration, then `HoverPID.action_for_acceleration` converts this to
bounded wing commands using a 30 Hz time-based pattern. The integrators and
clock are teacher state, absent from the deployed actor.

## Recommended order of experiments

1. Use retained run 05 and vary only the auxiliary imitation weight, preserving
   the original reward, physics, brain, starts and PPO settings. Compare the
   existing anchor against zero anchor, with matched random sampling and
   physical evaluation. This tests whether the anchor currently constrains RL.
2. If preserving a motion reference is useful, compare short complete wingbeat
   windows with one shared phase offset across all six wing channels. Include
   motion direction and amplitude; reject collapsed/stationary patterns. Do not
   align each wing independently, arbitrarily stretch time, or shift command
   transitions. Startup and recovery timing remain physically consequential.
   A trajectory loss needs actual student histories; merely relabeling recorded
   teacher inputs with unrelated shifted actions is not a valid implementation.
3. Consider PID guidance on the physical effect only after isolating the anchor.
   Query the feedback part from student state/history and compare desired versus
   achieved force/torque or velocity change over a bounded short horizon. Account
   for gravity, drag, reference frames, achievable actuator authority and actual
   integration. The logged wing wrench includes resistance terms, so it cannot
   be treated as pure propulsive force without decomposition. Maintain consistent
   teacher integral state or explicitly declare a stateless feedback reference.
   Validate the guidance on accepted PID trajectories before adding it.

Standard MuJoCo rollouts are not differentiable through PyTorch. A penalty on
measured physical outcomes feeds PPO's reward/advantage calculation. It does
not become a supervised action loss merely by calling backward on a recorded
force. Phase-aligned action imitation can provide a direct supervised gradient,
but requires correctly reconstructed student recurrent context.

Score counterexamples before another extended run: a common steady-state phase
shift should leave motion similarity nearly unchanged; incorrect relative wing
coordination, amplitude loss or stopped flapping should not receive the same
score. Good net velocity with excessive body bobbing should also be detected.
Commands and task response times must remain unshifted. No runtime PID, supplied
oscillator, new actor output or bypass around MaleCNS is proposed.

## Primary research context

- [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf),
  sections 5.1 and 11, explicitly supplies reference phase and identifies fixed
  timing as a limitation for recovery. Our current actor has no supplied clock.
- [AMP](https://xbpeng.github.io/projects/AMP/AMP_2021.pdf), sections 5 and 8,
  learns motion priors without synchronized reference phase and notes that
  unsynchronized similar motions can have large pose errors. It establishes
  relevant precedent, not evidence that an adversarial discriminator is needed
  for this fly or would improve its flight.
- [DAgger](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf), section 3,
  obtains expert labels on states visited by the learner. This addresses state
  distribution mismatch; it does not automatically resolve a teacher clock or
  hidden-state mismatch.

The run-09 recovery-practice trial completed four ten-second flights but regressed
to approximately 161 mm climb and 18.69 mm/s total velocity RMS. Run 05 remains
retained. These results do not establish phase mismatch as the cause. The overall
RL flight-improvement objective remains unfinished.
