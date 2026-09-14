# Recovery starts preserve a real body/brain history

The frozen run11 midpoint generated30 valid snapshots:10 initial conditions
at2,4,6seconds of physical flight. Training samples24 snapshots from starts0–7;
the six from starts8/9 remain reserved for recovery evaluation. These are
initial conditions, not expert action labels or teacher-assisted flight.

Each snapshot includes MuJoCo's `mjSTATE_INTEGRATION`, which covers time,
coordinates, velocities, activations, controls, applied forces and solver
warm-start state. It also saves the existing observation caches, previous
action, wing-force bookkeeping,166,700-cell recurrent state and50-sample
reward history. Restoring a row only initializes that episode. Its physical
time remains intact; the ten-second episode counter restarts at zero.

The reward ring is stored with its next-to-replace sample first, then rotated
to the destination ring index. Other worlds retain their own histories. PPO
replay records the actual nonzero neural state after recovery resets, including
resets inside a recurrent sequence. A regression test deliberately substitutes
zero memory and confirms that the action-probability audit rejects it.

## Verification and the two stopped preparation attempts

The first two GPU preparation attempts stopped before training. They required
near-bitwise closed-loop continuation for128ms; this failed despite every saved
physical/neural/auxiliary value restoring exactly. The first divergent action
was only about2e-7, then physical feedback amplified that numerical difference.
Both failed reports are retained, with their original tolerances and timers.

The third preparation separates the questions:

- Initial state, observation caches, neural memory and reward history restore
  **exactly**, with zero error across every saved field.
- Replaying the same recorded actuator commands for64 control steps gives
  **zero observation, qpos and reward error**, across all30 states. This isolates
  physical restoration from neural arithmetic.
- Running the GPU neural controller again produces a maximum initial action
  difference **1.78814e-7**. After128ms, the largest body-position difference is
  **7.51895e-5cm =0.752micrometres**, and the largest action difference is
  **9.75188e-6**. This is recorded separately from exact replay.

Acceptance retains exact initialization and tight recorded-action replay
(qpos1e-10; observation2e-6; reward2e-7). The initial neural action must remain
within2e-6. Subsequent closed-loop bounds are1micrometre body-position error,
5e-4 maximum qpos-component error,1e-4 action error,.01 normalized observation
error and1e-5 per-step reward error. These are diagnostic tolerances, not added
state filters, changed forces, or loosened hover-performance criteria.

The physical/neural model is unchanged throughout. The approved bank SHA256 is
`11ee5c15d9b572fa609da9d415bc849fa49e226d3fbc342b371c7bc4d4ee672f`.
The exact bank is retained in
`assets/embodied_fly/recovery_starts/hover_recovery_starts_03/` through Git LFS.
The first PPO rollout also passes the existing full-core and cached-readout
likelihood audits, without changing their tolerances.

Preparation03:setup10.231255s,collection23.324448s,restoration audit1.868616s.
It generates30,640world/action transitions and audits1,920 neural-control
transitions plus1,920 recorded-action physics transitions. These costs are
separate from PPO training. Preparation01/02 remain in their archived reports.
The immutable03 report's `audit_transitions` field counts neural steps only;
the recorded-action replay adds1,920 more. Later collector code reports both
parts and their total explicitly; the original bank report/hash stays intact.
