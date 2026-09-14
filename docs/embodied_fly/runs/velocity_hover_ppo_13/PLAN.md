# Recovery-start PPO from the retained hover policy

User-approved effort began **2026-09-14 19:56:14 UTC**. Implement the preceding
run12 recovery plan with the run11 midpoint parent, SHA256
`69061012aa237832fcbd42be2b39bc9c2e6530932865b82d4df5190af9f56077`.

Use32 worlds:16 ordinary cold starts,16 starts from the frozen parent's own
valid flights at2,4,6seconds. Generate24 training histories (episodes0–7) and
six separately held-out recovery histories (8–9). Store full MuJoCo integration
state, historical sensor values, previous commands, wing-force bookkeeping,
full neural memory and causal reward history. Verify restored continuation
against uninterrupted flight before optimization, including world isolation
and reward-ring alignment. Saved physical time remains; episode elapsed time
restarts. These assignments occur only at episode initialization.

Retain.0015 exploration,.005 KL limit, original reward, wing-readout-only PPO,
actor/critic optimizer state, light imitation, all78 output channels and the
same graph/physical model.1kHz CPU MuJoCo/mjbatch physics,500Hz CUDA full-brain
control,16CPUphysics threads,RTX4090. No gusts, added control help or reward change.

Run108rollouts/1,769,472transitions (~10minutes actual training), with ordinary
four-start10-second evaluations after54 and108rollouts. Separately compare
recovery from all six withheld histories. Record preparation, restoration
checks, training, evaluation, rendering and total effort times separately.

Retain only if ordinary evaluation improves total velocity RMS below12.76mm/s,
horizontal RMS below10mm/s, climb below50mm and allfour complete10seconds,
without a worse startup dip/error. Recovery tests must also reduce measured
drift. Keep every failure and checkpoint. If this block does not improve the
physical outcome, pause further PPO extensions and diagnose phase-conditioned
control authority through the existing learned interfaces. End the report
with that decision and a concrete next action.
