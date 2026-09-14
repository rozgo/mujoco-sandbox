# Longer physical credit for the learned wing rhythm

User approved this next PPO trial on September13 (local time), after preferring
the independent PID-imitation actor's wing motion. Observed effort start:
2026-09-14 00:01:59 UTC. Preserve that checkpoint and every earlier comparison.

Parent: assets/embodied_fly/diagnostics/pid_imitation_01_teacher_stage.pt,
SHA256 315ef3a259ed0e74757163bf58ab22feabbc2049d24992a5028d6802805df1d3.
One continuing399-input/78-output MaleCNS actor; fixed graph, body, instantaneous
wing-force model,1,000 Hz physics and500 Hz actions. No PID, action mask, added
oscillator, body-force controller, supervision or utility training.

## Deliberate changes from pid_imitation_ppo_01

- Rollout512 ->2048 actions:1.024 ->4.096seconds of physical experience per world.
- Discount gamma.999 ->exp(-.002/5): five-second exponential discount timescale.
- GAE lambda.995 ->exp(-.002/2)/gamma: two-second combined gamma*lambda trace
  timescale, formerly.333seconds. Trace decay is not a hard horizon; critic
  bootstrapping still carries later returns.
- Episode length5 ->10seconds, matching the frozen review duration.
- Requested training300 ->600seconds to allow physical collection and actor
  updates with these longer rollouts. Finish the current rollout/update and
  report any budget overrun. This is a bounded development trial, not a matched
  compute benchmark or a single-variable attribution experiment.
- Seed120403. This is an independent training run from the selected parent.

Keep64hover worlds,16CPU MuJoCo/mjbatch threads,RTX4090 neural training,
actor LR1e-7,critic LR1e-4,std/floor.001,two epochs,KL limit.03,entropy0.
Keep128-action recurrent backpropagation chunks (.256seconds) with activation
recomputation. Physical/neural state persists across chunks; gradient truncation
is unchanged. Longer return credit does not mean four-second full-graph BPTT.

Same bounded physical reward as pilot06/prior PPO, including20mm/s vertical
speed scale. No new reward weight or wing-frequency target. Existing near-nominal
reset curriculum is unchanged; harder offsets remain gated on good tracking.
Because the parent is imitation and the discount changes, use fresh PPO/critic
Adam and a new critic. Four critic-only rollout updates are included in measured
training time, with the actor fixed. Their larger experience count is disclosed.

## Review and decision

Reuse the matched frozen ten-second evaluation: nominal start,0.2mm lower with
10mm/s downward speed, and0.2mm lateral with5mm/s lateral speed; plus independent
accepted PID. Same source/body/starts for before and after. Compare full-window
and after-first-second position/height errors, all failures, and seconds6–10
wing frequency, sweep range and repeated height ripple. The non-nominal starts
test small-error behavior; surviving alone does not establish recovery.

Success for retaining this trial: all three starts remain airborne; nominal
position RMS and altitude RMS both improve at least10% over the preferred
parent, without either perturbed case worsening by more than10%; settled
wing frequency stays25–35Hz and repeated height ripple remains below0.2mm.
These are predeclared development-progress criteria. The original stricter
PID-quality gates remain unchanged and must be reported separately.

If the progress criteria fail, preserve the promising imitation parent and
report why; do not silently promote a longer-trained actor. Render/open both
PID/PPO and parent/PPO comparisons. Archive timings, source, graph/body/optimizer
checks and failed traces. No additional motor stage starts in this round.
