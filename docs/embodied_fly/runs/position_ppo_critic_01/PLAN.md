# Independent critic continuation

User request: continue from sustained hover and fix the critic. Start from
`position_ppo_timing_02.pt` (SHA256
`1b926a9e1489d1792177519b3b7de6d56cc2723882077daf7ff7a1bf1ad0fd1e`).
Preserve its actor, critic, both Adam optimizers and learned exploration state.

Run **600 seconds**, **64 worlds**: 16 stand, 16 walk, 32 hover. Keep 16 CPU
MuJoCo/mjbatch physics threads and RTX 4090 neural training. Keep the same
wing_position body, frozen measured edges, actor/critic architectures,
observations, 78 actions, rewards, reset curriculum and fixed episode commands.
No imitation, teacher, output mask or runtime helper. New run seed99703;
live physical/neural episodes reset, and the existing reset-widening schedule
runs over the first ten simulated seconds per world as in prior continuations.

The single intended training change is `--independent-critic`. Save the critic's
existing normalized-observation and preceding descending-state features during
physical collection. The actor retains its policy-KL early stop. The critic
then completes both passes over the entire saved rollout, even when the actor
has stopped. This uses the actual collection features and fixed GAE targets;
no actor recurrent replay or new physics is needed for value fitting. A separate
seeded shuffle stream avoids perturbing the actor's sampling stream. Feature
storage adds approximately 214 MiB at this batch size.

Keep 512-action rollouts, 128-action recurrent sequences, .999 discount,
.995 GAE lambda, two epochs, actor LR1e-6, critic LR3e-4, .003 initial/minimum
pre-tanh exploration, KL limit .03 and five-second episodes. No repeated
critic-only warmup. Eight critic minibatch updates are now expected per complete
rollout instead of the previous one. Log sample presentations, policy stops,
critic fit before/after, per-task errors and separate critic training time.

Test forced actor-KL rejection while the critic still completes all updates;
verify detached inputs/targets, unchanged actor during critic-only work, correct
saved-feature values across resets and checkpoint/optimizer continuation. Run
the relevant suite before the measured GPU run and save its source commit.

Evaluate the same nominal five-second three-command review, then the existing
three additional ten-second starts (98103,98113,98123) if nominal hover sustains.
Compare against the already captured timing02 cases; no redundant parent run.
If nominal hover fails, retain the complete failure and omit additional starts.
Record and automatically open the new full three-command video in either case.

Preserve all results. Primary improvement is continued flight with lower vertical
oscillation; track height span, vertical-speed RMS and target-altitude error on
matched airborne windows. A better fit to training returns is not proof of a
better controller. Do not promote a failed candidate or obscure ground regressions.
