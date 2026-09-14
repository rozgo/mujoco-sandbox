# PPO hover: enforce the proposed update limit

The user authorized continued improvement after a failed pilot. Run 01 completed
604.37 training seconds but its final actor fails all four starts at 0.47 s.
Its pre-update KL monitor stops additional minibatches while retaining the first
oversized Adam step; observed KL frequently exceeds the intended 0.02 by orders
of magnitude. This is a concrete optimizer-control problem, not evidence that
PPO or the fixed connectome cannot learn hover.

Return to the same original `velocity_imitation_30m_01.pt` parent and its exact
physical contract. All rewards, 32 worlds/16 CPU physics threads, observations,
commands, network, action distribution, imitation anchor, critic and timings
remain those declared in run 01. Reuse its hash-checked pre-update PID and
parent captures, with no extra training or re-recording of those baselines.

Intentional change: make PPO updates bounded in practice. Initial actor LR is
3e-7 (one tenth of run 01). After each proposed step, recompute the same recurrent
minibatch and measure analytic old-to-new Gaussian KL, summed over all 78
channels; the common tanh mapping cancels. Reject proposals above 0.02, restore
weights AND Adam moments/step counters, halve LR and retry the same gradient
up to eight times. Keep only accepted steps. Stop an epoch when the current
minibatch's analytic KL already exceeds 0.015, reserving room for a proposal.
Restore the declared initial LR at each new on-policy rollout so late-minibatch
backtracking cannot permanently collapse learning. Log every attempted and
accepted step. This is a measured minibatch constraint, not a guarantee on all
possible physical states.

Again allow 600 measured training seconds, with deterministic midpoint and final
ten-second physical evaluations on starts 0,1,8,9. The same milestone applies:
four ten-second survivals and at least 20% lower velocity RMS on the matched
first two seconds than the original parent. Preserve failure cases, checkpoint
ancestry and the prior video. Subsequent changes depend on physical evidence.

Command (headless Linux adds MUJOCO_GL=egl):

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_hover_ppo \
  --checkpoint assets/embodied_fly/diagnostics/velocity_imitation_30m_01.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_teacher_dataset_01 \
  --output outputs/embodied_fly/velocity_hover_ppo_02 \
  --baseline-dir outputs/embodied_fly/velocity_hover_ppo_01 \
  --seconds 600 --lr 3e-7 --bounded-updates
```
