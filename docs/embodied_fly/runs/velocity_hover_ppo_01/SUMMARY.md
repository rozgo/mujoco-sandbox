# Velocity hover PPO: first pilot regressed

The first PPO pilot completed **604.368 s** on the unchanged agile flight plant.
Its final actor fails all four predetermined starts at **0.470 s**. Preserve the
original imitation parent; do not promote this checkpoint as improved hover.
The [43-second comparison](../../../../previews/embodied_fly/velocity_hover_ppo_01_comparison_v1.mp4)
shows the PID, frozen parent and final actor at 1x, including explicit failures.

| Start | Split | Parent airborne | Midpoint airborne | Final airborne |
|---|---|---:|---:|---:|
| 0 | Training | 10.000 s | 2.434 s | 0.470 s |
| 1 | Training | 6.772 s | 2.434 s | 0.470 s |
| 8 | Held out | 10.000 s | 2.434 s | 0.470 s |
| 9 | Held out | 4.190 s | 2.434 s | 0.470 s |

Ten seconds is the evaluation ceiling. The parent's historical 5.6-second result
came from the full mixed-command exercise. These new captures keep the command
zero throughout. Parent first-two-second velocity RMS is about 25.96 mm/s;
midpoint is about 32.77 mm/s. The final does not complete that window, so its
shorter-interval error must not be compared as a full-window tracking score.
The PID completes all four with settled velocity RMS 0.030–0.063 mm/s.

## What was trained

One inherited 391-input/78-output actor, same fixed MaleCNS graph and physical
contract, 32 hover worlds, 16 CPU physics threads, CUDA neural learning on an RTX
4090. 1,000 Hz physics and 500 Hz actions, four internal neural updates per action.
The new reward scores alive/upright flight, low mean velocity and low rotation;
no position target, old motor rewards, action-change penalty or teacher control.
One light PID hover-imitation batch per updated rollout preserves wing examples.
Encoder, cell gain/leak/bias and motor readouts receive learning; utilities stay
disabled/frozen. Graph weights and anatomy are unchanged.

The original imitation optimizer is replaced explicitly by fresh PPO Adam and
a fresh critic. The critic uses causal observation, descending-neuron memory,
reward averages and height; the last two remain training-only features. See
[the frozen recipe](recipe.json) and [the plan](PLAN.md).

## Measured work

- Training: 604.367622 s, 66 rollouts, 65 actor updates, 2,112 critic minibatches.
- 1,081,344 physical world/action transitions; 2,162,688 physics steps; 2,162.688
  aggregate simulated seconds (36.04 minutes). Approximately 1,789 transitions/s,
  including optimization. This is CPU physics with GPU neural learning, not Warp.
- Collection 363.480228 s; actor optimization 210.366183 s, including 92.726517 s
  of imitation; separate critic fitting 2.645510 s; memory refresh 27.851794 s.
  Imitation contributes 33,280 supervised target presentations. Do not add its
  timer again to actor optimization.
- Peak PyTorch CUDA allocation 3.682 GiB; GPU had other work present. Setup
  12.076329 s, replay audit 3.387631 s, checkpoint writes 0.149848 s, midpoint/final
  evaluation 23.802115 s are outside the training budget.
- Pre-update PID/parent evaluations were captured once in 24.238560/43.944631 s
  during startup verification, then reused with checked hashes. The aborted
  first replay audit collected another 16,384 transitions with zero updates;
  its collection/audit timer was not retained. These are not part of the final
  checkpoint's training time or its 1,081,344-transition count.
- Selected ancestry: original fresh velocity imitation, first 30-minute
  continuation, then this PPO pilot. The failed second imitation continuation
  is not an ancestor. Cumulative selected learning time is 2,470.517087 s.

Training began 2026-09-14 08:26:26.079497 UTC; final evaluation/report completed
08:36:57.799459 UTC. Video was decoded, inspected and opened at 08:43:53 UTC,
33m06s after the first observed implementation timestamp. Development, transfer,
rendering and verification are not neural training time.

## What the checks revealed

The full suite passes 223 tests. Actual GPU replay differs by at most 4.77e-7
in action means and 0.002289 in summed log probability; aggregate numerical KL
is 1.08e-7. The initial overly tight absolute threshold stopped before updating;
the revised check also limits aggregate KL. See [startup evidence](startup_audit.json)
and [the successful replay check](replay_audit.json). Physical reward gradients
reach the recurrent core, and parameter changes are recorded in training.json.

The more consequential issue is **update overshoot**. Every updated rollout
performs one actor minibatch, then the pre-update KL monitor stops further
minibatches. It does not undo the step already taken, and subsequent sampled KL
often far exceeds the intended 0.02. Good fitted critic values or temporarily
longer noisy training episodes did not translate into final deterministic flight.
This does not isolate every cause of failure, but it identifies a concrete
optimizer safeguard deficiency to correct before spending more time.

The user authorized continued improvement. [Run 02](../velocity_hover_ppo_02/PLAN.md)
returns to the original parent and adds post-step analytic KL checks with exact
weight/Adam rollback and a lower initial LR. Rewards and physical setup remain
unchanged. No successful hover claim is made for run 01.

Final checkpoint SHA256:
`0a90edebc341a57db786e8f495c947553abd350eeb3871c5d5b8ce9f6eb0e392`.
Source commit: `cbc84f3`. Final/midpoint checkpoints, reports and video are
preserved; physical trajectories remain in the corresponding ignored outputs.
