# PPO continuation: sustained hover with substantial oscillation

The continued actor restores the nominal five-second hover and sustains one of
three additional ten-second starts. The original parent fails all three matched
additional starts. This is limited progress in maintaining flight; it has not
produced quiet or consistently accurate hover. The user reviewed the video and
identified sustained flight as the useful improvement to preserve.

Standing passes. Walking stays upright but fails yaw tracking. One checkpoint
controls all three commands without teachers, action masks or runtime corrections.
Body, graph, actor architecture, observations, force law and reward recipe match
timing01. Both candidates and the original development parent remain available.

## Physical results

Nominal five-second hover has root RMSE **7.065 mm**, above the unchanged 5 mm
gate (original parent: 6.004 mm). Full height span is **14.708 mm**, versus
12.044 mm for the parent; vertical-speed RMS is **8.348 cm/s**, versus 8.267 cm/s.
Sustaining flight is not evidence of reduced bobbing. Timing01 falls after
crossing the height floor at 2.216 seconds.

Additional cases run ten seconds with mean actions, identical initial physical
states and observations, and no resets, teacher, critic or optimizer updates.
Every attempt is retained in [the comparison](matched_start_comparison.json).

| Starting-state seed | Parent first envelope failure | Continuation first envelope failure |
| --- | --- | --- |
| 98103 | 0.638 s | 2.084 s |
| 98113 | 1.406 s | None during 10 s |
| 98123 | 0.758 s | 3.038 s |

The sustained case has 11.701 mm full root RMSE. Its final two seconds have
9.178 mm height span and 8.200 cm/s vertical-speed RMS. Seed98103 crosses the
0.5 cm safety floor while staying upright above the ground; seed98123 eventually
flips and contacts the ground. Do not equate every envelope failure with a crash
or use post-crash samples to claim reduced airborne motion. All additional
standing/walking cases remain upright; the separate walking yaw failure persists.

## Measured training

- Setup: 8.384539 s. Training: **603.603592 s**.
- **64 worlds:** 16 stand, 16 walk, 32 hover; 16 native CPU MuJoCo/mjbatch
  physics threads and RTX 4090 neural training. This is not MuJoCo Warp.
- **819,200 transitions / 1,638.4 aggregate simulated seconds**;
  819.2 s is hover. Throughput: **1,357.182 transitions/s**, including learning.
- Collection/forward: 466.165813 s; optimization: 137.424022 s.
  Collection does not separate pure physics time from neural inference.
- 25 rollouts, **25 actor and 25 critic updates**. Both Adam states and
  exploration resume from timing01; no repeated critic-only warmup.
- Peak CUDA allocation: **12,394,289,664 bytes**, no OOM.
- 553 completed training episodes: 80 stand and 80 walk complete, 55 hover
  complete, 338 hover failures. These include noise and perturbed resets and
  are not frozen-policy evaluation success rates.
- 5 kHz physics, 500 Hz actions, 512-action rollouts, 128-action recurrent
  gradients; gamma .999, GAE lambda .995, actor LR 1e-6, critic LR 3e-4,
  initial/minimum pre-tanh noise .003, target KL .03. No imitation losses.

Both stages total **1,210.585537 s (20 min 10.586 s)** of training,
**1,392,640 transitions / 2,785.28 aggregate simulated seconds**.
Hover contributes 1,392.64 simulated seconds. Setup, evaluation, rendering,
transfer and development are separate. Observed 64-world throughput is about
43.7% higher than timing01; continued weights and different warmup prevent an
isolated matched scaling claim.

## Feedback and next learning issue

The actor receives measured/requested altitude every 2 ms, body velocity and
orientation, wing angles/speeds, joints, contacts and previous actions. Height
is an input; outputs are 78 actuator targets, including six wing targets. The
flight law reads actual wing motion; MuJoCo produces the resulting altitude.
The training critic receives those observations plus prior descending-neuron
activity and predicts return, not height or joint commands.

A [frozen sensor probe](feedback_probe.json) changes altitude and vertical-speed
readings using true preceding neural memory. Saved actions reproduce within
5.067e-7. Both perturbations change wing outputs through the graph actor. At six
sampled histories, immediate height-response magnitude is smaller than the
measured-state reference; direction differs at some phases. This is a local
neural diagnostic, not proof of the right closed-loop response or its cause.
Held inputs are counterfactual; wing-target sign alone is not lift sign. The
probe adds zero optimization and zero physical experience.

Essential altitude feedback is present, but the observation is not the full
physical state: world-vertical speed and the flight law's filtered wing activity
are not explicit features. Body velocity/orientation and recurrence provide
related information. Exact horizontal hold also lacks a position-error input.
No observation or interface changes occur in this round.

Current PPO stops both actor and critic optimization when policy KL exceeds the
limit. After one update, subsequent chunks commonly trip this stop. Timing02
gets only 25 critic updates; final hover explained variance is -0.151. The next
candidate should let the critic finish learning from a rollout when actor
updates stop, preserving the sustained actor and existing body/reward recipe.
This limitation is diagnosed; fixing it has not yet been shown to cure bobbing.
No third run or optimizer change is hidden in these results.

## Verification and review

143 implementation tests pass with 43 known dependency warnings; source lint
passes. All 338 failure windows, three nominal captures and 18 additional
parent/candidate task captures pass their applicable hash, finite-state,
action-bound, causal-feedback and matched-start checks. Parameter comparison
verifies 18 changed and 14 fixed tensors, unchanged graph/body contracts, and
physical-reward gradients reaching modeled neuron dynamics.

Nominal evaluation: 7.746852 s setup, 32.256468 s stepping/capture.
Candidate additional starts: 7.906389 s setup, 203.337647 s capture.
Parent additional starts: 7.714690 s setup, 204.830078 s capture.
Sensor probe: 6.431241 s setup, 11.716416 s diagnosis, no training/physics.

[Complete video](../../../../previews/embodied_fly/position_ppo_timing_02_all_tasks_v1.mp4):
15 s, 750 frames, 1600x900, 50 fps, 1x; 62.448826 s render/encode.
Fully decoded, 13 frames visually inspected, and opened on the Mac. The damped
observer camera does not change physical motion.
