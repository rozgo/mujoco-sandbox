# Critic and reward review — September 14, 2026 UTC

Follow-up user steering prioritizes reward design before critic changes.
See the [implemented goal-directed reward correction](runs/flight_reward_01/SUMMARY.md).
The active-reward table below describes the preceding round_trip_ppo_01 recipe;
the explicit new variant changes its preferred velocity, not its weights.

The current critic is a reasonable PPO baseline; it has not been shown to be
best for the embodied fly. This is research and code review, not another trained
variant. Preserve the parent checkpoint and completed round_trip_ppo_01 results.

## Actual critic

`ppo.Critic` consumes 399 actor observations plus 1,314 preceding descending-cell
states: 1,713 -> 128 tanh -> 128 tanh -> 1, 236,033 parameters. It has no private
recurrent state; the descending cells supply a partial summary of actor history.
Its inputs and neural features are detached from the actor for value fitting.
Targets are timeout-aware GAE returns, fitted with MSE and separate Adam.

The latest trial uses original input scaling, 16 critic epochs with shuffled
individual time/world samples, LR1e-4, and two critic-only warmup rollouts.
Current fitting uses no PopArt or categorical/distributional value head.
Individual sample shuffling is compatible with this feedforward critic; adding
a recurrent critic would require correctly ordered sequences, resets and memory.

The last rollout's post-fit explained variance is 0.96293. It is fit on the
training rollout and its bootstrapped targets, not held-out prediction of actual
future returns. It does not establish correct action ranking or useful actor
updates. The frozen child still fails 7/7 tracking gates, with mean route RMS
2.38% worse than the parent. Learning useful correction remains unproven.

## Relevant primary research

1. [What Matters In On-Policy Reinforcement Learning?](https://arxiv.org/abs/2006.05990)
   supports separate policy/value networks and studies critic width, MSE,
   normalization and minibatch choices on continuous-control benchmarks. It is
   evidence for a conventional baseline, not a universal optimal configuration.

2. [Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2105.11674)
   motivates conditioning on history as well as simulator state. A state-only
   value function can be inadequate for a history-dependent policy. A learned
   compressed history is still an approximation, not an automatic guarantee.

3. [Informed Asymmetric Actor-Critic, ICML2026](https://orbi.uliege.be/bitstream/2268/345901/1/informed-critic-icml.pdf)
   studies selected privileged training signals, including tests of their value
   for prediction. Our proposed application is a training-only critic with
   useful physical state and task-scheduler state, retaining history. Actor
   inputs and the deployed MaleCNS graph need not change. This is an inference
   for our task; the paper does not evaluate this fly.

4. [Stop Regressing, 2024](https://arxiv.org/abs/2403.03950)
   replaces scalar squared-error value fitting with categorical cross-entropy
   using Gaussian-smoothed value-bin targets (HL-Gauss). It addresses value
   optimization under noisy/changing targets. This classification surrogate
   is distinct from explicitly learning the full return distribution for risk.
   [Learning the Supports for Categorical Critic, July2026 preprint](https://arxiv.org/abs/2607.01880)
   learns the bin range and evaluates continuous-control tasks. Both motivate
   an experiment; neither demonstrates success for our recurrent fly PPO.

5. [Learning Risk-Aware Quadrupedal Locomotion using Distributional RL](https://arxiv.org/abs/2309.14246)
   integrates return distributions and a risk metric with PPO and demonstrates
   ANYmal behavior. This is directly relevant to later hazards/survival, but
   risk preference changes the objective. It is not the first intervention for
   establishing basic target correction.

6. [PopArt: Learning values across many orders of magnitude](https://arxiv.org/abs/1602.07714)
   normalizes value targets while preserving unnormalized predictions. It is
   different from our earlier StandardizedValueNetwork input normalization.
   It may be useful when later motor/utility tasks have different return scales;
   current bounded hover rewards make it a lower-priority hypothesis.

Additional recent robotics evidence: [RAFT, August2026 preprint](https://arxiv.org/abs/2608.22976)
uses privileged critic information for thruster faults with a recurrent actor.
Its setting is a planar floating platform, not articulated wing-driven flight.

## Exact active reward

`HoverBalancedReward` in `hover_only.py`, version `hover_bounded_scores_v2`,
is used for both stationary hover and moving targets. Every world is scored
independently after each 2ms action interval (two 1ms physics ticks).
Define S(e,s) = 1/sqrt(1+(e/s)^2). The following are reward rates per simulated
second; add them and multiply by 0.002 for a normal transition.

| Term | Rate | Measurement / scale |
|---|---:|---|
| Remain alive | +0.5 | Valid airborne state |
| Altitude tracking | +2 S(error,1mm) | Current altitude minus current requested altitude |
| Horizontal position | +1 S(error,1mm) | XY distance to current requested target |
| Vertical velocity | +1 S(speed,20mm/s) | Target velocity is zero |
| Horizontal velocity | +1 S(speed,5mm/s) | Target velocity is zero |
| Upright | +0.5 clip(cos(tilt),0,1) | Body upright axis vs world up |
| Angular velocity | +0.1 S(speed,1rad/s) | Body angular speed magnitude |
| Actuator effort | -0.002 mean(effort^2) | Each force / force limit, clipped to[-1,1] |

Ideal rate is 6.1 minus actuator effort, at most 0.0122 per action interval.
A failed step instead receives exactly -1, all ordinary terms are zeroed and
the episode terminates. Failure: root height <5mm, upright cosine <0.5, or
forbidden contact load >10% of body weight. Failure penalty is once, not per
physics substep. Episode return is the sum; PPO uses discounted GAE estimates.
The critic predicts returns and never assigns these physical rewards.

There is no waypoint-completion bonus, return-home bonus, PID-action matching,
wing-frequency reward, action-difference penalty, ground-posture retention or
utility reward in this pilot. Waypoint/return/100ms-drift gates are measurement
criteria, not reward terms. The return-home request uses ordinary tracking scores.

Two concerns for the next experiment: zero-speed scores can compete with travel
while targets move; distant position errors yield weak differences in bounded
position scores. These are plausible mechanisms, not established causes of the
failed learning. A better critic will still optimize the reward it is given.

## Proposed decision, no implementation yet

Keep the same body and MaleCNS actor. First audit target-to-wing sensitivity and
critic prediction on held-out complete trajectories, including outcome variation
at the same elapsed time. Compare the existing critic with a history-aware critic
receiving explicitly declared simulator/task state: body pose/velocity, measured
wing dynamics/actuator state and current trajectory phase/target velocity.
Some physical information already exists in actor observations; test incremental
information rather than assume every added feature is new or useful. No future
physical outcome is an input, and privileged scheduler information stays outside
the actor. Do not claim theory guarantees for an arbitrary compressed history.

Keep reward and actor update settings fixed for the critic comparison. Evaluate
both value generalization and actual frozen-actor tracking at equal experience
and measured compute. Categorical HL-Gauss is a separate second candidate. Audit
movement-specific rewards in a separate comparison, not simultaneously with a
critic swap. PID correction teaching remains an option if actor feedback is weak,
not a claim that additional imitation is necessarily required.
