# Physical rewards for standing, walking and hover

The opt-in `--motor-all` PPO stage continues the same `wing_position` graph actor.
It learns physical outcomes and retains ground behavior using a frozen parent
for supervision. The deployed checkpoint still has one actor and no teacher.

Every world chooses its reward by its current command; ground and flight rewards
are never added together. Ground rewards measure tracking, valid support and
standing/resting-wing posture. Hover rewards measure altitude, horizontal and
vertical velocity, orientation and continued airborne support. Reward rates are
multiplied by 0.002 s. A physical failure adds one -1 penalty and resets that world.
Time-limit truncations bootstrap from their actual terminal physical state.

The training-only critic predicts returns from the 397 observations plus prior
descending-neuron state, through 128 and 128 tanh units and one value output.
The critic cannot control joints. The frozen parent's separate neural state
supplies mean ground-action targets; weight-4 MSE discourages forgetting. All
executed commands come from the student's tanh-normal exploration distribution.
Physical-return and retention gradients are audited separately through MaleCNS.

The actor's measured edges and utility/intentions stay fixed. Its motor encoder,
decoder, nonlinear wing readout and modeled neuron dynamics can learn. No changed
body, aerodynamic forces, teacher oscillator or observation-to-motor bypass is
introduced. The six wing outputs still specify MuJoCo position-actuator targets;
the unchanged flight law reads actual wing motion.

Use fresh output directories and the checksummed graph cache on the GPU host:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-all --preset wing_position --motor-retention-weight 4 \
  --resume assets/embodied_fly/diagnostics/position_sustain_retention_01.pt \
  --graph outputs/fly_survival/malecns --output outputs/embodied_fly/position_outcome_01 \
  --seconds 180 --worlds 32 --threads 16 --episode-seconds 5 \
  --horizon 128 --sequence 32 --epochs 2 --lr .000003 --noise .01 \
  --target-kl .03 --entropy 0 --gamma .995 --gae-lambda .95 --seed 98003
```

Follow the run with `motor_focus evaluate --preset wing_position --seconds 5`,
using the saved actor and the same graph. Evaluation uses mean actions and loads
no critic or teacher. Preserve existing gates and every failed capture; training
return is not proof of a usable skill. See the [predeclared first pilot](runs/position_outcome_01/PLAN.md).

## Bounded exploration and critic diagnosis

[Paired frozen-actor tests](runs/position_exploration_01/SUMMARY.md) show two
separate limitations: the parent fails hover from three additional starting
states even without noise, and .01 exploration disrupts its recorded successful
start while .003 preserves that start in one paired trial.

[Outcome03](runs/position_outcome_03/SUMMARY.md) therefore uses initial and minimum
noise .003. It preserves the ground motors but fails hover. [Outcome04](runs/position_outcome_04/SUMMARY.md)
adds eight critic-only rollouts, then ordinary actor learning, all within the
same three-minute budget. It is the first position PPO candidate to retain
five-second airborne control in the matched review. Root error falls from
6.004 to 5.499 mm, but the accuracy gate and walking yaw gate still fail.
Vertical oscillation remains substantial and essentially unchanged. Additional
starting-state checks are required before treating this as a robustness gain.

`--minimum-noise` makes the exploration floor explicit; its default .01 keeps
older recipes unchanged. `--critic-warmup-rollouts` defaults to zero and is
restricted to the all-motor curriculum. During warmup the actor and exploration
receive no optimizer updates; the critic learns from actual sampled physical
experience. Warmup time and transitions are included in training totals, with
actor and critic update counts reported separately. No extra deployed module
is introduced.

Reviews now use the [damped observer camera](CAMERAS.md), with the previous
locked view available explicitly. Camera improvements must not be described
as calmer physical flight.

The completed [additional-start comparison](runs/position_outcome_04_exploration/SUMMARY.md) still fails all three deterministic hover starts (and all six noisy hover cases). Outcome04 is not promoted over the preserved parent. The two-pilot round is complete; further work should address hover consistency and vertical oscillation explicitly.

## Physical-time PPO follow-up

The user approved a fixed ten-minute [timing01 recipe](runs/position_ppo_timing_01/PLAN.md)
after reviewing the prior failures, then explicitly authorized a further
[continuation with more worlds](runs/position_ppo_timing_02/PLAN.md). The actor,
physical body and measured graph remain unchanged. No MuJoCo Warp port occurs
in these runs: native CPU physics and RTX 4090 neural training remain explicit.

The opt-in `--hover-physical` stage uses half hover worlds and equal quarters
standing/walking, no imitation losses, and a declared reset-widening curriculum.
`--checkpoint-activations` recomputes identical actor activations during backward
passes. This enables 128-action recurrent sequences (256 ms), 512-action rollouts
(1.024 s/world), discount .999 and GAE lambda .995 without changing the physical
or policy clock. Output/gradient equivalence and true episode resets are tested.
The hover reward uses physical outcome costs that retain differences away from
the target, rather than the earlier exponential bands.

[Timing01 results](runs/position_ppo_timing_01/SUMMARY.md) preserve ground stability
but lose the nominal hover. A specific remaining limitation is observed in the
trainer: the joint policy KL stop also terminates critic optimization. Most
rollouts consequently receive one actor and one critic update after warmup,
while hover value predictions remain weak. Longer credit and more experience
have not by themselves established reliable hover. A future independent critic
schedule would be an explicit new recipe, not a change hidden inside these runs.

[Timing02](runs/position_ppo_timing_02/SUMMARY.md) continues for another measured
603.604 s with 64 worlds, preserving actor/critic optimizers. It restores the
nominal five-second hover and sustains one of three additional ten-second starts;
the original parent fails all three matched additional starts. Bobbing remains
large, including a 9.178 mm final-two-second height span in the sustained case.
Both stages total 20 min 10.586 s of training and 1,392,640 transitions. This is
progress in maintaining flight, with accuracy and robustness still unfinished.
The latest sensor probe confirms that height and vertical-speed inputs change
wing outputs. Receiving feedback is not the same as learning a stable response;
the next bounded change should address the coupled actor/critic stop.

## Independent value fitting

The [critic continuation](runs/position_ppo_critic_01/PLAN.md) explicitly enables
`--independent-critic`. The policy retains its KL early stop; the critic trains
for every requested epoch over saved causal features and fixed GAE returns.
At 512 actions and 128-action chunks, two epochs give eight critic updates per
rollout. The unchanged critic is small enough to fit these features directly,
without replaying the 166,700-neuron actor. No additional physical transitions
or deployed controller are introduced. Historical commands without the flag
retain the previous coupled schedule.

The input features are the normalized observation and preceding descending-cell
state actually used during collection. They and the return targets are detached;
value fitting cannot update actor weights. Per-task before/after fit, sample
presentations, policy stops and critic update time are recorded. In-sample value
fit remains separate from frozen-policy physical performance.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-all --hover-physical --checkpoint-activations --independent-critic \
  --preset wing_position \
  --resume assets/embodied_fly/diagnostics/position_ppo_timing_02.pt \
  --graph outputs/fly_survival/malecns \
  --output outputs/embodied_fly/position_ppo_critic_01 \
  --seconds 600 --worlds 64 --threads 16 --episode-seconds 5 \
  --horizon 512 --sequence 128 --epochs 2 --lr .000001 \
  --noise .003 --minimum-noise .003 --critic-warmup-rollouts 0 \
  --target-kl .03 --entropy 0 --gamma .999 --gae-lambda .995 --seed 99703
```

Use the recorded graph-cache location and matching compiled-body platform for
the actual run. The command above documents the parameters; Linux headless
evaluation uses EGL, while macOS replay uses its native rendering backend.
