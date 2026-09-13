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
