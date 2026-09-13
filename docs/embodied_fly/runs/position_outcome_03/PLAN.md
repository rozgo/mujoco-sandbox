# PPO with reduced exploration

First of at most two new three-minute pilots authorized September 13, 2026.
Preserve the position_sustain_retention_01 parent. New matched tests show all
three other hover starts fail even without noise; standing and walking remain
stable. In a second diagnostic using the successful video's exact initial
physical state, hover survives five seconds at noise 0 and .003, but fails at
2.768 seconds with .01 noise. All actor parameters remain frozen in those tests.

Change one exploration setting from outcome02: lower initial and minimum
pre-tanh standard deviation from .01 to .003. Keep the smaller actor learning
rate (3e-7), seed 98003, 32 worlds (11 stand, 11 walk, 10 hover), 16 CPU physics
threads, 128-step rollouts, 32-step sequences, two epochs, five-second episodes,
gamma .995, lambda .95, zero entropy bonus, ground retention weight 4, target KL
.03. Neural work runs on RTX 4090. All 78 actions still receive exploration.
No changes to the body, flight forces, actor or physical rewards.

Critic warmup was considered before the recorded-start diagnostic returned;
its implementation is available but OFF in this pilot. Test it in the second
pilot if the first result warrants it. Do not describe this run as warmup.

Evaluate all three commands with seed 97013, five seconds each, without noise
or teacher. Preserve and open the full video regardless of result. Promotion
requires preserved ground behavior and improved hover followed by checks on
the predetermined diagnostic starts; one attractive rollout is insufficient.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-all --preset wing_position --motor-retention-weight 4 \
  --resume assets/embodied_fly/diagnostics/position_sustain_retention_01.pt \
  --graph outputs/fly_survival/malecns --output outputs/embodied_fly/position_outcome_03 \
  --seconds 180 --worlds 32 --threads 16 --episode-seconds 5 \
  --horizon 128 --sequence 32 --epochs 2 --lr .0000003 \
  --noise .003 --minimum-noise .003 --critic-warmup-rollouts 0 \
  --target-kl .03 --entropy 0 --gamma .995 --gae-lambda .95 --seed 98003
```
