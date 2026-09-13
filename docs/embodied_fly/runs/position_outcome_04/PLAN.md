# Second bounded PPO pilot: critic warmup

The reduced-exploration outcome03 preserves standing and walking, but still
fails unassisted hover (21.758 mm root RMSE). It does not replace the parent.
This is the second and final new three-minute PPO pilot in the authorized round.

Start again from position_sustain_retention_01, with fresh PPO optimizers and
critic. Keep outcome03's entire recipe, including .003 initial/minimum noise,
actor LR 3e-7, seed98003, 32 worlds (11 stand/11 walk/10 hover), 16 CPU physics
threads, 128-step rollouts, 32-step sequences, two epochs, five-second episodes,
gamma .995, lambda .95, zero entropy bonus, ground retention weight4, KL .03.

The only new training choice is eight critic-only rollout updates before actor
learning begins. All physical collection remains the noisy student. Warmup
uses no extra worlds, is included in the 180-second wall budget, and has no
actor/exploration optimizer steps. The final checkpoint remains one shared
actor with the same physical fly, graph, sensors, rewards and 78 outputs.

Evaluate all three commands without noise using seed97013 and five-second
episodes. Render/open all cases using the newly requested damped observer
camera; captures and physics remain unchanged by camera styling. Require
preserved ground behavior plus improved hover before any further promotion
checks. If it fails, preserve the checkpoint and stop this PPO round.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.ppo \
  --motor-all --preset wing_position --motor-retention-weight 4 \
  --resume assets/embodied_fly/diagnostics/position_sustain_retention_01.pt \
  --graph outputs/fly_survival/malecns --output outputs/embodied_fly/position_outcome_04 \
  --seconds 180 --worlds 32 --threads 16 --episode-seconds 5 \
  --horizon 128 --sequence 32 --epochs 2 --lr .0000003 \
  --noise .003 --minimum-noise .003 --critic-warmup-rollouts 8 \
  --target-kl .03 --entropy 0 --gamma .995 --gae-lambda .95 --seed 98003
```
