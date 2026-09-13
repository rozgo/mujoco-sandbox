# Nonlinear online readout 01 — declared before execution

The offline nonlinear fit reduced recorded-history error but failed unassisted walking and hover. Continue that diagnostic checkpoint on the physical states produced by its own actions. Preserve retention02 as the selected parent; this is an unaccepted candidate lineage.

Resume nonlinear_motor_readout01 (SHA256 `9a39a8011eeac00a6bc5be600176a83558fea1425a298735733af6c121bfb049`). Update only its 105,222 nonlinear wing-readout parameters. Freeze its feature normalization and every original actor weight/buffer, including sensory encoding, MaleCNS dynamics and all original motor output rows. No physical or deployment architecture change. Save the extension metadata so the ordinary loader deploys the same single actor across commands.

Use the established motor_focus online imitation loop for180seconds:32worlds (11stand/11walk/10hover),16native CPU MuJoCo threads,5kHz physics,500Hz actions,32-action chunks,2-second maximum training episodes,seed93002,freshAdam0.0003,gradient cap1. No clock-driven reference, synthetic response loss or reset perturbations. Ground task weights4:4,hover1;extra wing MSE10on ground and2in air. This stage collects new physical experience; it is supervised online imitation, not PPO.

The actor executes every action, with zero teacher mixing for every task. A frozen copy of the starting actor supplies the ground body-action targets on identical histories; current-state initial-pose wing corrections replace only its six ground-wing targets. The established measured-state hover reference supplies airborne labels. References supply training targets and never execute a command. Failed training episodes reset explicitly and retain the final64frames as evidence. This changes the preceding offline ground targets from matching previously recorded outputs to restoring the actual wing pose, as requested by the user.

Full unassisted evaluation:5seconds per stand/walk/hover,seed72011,unchanged physical fingerprint and strict gates,all frames retained. Render, decode, visually inspect and open the complete15second1×film. Success in fitting or teacher-labeled training is insufficient for promotion.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
 --resume assets/embodied_fly/diagnostics/nonlinear_motor_readout_01.pt \
 --graph ../fly-survival/outputs/fly_survival/malecns \
 --teacher assets/embodied_fly/teachers/walking.npz \
 --output outputs/embodied_fly/nonlinear_online_01 \
 --seconds 180 --worlds 32 --threads 16 --sequence 32 --episode-seconds 2 \
 --lr 0.0003 --trainable-subset wing-residual --seed 93002 \
 --teacher-mix 0 --hover-teacher-mix 0 --retain-ground \
 --ground-retention-weight 4 --ground-wing-loss 10 --ground-posture --hover-reference state
```
