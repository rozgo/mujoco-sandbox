# Readout refinement with the learned sensor encoder frozen

Parent: wing_feedback01, SHA256 `6c687bdcaac301ae4490c0f0edf9e909a1f7f3e59e1ac6dc6206970a23549989`. It learned stronger sensor weights, but its full review failed standing/hover and walking posture. It is an unaccepted diagnostic lineage; retention02 remains preserved.

Test a staged continuation: freeze the newly learned sensor extension along with all other encoder/core/base-decoder weights and normalization buffers. Update only the existing 105,222 nonlinear wing-readout parameters on actual actor-driven physics. This keeps the recurrent feature computation fixed throughout training. It tests whether adapting the readout to that fixed representation helps; it does not assert that changing representations was the unique cause of failure.

Reuse the existing online motor loop without source changes: 180 seconds, 32 worlds (11 stand/11 walk/10 hover), 16 CPU MuJoCo threads, 5 kHz physics/500 Hz actions, 32-action chunks, two-second training episodes, seed93004. Fresh Adam0.0003 and gradient cap1. Ground task weights4:4,hover1,extra wing MSE10/2,zero teacher mixing,zero synthetic-response loss andno reset perturbations. Keep ground non-wing distillation weight4; with this fixed encoder/base decoder, its same-history output differences should be roundoff only. References supply labels and never execute actions.

No body, flight law, runtime architecture or input-definition change. The same checkpoint commands all78actuators for all tasks. Independently verify that every parent state entry outside the nonlinear readout is bitwise unchanged, including the sensor weights learned in the preceding stage.

Evaluate for5seconds per command,seed72015,unchanged strict gates. Save all failures and the full15second1×video,decode/inspect/open it. No promotion based on two-second training timeouts. Broader motor/utility/survival requirements remain unchanged.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
 --resume assets/embodied_fly/diagnostics/wing_feedback_01.pt \
 --graph ../fly-survival/outputs/fly_survival/malecns \
 --teacher assets/embodied_fly/teachers/walking.npz \
 --output outputs/embodied_fly/feedback_readout_01 \
 --seconds 180 --worlds 32 --threads 16 --sequence 32 --episode-seconds 2 \
 --lr 0.0003 --trainable-subset wing-residual --seed 93004 \
 --teacher-mix 0 --hover-teacher-mix 0 --retain-ground \
 --ground-retention-weight 4 --nonwing-retention-weight 4 \
 --ground-wing-loss 10 --ground-posture --hover-reference state
```
