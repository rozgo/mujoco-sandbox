# Wing feedback learning 01 — declared before execution

The nonlinear online actor still has weak restoring response and some wrong-sign damping responses on its recorded states. Train the existing measured-wing input path as well as its nonlinear wing readout, instead of another decoder-only fit.

Parent: nonlinear_online01, SHA256 `e3248b4db2adf40a26e94960674fbf2568d67fbfc7fdcd2aa0c2a4348761861c`. Keep retention02 preserved as the selected development checkpoint.

Eligible parameters: 1,792 weights in the existing 14-input sensor extension, plus 105,222 nonlinear wing-readout parameters; total 107,014. The 14 channels already contain wing angles/speeds and current/requested altitude. Input definitions, scaling and normalization stay unchanged. Gradients pass through the fixed MaleCNS dynamics from motor loss to sensory encoding. No input is routed directly to the motor decoder. No new runtime layer, policy, memory or physical mechanism is introduced.

Freeze all remaining parameters and buffers: original sensory encoder, cell gains/leaks/biases, utility/intention heads, original motor decoder and feature normalization. Because changed sensory encoding can affect every motor output, add weight4 times mean squared error against the frozen starting actor's 72 non-wing commands on ground worlds. This is a training loss, not an action mask or guarantee of identical behavior. Record actual same-history output drift; the decoder-only near-zero drift assertion does not apply to this explicitly changed encoder.

Run the current online motor imitation loop for180seconds with32worlds (11stand/11walk/10hover),16native CPU MuJoCo threads,5kHzphysics/500Hzactions,32-action chunks,2-second training episodes,seed93003. FreshAdam uses learning rate0.003 for the sensor extension and0.0003 for the nonlinear readout; global gradient-norm cap1. Keep ground task weights4:4 andhover1,extra ground-wing MSE10 andhover-wing MSE2,zero teacher mixing,zero synthetic response loss andno reset perturbations. All executed actions are the actor's own. References supply current-state labels only.

The canonical body and force-law fingerprint remain identical. Wings retain zero mass/spatial inertia/collision/aero, with independent armature. The learned actor still commands all78actuators, and wing motion alone drives the declared flight-force law.

Evaluate the saved checkpoint unassisted for5seconds per command using seed72013,unchanged strict gates,full neural/camera captures. Preserve all failures,render/decode/inspect/open the complete15second1×video. No promotion from fitting loss or short training episodes. Broader motor/utility/survival requirements remain intact.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
 --resume assets/embodied_fly/diagnostics/nonlinear_online_01.pt \
 --graph ../fly-survival/outputs/fly_survival/malecns \
 --teacher assets/embodied_fly/teachers/walking.npz \
 --output outputs/embodied_fly/wing_feedback_01 \
 --seconds 180 --worlds 32 --threads 16 --sequence 32 --episode-seconds 2 \
 --lr 0.0003 --feedback-lr 0.003 --trainable-subset wing-feedback --seed 93003 \
 --teacher-mix 0 --hover-teacher-mix 0 --retain-ground \
 --ground-retention-weight 4 --nonwing-retention-weight 4 \
 --ground-wing-loss 10 --ground-posture --hover-reference state
```
