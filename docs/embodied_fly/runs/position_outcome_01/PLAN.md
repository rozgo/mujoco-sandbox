# Physical outcomes after successful walking retention

Goal continuation began 2026-09-13 14:56:13 UTC. The prior goal turn made progress:
reference recovery evidence, two trained checkpoints, complete evaluations and
opened videos. Both hosts were synchronized at 82eb6f7 and no training job remained.

Parent position_sustain_retention01 stands, walks 4.84 cm and remains airborne.
Hover's 6.00 mm root RMSE narrowly fails the 5 mm gate; error includes an early
height dip and lateral drift. Walking yaw also fails. These are the next physical
outcomes to improve; copying reference commands is not equivalent to optimizing
those outcomes. Introduce --motor-all in the existing recurrent PPO trainer.

One canonical wing_position body, one 397-input / 78-output actor, four graph
updates, fixed measured adjacency and utility/intentions. Preserve the six-output
wing residual's metadata on PPO checkpoint save/load. No runtime teacher, oscillator,
action mask, body controller or new actor inputs. A separate value network and
frozen parent reference are training-only; neither is present at evaluation.

Each world gets one selected reward per 2 ms action. Ground tasks keep the existing
tracking/support/full-pose reward. Hover uses the existing physical flight reward,
with horizontal speed width 0.5 cm/s instead of the legacy 5 cm/s, and failure
height 0.5 cm matching MotorTasks. The 0.3 cm altitude width, vertical-speed and
orientation rewards remain. Multiply rate terms by the actual action interval;
apply one -1 failure penalty, then reset. Store reward terms in failure traces.
Timeouts bootstrap the critic from the final state before reset.

Add weight-4 MSE against the frozen parent's 78 mean ground outputs on collected
student observations. This is explicitly supervised retention alongside PPO.
Hover has no reference-action targets. Audit physical-return gradients separately
from retention gradients. All 78 sampled actions execute; no teacher blending.

Predeclared pilot: 180 seconds, 32 worlds (11 stand / 11 walk / 10 hover), 16 native
CPU MuJoCo/mjbatch threads, RTX 4090 neural computation. 5 kHz physics / 500 Hz
actions. Five-second episodes, horizon128, recurrent sequence32, two epochs,
fresh actor Adam3e-6, critic Adam3e-4, tanh-normal initial sigma0.01, targetKL0.03,
entropy0, gamma0.995, GAE lambda0.95. Seed98003. The existing distribution clamps
sigma to [0.01,0.15]; the actual training clocks include every inference/learning
operation. This is an exploratory continuation, not a speed comparison.

Complete five-second mean-policy evaluation for stand/walk/hover with seed97013
(same starts as the previous candidate), unchanged physical/posture gates, no
resets/teacher/critic/exploration. Save all failures and decode/inspect/open the
complete 15-second 1x video. Any regression keeps the parent preferred. No claim
of utility learning, takeoff/landing, transitions or completed multi-agent survival.
