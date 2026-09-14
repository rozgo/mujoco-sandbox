# Continued PPO reduces drift while preserving four completed flights

All four fixed starts complete **ten seconds airborne**. Relative to run 04,
velocity RMS decreases from approximately 24.4 to **14.92 mm/s**, horizontal RMS
18.83 -> **9.30 mm/s**, and vertical RMS 15.51 -> **11.67 mm/s**. Net climb falls
from approximately 152 to **106 mm**, and peak displacement 217 -> **117 mm**.
The actor is still ascending; this is useful flight-control progress, not solved
hover. The PID reference remains substantially steadier.

Against the original imitation parent on the two comparable complete flights
(starts 0 and 8), velocity RMS is approximately **53% lower**, horizontal RMS
**70% lower**, and peak displacement **21% lower**. Vertical RMS and net climb
remain worse than that original parent's approximately 8.23 mm/s and 45 mm.
RMS uses the common 0.2–10 s window with the declared 100 ms velocity average.
The common 0–2 s velocity RMS across all four starts improves 25.96 -> **15.59
mm/s**, approximately **40%**. Do not compare failed shortened intervals as if
they were full flights.

[Full progress: PID / original imitation / final PPO](../../../../previews/embodied_fly/velocity_hover_ppo_progress_v1.mp4).
[This continuation only: PID / run 04 / run 05](../../../../previews/embodied_fly/velocity_hover_ppo_05_comparison_v1.mp4).
Both show all four starts, 43 seconds, 50 fps, 1920x1080, 1x, wing detail and shared
chart scales. Full decode and visual checks pass. Opened on the local Mac at
**09:52:15 UTC** and **09:51:28 UTC**, respectively.

## Same policy and training recipe

Continue run 04's final weights, wing-readout Adam state, critic/Adam and actor
sampling RNG. Same 105,222 trainable wing-readout parameters, fixed encoder/core/
base decoder, fixed .003 exploration, reward and light PID-label anchor. Physical
episodes restart at the same declared starts; critic minibatch shuffle restarts
from its recorded seed. No new critic calibration or warmup. The full MaleCNS
still runs on real observations and produces all 78 outputs during every flight
step. The teacher never controls the PPO evaluation or rollout.

All 453 accepted updates stay below analytic minibatch KL .02 (maximum
**.01999249**); 141 backtracking retries. Frozen upstream parameters remain
identical. GPU continuation smoke and actual recurrent/cached replay audits pass.
A 1712 -> 128 -> 128 -> 1 critic uses observation, preceding descending state,
causal reward-history statistics and height. It is training-only and never drives
the body. Height is absent from actor inputs; commands remain four zero velocities.

## Measured work

- **604.488490 s training**, **32 worlds**, **16 CPU physics threads**. RTX 4090
  neural inference/learning; CPU MuJoCo/mjbatch physics, not Warp. 1 kHz physics,
  500 Hz actor, four internal neural steps per action.
- **1,769,472 transitions**, **3,538,944 physics steps**, **3,538.944 aggregate
  simulated seconds**, approximately **2,927 transitions/s** including learning.
- 108 rollouts, 453 actor updates, 3,456 critic minibatches, 55,296 imitation targets.
- Collection **597.629097 s**; actor optimization **2.649631 s**, including
  **.145154 s imitation**; critic **4.181981 s**. Upstream memory needs no refresh.
- Separate setup **15.339663 s**, audit **3.379324 s**, writes **.071080 s**,
  physical midpoint/final evaluation **90.723251 s**. Parent/PID captures reused
  after physical-contract and checkpoint-correspondence checks.
- Peak PyTorch CUDA allocation **1,548,877,824 bytes**, approximately 1.44 GiB.
- Training began **09:34:02.834884 UTC**, final report **09:45:41.528016 UTC**.
- Selected ancestry **3,072.803470 s**: original imitation **1,866.149465 s** plus
  two productive PPO stages **1,206.654006 s**. This excludes failed PPO pilots
  and discarded smoke updates; total trial cost is in the series report.
- Continuation video render **69.774867 s**, decode **4.347518 s**. Full-progress
  video render **68.475034 s**, decode **4.303401 s**. These are not training times.
- Training source `3baa5b6`. Final checkpoint SHA256
  `3106cb363aa75dab78322181ad9f46c229a8a29ca8f1f0599bc18e8d59763a32`.

Full fly suite for the implementation: **229 passed in 165.88 s**. Subsequent
continuation changes pass nine focused tests and the GPU optimizer/critic restore
smoke. Rendering verifies identical initial qpos/qvel across comparison panels.
All prior runs, weights, failures and midpoint captures remain available. This
completes the initial sustained-flight/reduced-error pilot milestone; improved
altitude regulation is the remaining motor-learning objective.

From the repository root on macOS:

```sh
open previews/embodied_fly/velocity_hover_ppo_progress_v1.mp4
```
