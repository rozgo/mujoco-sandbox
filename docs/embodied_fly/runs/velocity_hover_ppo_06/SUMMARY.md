# More unchanged training plateaued and worsened sideways control

All four predetermined starts complete ten seconds, but this continuation is
**not promoted over run 05**. Final net climb decreases only about 106 -> 101 mm,
while horizontal velocity RMS worsens 9.30 -> 15.43 mm/s and total velocity RMS
14.92 -> 18.66 mm/s. Vertical RMS improves 11.67 -> 10.50 mm/s. Metrics use the
declared 100 ms velocity average and common 0.2–10 s evaluation window.

The midpoint also trades axes: approximately 80 mm climb and 8.56 mm/s vertical
RMS, but 15.04 mm/s horizontal RMS. Neither snapshot meets the predefined joint
objective of climb toward 50 mm and horizontal RMS no greater than 10 mm/s.
Survival alone does not establish stationary hover.

This run restored run 05's actor Adam, critic/Adam and actor sampling RNG with
the **identical recorded recipe**, including rewards, trainable wing readout,
32 worlds, 16 CPU MuJoCo/mjbatch physics threads and CUDA neural computation.
No physical, observation, action or clock changes. All upstream actor parameters
remain unchanged; the full MaleCNS graph runs in every live actor step.
The recurrent replay audit includes eight resets and passes, as does cached
readout replay. Both checkpoints and all physical captures are preserved.

## Measured work

- Training **603.990932 s**, 108 rollouts, **1,769,472 transitions**,
  **3,538,944 physics steps**, 407 actor updates and 3,456 critic minibatches.
- Collection **597.360504 s**, actor optimization **2.459984 s** (including
  **.141817 s** imitation), critic **4.143642 s**.
- Separate setup **15.127430 s**, replay audit **3.388092 s**, checkpoint IO
  **.069445 s**, physical evaluation **90.020289 s**.
- Start **2026-09-14 14:53:55.921374 UTC**, report **15:05:33.423539 UTC**.
  Goal started **14:52:05 UTC**. GPU peak allocation **1,548,877,824 bytes**.
- Source `15e73e0`; checkpoint SHA256
  `cbfa256641268d0de5918d21f9b40f74e5683d609c64b995eb561ea28fc293e9`.
- Video render **69.082268 s**, full decode **4.295003 s**, opened locally
  **15:15:04 UTC** after visual inspection. 43 seconds, 2,150 frames, 50 fps,
  1920x1080, 1x. PID / run 05 / run 06 use matched starts and common chart scales.

[Watch the preserved comparison](../../../../previews/embodied_fly/velocity_hover_ppo_06_comparison_v1.mp4).

```sh
open previews/embodied_fly/velocity_hover_ppo_06_comparison_v1.mp4
```

The agreed next trial returns to run 05 and increases only the vertical tracking
reward rate from 2 to 3, with one critic-only adaptation rollout. Run 06 remains
in total experiment cost, but is excluded from that next checkpoint's ancestry.
