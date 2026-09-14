# Wider horizontal reward does not yet produce controlled hover

Run 03 changes only the horizontal reward width from 0.5 to 2 cm/s, retaining
zero velocity as the optimum. It starts from the original imitation parent,
with the bounded optimizer introduced in run 02. Training took **603.909827 s**.

| Start | Original airborne | Final airborne |
|---|---:|---:|
| 0 | 10.000 s | 10.000 s |
| 1 | 6.772 s | 10.000 s |
| 8, held out | 10.000 s | 7.232 s |
| 9, held out | 4.190 s | 4.732 s |

Ten seconds is the evaluation ceiling. Common first-two-second velocity RMS
changes only from approximately **25.96 to 25.88 mm/s**, far below the declared
20% improvement milestone. The two complete final cases climb approximately
76 mm and reach 171–172 mm peak displacement. Neither the survival tradeoff
nor the tiny early velocity change establishes improved hover. Midpoint and
final failures are preserved; this checkpoint is not promoted.

[PID / original / final comparison](../../../../previews/embodied_fly/velocity_hover_ppo_03_comparison_v1.mp4):
all four starts, 43 seconds, 1920x1080, 50 fps, 1x, damped cameras and wing details.
Decoded all 2,150 frames, visually inspected, opened **09:17:39 UTC**.

## Measured work

- 32 worlds, 16 CPU MuJoCo/mjbatch physics threads; CUDA neural learning on RTX 4090.
  1 kHz physics, 500 Hz actor, four internal neural steps per action.
- **720,896 transitions**, **1,441,792 physics steps**, **1,441.792 aggregate
  simulated seconds**, approximately **1,194 transitions/s** including learning.
- 44 rollouts, 143 actor updates, 1,408 critic minibatches, 22,016 imitation targets.
- Collection **243.580163 s**; actor optimization **339.910927 s**, including
  **61.120525 s imitation**; critic **1.813135 s**; memory refresh **18.587426 s**.
- Separate setup **12.611745 s**, audit **3.371672 s**, checkpoint writes
  **0.201928 s**, midpoint/final evaluation **90.428619 s**. Baseline captures reused.
- Peak PyTorch CUDA allocation **4,101,086,720 bytes**, 3.819 GiB.
- Training began **09:00:16.736797 UTC**; final report completed
  **09:11:54.695054 UTC**. Rendering **70.095797 s**, decode **4.366964 s**.
- Source `8ca2023`. Final SHA256
  `55a7f6adc7bc427a16326ab6573f8f33becc73c4266216a26210ce901ccf3f0e`.

The user requested continued improvement. [Run 04](../velocity_hover_ppo_04/PLAN.md)
focuses PPO on the existing wing readout while holding the upstream representation
steady. This stage keeps the same deployed architecture and physics; it explicitly
changes the trainable parameter scope and uses exact recorded neural features to
make optimization cheaper. Success remains physical flight, not a lower loss.
