# Bounded PPO updates preserve flight, but do not solve hover

The second pilot completed **603.390 s**. Post-step acceptance fixes the first
pilot's update overshoot: all 146 accepted updates have analytic minibatch KL
at or below **0.0199765**, with 13 rejected proposals retried at smaller step
sizes. No rejected proposal advances Adam's history. This improves optimization
control, not yet physical hover.

| Start | Original parent airborne | Final airborne |
|---|---:|---:|
| 0 | 10.000 s | 10.000 s |
| 1 | 6.772 s | 7.812 s |
| 8, held out | 10.000 s | 10.000 s |
| 9, held out | 4.190 s | 4.924 s |

Ten seconds is the evaluation ceiling. Survival improves modestly, but the
matched first-two-second velocity RMS increases from about **25.96 to 26.54
mm/s**. The two complete cases climb about 80 mm and reach peak displacement
about 183 mm, worse than the parent's approximately 45 mm climb/149 mm peak
displacement. Do not promote this as successful hover or compare conditional
error from different-length failed episodes as a tracking improvement.

[Open the 43-second PID/parent/PPO comparison](../../../../previews/embodied_fly/velocity_hover_ppo_02_comparison_v1.mp4).
All four starts, common chart scales, 1x playback, damped overview cameras and
close wing views. Decoded and visually inspected; opened **09:02:33 UTC**.

The body, fixed graph, 391 observations, 78 actions, four zero velocity commands,
reward, imitation anchor and critic recipe are unchanged from run 01. Both
trials start from the same original imitation parent with fresh PPO/critic
optimizers. The changes are a smaller initial actor LR and post-step analytic
KL enforcement with parameter/Adam rollback. See [the plan](PLAN.md).

## Measured work

- **32 worlds**, **16 CPU physics threads**, GPU neural learning; CPU MuJoCo/mjbatch,
  not Warp. 1,000 Hz physics, 500 Hz control, four neural updates per action.
- **704,512 transitions**, **1,409,024 physics steps**, **1,409.024 aggregate
  simulated seconds**, approximately **1,168 transitions/s** including learning.
- 43 rollouts, **146 actor updates**, 1,376 critic minibatches, 21,504 supervised
  imitation target presentations. More accepted actor updates than run 01,
  despite fewer collected transitions in the same time allowance.
- Collection **238.886560 s**; actor optimization **344.510639 s**, including
  **59.773982 s imitation**; critic **1.809208 s**; memory refresh **18.165151 s**.
- Setup **12.254148 s**, replay audit **3.381664 s**, checkpoint writes **0.151360 s**,
  physical midpoint/final evaluation **89.929249 s**, all outside training.
  PID/parent baseline captures are reused; their original timers are in run 01.
- Peak PyTorch CUDA allocation **4,101,086,720 bytes** (3.819 GiB).
  Full suite: **227 passed in 166.40 s**, 45 known upstream warnings.
- Training began **08:43:21.035178 UTC**; final evaluation/report completed
  **08:54:57.922854 UTC**. The video's exact rendering timer is in its JSON sidecar;
  full decode took **4.347555 s**. These times are separate from neural learning.

Source commit `45ca000`; final checkpoint SHA256
`377093936a91ab1621160c356353f395b30f41db0b9f04461cc7d25f5dcebc63`.
Final/midpoint weights, optimizer state, learning traces and evaluations are
preserved. The original parent remains the development reference for regulation.

A frozen-parent reward audit finds typical horizontal speeds around 3.25 cm/s,
where the original 0.5 cm/s reward width gives little credit for reducing large
drift. [Run 03](../velocity_hover_ppo_03/PLAN.md) changes only that horizontal width
to 2 cm/s, preserving the zero-velocity optimum and physical success thresholds.
It retains the bounded optimizer and returns to the same original parent.
