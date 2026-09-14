# PPO sustains all four flights and slows sideways motion

The focused wing-readout stage produces the first clear physical progress in
this series. **All four fixed starts complete ten seconds airborne**. The original
parent completes two, with the others falling at 6.772 and 4.190 seconds.
Common first-two-second velocity RMS falls from approximately **25.96 to 20.28
mm/s**, a **22% reduction**, satisfying the pilot's initial survival/early-error
milestone. This is not yet satisfactory hover: upward drift increases.

| Start 0 metric | Original parent | Run 04 final |
|---|---:|---:|
| Full ten-second velocity RMS | 31.77 mm/s | 24.40 mm/s |
| Horizontal velocity RMS | 30.68 mm/s | 18.83 mm/s |
| Vertical velocity RMS | 8.23 mm/s | 15.51 mm/s |
| Net climb | 44.68 mm | 152.00 mm |
| Peak displacement | 148.85 mm | 217.18 mm |

The other paired complete start (8) shows the same pattern. Report the climb
and displacement regression alongside the gains; do not label this solved hover.
The two formerly failed starts now complete the full window, so their earlier
shortened RMS values are not a fair full-duration error baseline.

[Watch PID / original / learned flight](../../../../previews/embodied_fly/velocity_hover_ppo_04_comparison_v1.mp4).
Four starts, 43 seconds at 1x, 50 fps, 1920x1080, damped cameras, wing details and
shared error scales. All 2,150 frames decoded, visually inspected and opened
**09:35:41 UTC**. Render **68.708463 s**, decode **4.302793 s**.

## What changed

The deployed actor/body/interface is unchanged. PPO now trains only the existing
**815 -> 128 -> 6 wing readout, 105,222 parameters**. Its encoder, recurrent cell
parameters and base motor decoder remain fixed during this stage. The full
166,700-neuron MaleCNS graph still processes live observations every actor step;
all 78 outputs still come from the same actor. There is no PID assistance, action
mask, additional controller, supplied phase, or force-law change.

With upstream weights fixed, recorded motor-neuron features can be reused exactly
for PPO updates and the light original PID-label anchor. Cached and full recurrent
actions/log probabilities agree within numerical tolerances, including a test
after a readout update and episode resets. Physical-policy gradients reach the
readout; every upstream parameter remains bit-identical. Exploration is fixed
at latent standard deviation .003. LR proposals begin at 3e-6 with the same
post-step analytic KL ceiling .02; 381 accepted updates, maximum .01995085.
Rejected proposals roll back both weights and Adam state.

## Measured work

- **602.165516 s training**, **32 worlds**, **16 CPU physics threads**. RTX 4090
  neural inference/learning; CPU MuJoCo/mjbatch physics, not Warp. 1 kHz physics,
  500 Hz control, four internal neural steps per action.
- **1,753,088 transitions**, **3,506,176 physics steps**, **3,506.176 aggregate
  simulated seconds**, approximately **2,911 transitions/s** including learning.
- 107 rollouts, 381 actor updates, 3,424 critic minibatches, 54,272 imitation targets.
- Collection **595.695655 s**; actor optimization **2.233227 s**, including
  **0.137114 s imitation**; critic **4.210001 s**. Memory refresh unnecessary
  because upstream weights are fixed. These are measured component times.
- Separate setup **16.566958 s**, replay audit **3.388102 s**, writes **0.090672 s**,
  physical midpoint/final evaluation **91.460285 s**. PID/parent baseline reused.
- Peak PyTorch CUDA allocation **1,547,995,136 bytes** (approximately 1.44 GiB).
- Training began **09:18:13.429060 UTC**; final report completed **09:29:50.570761 UTC**.
  Selected ancestry **2,468.314980 s** includes the first imitation checkpoint and
  this stage. Failed pilots 01–03 and discarded smoke weights are not ancestors.
- Source `d47b534`; final SHA256
  `9530e3df05ecf38e70ce4716dae8a9362b9cf29dcd9ec5c2e1ccce53dea79c76`.
  Full fly suite **229 passed in 165.88 s**; GPU replay/scope smoke passes.

[Run 05](../velocity_hover_ppo_05/PLAN.md) continues this final checkpoint with the
same settings and restored optimizer/critic, to test whether the remaining climb
can improve without restarting or changing the reward.
