# Half exploration improves learned hover; retain the five-minute checkpoint

Run 11 resumes run 05 with the same rewards, brain, physical model and optimizer
state, changing fixed exploration standard deviation **.003 -> .0015**. The
predeclared midpoint is better than both the parent and the final continuation.
Retain `velocity_hover_ppo_11_midpoint.pt`; preserve the final and run 05.

| Checkpoint | Ten-second flights | Total velocity RMS | Horizontal RMS | Vertical RMS | Net climb | Peak displacement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Run 05 parent | 4/4 | 14.92 mm/s | 9.30 mm/s | 11.67 mm/s | 105.99 mm | 117.10 mm |
| Run 11 midpoint, retained | 4/4 | **12.76 mm/s** | **9.19 mm/s** | 8.85 mm/s | **60.73 mm** | **75.76 mm** |
| Run 11 final | 4/4 | 13.59 mm/s | 10.66 mm/s | **8.44 mm/s** | 68.23 mm | 94.37 mm |

Means over the same four fixed development starts. Existing 100 ms averaged
velocity and 0.2–10 s RMS interval are unchanged. The retained midpoint reduces
total RMS **14.5%**, climb **42.7%** and peak displacement **35.3%** versus run 05.
Sideways RMS is slightly lower, rather than buying vertical improvement with a
sideways regression. Final updates regress total RMS, sideways RMS, climb and
displacement relative to midpoint; they do not replace it.

This is progress, not completed stationary hover. Climb remains **60.73 mm**,
above the 50 mm goal. Early 0–2 s velocity RMS worsens from **15.59 to 16.79 mm/s**
at midpoint; the improvement is over the longer flight, not a better cold start.
These four development starts and one training seed do not prove broad
robustness or learned velocity-command flight beyond the zero-command stage.

## Why this experiment

[Frozen-policy diagnostic](../hover_exploration_01/SUMMARY.md): without changing
weights, zero/half/current noise survives **32/32, 30/32 and 18/32** ten-second
flights. Eleven of 14 current-noise failures occur by 1.094 s. This isolates an
exploration effect and supports the smaller-amplitude trial. It does not prove
that noise alone caused every changing-policy failure. Run 11 still records
**265 failures among 584 completed training episodes**; evaluation remains
separate from the changing weights and sampled actions used during learning.

The deployed actor always runs the full MaleCNS graph. Only the existing
105,222-parameter wing readout learns during this continuation. No PID acts
during physical PPO collection or evaluation. The original light PID replay
anchor remains at weight 1. All 78 actuator commands remain in the policy's
action distribution, sampled every 2 ms. Physics reads actual wings every 1 ms.

The smaller Normal scale is used consistently in collection, replay and KL.
It also tightens the absolute action change allowed by the same .02 KL cap.
No command or force filter, supplied oscillator, actor bypass, reward change,
new curriculum, Adam reset or physical assistance is introduced.

Two startup checks stopped before optimizer updates because cached float32
matrix-batch roundoff became more visible with smaller noise. The dedicated
probe measured aggregate distribution error 2.11635e-6. Cached replay now checks
aggregate KL ≤ 4e-6, maximum action error ≤ 2e-6 and maximum log-probability error ≤ .02;
the full-core recurrent check remains unchanged. Only validation tolerances
change; replay arithmetic and gradients do not. Both stopped attempts remain
recorded in [the diagnostic plan](../../EXPLORATION_DIAGNOSTIC.md).

## Measured cost and checkpoint lineage

- **606.860980 s training (10 min 7 s)**, **108 rollouts**, **1,769,472 transitions**,
  **3,538,944 physics steps**, **3,538.944 aggregate simulated seconds**.
  427 actor updates, 3,456 critic minibatches, 55,296 imitation presentations.
- Selected midpoint: **301.325884 s additional training**, 54 rollouts,
  **884,736 transitions**. Full ten-minute trial cost remains recorded; selecting
  the earlier checkpoint does not erase the remaining compute.
- Selected ancestry: **3,374.129354 s (56 min 14 s)** including its original
  imitation and productive PPO parents. Eleven full PPO trials including
  failures cost **6,646.224680 s (110 min46s)** and 16,629,760 transitions.
- Same **32 worlds**, 16 CPU MuJoCo/mjbatch threads, RTX 4090 neural work. This
  is not MuJoCo Warp. 1 kHz physics / 500 Hz actor; fixed MaleCNS and same FlyBody.
- Collection **600.095070 s**, actor optimization **2.685769 s**, critic
  **4.052825 s**. Imitation **.142836 s** is included in actor optimization.
  Peak CUDA allocation **1,548,877,824 bytes**. Approximately **2,916 transitions/s**
  including learning, measured separately from setup and evaluation.
- Separate setup **14.829247 s**, replay audit **3.372205 s**, checkpoint IO
  **.071106 s**, physical evaluation **90.199609 s**. Stopped startup checks,
  rendering, file transfer and documentation are not training time.
- Training source `04d7457`; started **2026-09-14T18:45:26.818510+00:00**;
  report completed **2026-09-14T18:57:07.352876+00:00**.
- Midpoint SHA256 `69061012aa237832fcbd42be2b39bc9c2e6530932865b82d4df5190af9f56077`.
  Final SHA256 `79caefcf879e92b574ab07f95133d3e2caf10e4e450ff055aed71e9e88af746e`.
- Focused sampling, replay and optimizer tests: **27 passed**, one upstream
  warning, **1.49 s**. Source/parent/graph/physics and matched recipe checks are
  archived with the completed verification report.

[Watch PID / run 05 / retained midpoint](../../../../previews/embodied_fly/velocity_hover_ppo_11_comparison_v1.mp4).
The right panel explicitly identifies the midpoint; final results remain in
the table and archived captures. Same starts, cameras, shared chart scales, 1x.

```sh
open previews/embodied_fly/velocity_hover_ppo_11_comparison_v1.mp4
```

Video verification: **43 s / 2,150 frames / 50 fps / 1920 × 1080 / 1x**.
GPU render **69.270898 s**, full decode **4.287509 s**. Inspected and opened
**2026-09-14 19:03:45 UTC**, 30 min 29 s after this effort began. The video
metadata verifies the selected midpoint SHA256.
