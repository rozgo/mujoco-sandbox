# Uninterrupted PPO experience; modest height tracking improvement

All 192 completed five-second training episodes finish without physical failure.
The frozen actor also stays airborne on all three ten-second review starts.
Accurate hover remains unsolved: bobbing is essentially unchanged and position
drift remains nearly 6 cm. The [new PID/PPO video](../../../../previews/embodied_fly/hover_only_pid_comparison_v5.mp4)
shows this gap at 1x with matched starts, body, cameras and error scales.

Frozen probes identified destabilizing exploration: three of eight sampled
starts failed at .003 latent standard deviation, none at .001. We continue
pilot04, explicitly reset exploration and its Adam moments, and lower actor LR
from 3e-7 to 1e-7 for the narrower distribution. Actor/critic weights and Adam
histories resume; no repeated critic warmup occurs. These are paired changes,
not an isolated causal experiment. Lower noise can also limit new exploration.

| Nominal measurement, after first second | Pilot04 | Pilot05 | PID |
| --- | ---: | ---: | ---: |
| Altitude RMS error | 2.768 mm | 2.349 mm | 0.036 mm |
| Height span | 5.569 mm | 5.590 mm | 0.131 mm |
| Position RMS error | 35.575 mm | 35.158 mm | 0.124 mm |
| Peak position error | 59.648 mm | 58.903 mm | 0.197 mm |
| Vertical-speed RMS | 58.423 mm/s | 57.998 mm/s | 13.537 mm/s |

Reduced average altitude error does not mean reduced bobbing. All three actor
starts fail the reference-quality gate. PID's result dictionary is unchanged
from prior captures. All 78 channels execute without a teacher, PID, oscillator,
output mask or body-state correction. Utility stays disabled; fixed graph and
routing are preserved while cell dynamics and sensory/motor interfaces learn.

Training takes **614.579 s**, with **7.411 s** separate setup: **491,520 physical
transitions**, **983.040 aggregate simulated seconds**, **120 actor / 120 critic
updates**, **zero KL stops** and **zero failures among 192 completed episodes**.
Pilot04 had 47 failures among 153 episodes under the different noise setting.
All **64 worlds hover**, using **16 native CPU MuJoCo/mjbatch threads** and an
**RTX 4090 for neural work**. Physics stays **1,000 Hz**, actions **500 Hz**.
Collection/forward inference takes **126.715 s** and optimization **487.850 s**,
both included in training. Throughput is **799.77 transitions/training-second**;
peak PyTorch allocation is **7,506,579,968 bytes**, not total device usage.
The ten-minute budget finishes its fifteenth complete rollout. No harder reset
curriculum unlocks. Training survival is separate from accurate-hover acceptance.

Source `41b6616`; **22 focused tests pass in 36.14 s** and the **full 175-test
suite passes in 152.17 s** (45 dependency warnings). Verification confirms finite
tensors, unchanged graph/body/utility/normalization, 120 additional Adam steps
on each trained actor/critic parameter, and fresh exploration history. No failed
trace windows are produced. The video is fully decoded, inspected and opened;
see [verification](video_verification.json) for the exact recorded milestone.

New-plant ancestry is pilots01,04,05: **1,523.044 s (25 min 23.044 s)**. All five
pilots cost **2,432.301 s (40 min 32.301 s)** and collect **4,358,144 transitions**.
Older motor training predates these totals; this is not training from scratch.
Frozen probes, evaluations and rendering are separate in [the time log](../../../TIME_LOG.md).

See [statistics](MEASURED_STATS.json), [evaluation](evaluation.json),
[parameter/optimizer verification](artifact_verification.json) and [recipe](PLAN.md).
The broader survival goal remains open. Accurate hold must improve before
advancing to the next motor skill or learned utility.

Settled motion diagnosis, t=6–10 s: PID sweeps at 30 Hz through about 30 degrees, PPO at about 9.25 Hz through about 96 degrees. Both controllers act at 500 Hz. Mean sweep speeds are about 31 rad/s for both; mean per-cycle height ripple is .101 mm for PID and 2.797 mm for PPO. The latter has -2.240 mm mean height bias. The earlier 5.590 mm span includes settling and is not pure periodic ripple. See [method and measurements](wing_diagnostic.json).
