# One-body motor learning, first pilot

The new curriculum isolates stand, straight walk and hover under one MaleCNS
actor and exactly one physical FlyBody preset. Utility selection is disabled;
the measured graph stays fixed while sensory interfaces, intrinsic cell dynamics
and the motor decoder learn. All 78 actuator outputs remain active. No runtime
teacher or wing oscillator is present in the independent student review.

Source `90da9e0`, seed 71001, preserved angle01 walking parent, fresh Adam 1e-5.
Training executes 80% reference / 20% student commands and imitates the reference
on actual visited states. This is **online motor imitation**, not PPO.

| Measurement | Actual result |
| --- | --- |
| Training wall time | 180.624521 s |
| Setup | 8.497658 s |
| Collection and forward inference | 140.233420 s |
| Backpropagation and optimization | 40.386593 s |
| Physical worlds | 32: 11 stand, 11 walk, 10 hover |
| CPU physics threads | 16 |
| Brain and learning device | RTX 4090 / CUDA |
| Physics / control | 5,000 / 500 Hz |
| Updates / transitions | 162 / 165,888 |
| Aggregate simulated experience | 331.776 s |
| Peak CUDA allocation | 9,411,618,816 bytes |
| Actor / trainable parameters | 2,410,924 / 2,231,318 |
| Measured graph | 166,700 neurons / 25,582,938 connections |

All 160 completed assisted episodes reached two seconds without a training
envelope failure. **All three unassisted two-second cases failed.** First envelope
exits were 0.418 s standing, 0.268 s walking and 0.058 s hovering. Initial hover
sweep speeds were insufficient for lift. Assistance had concealed the remaining
closed-loop motor errors; lower imitation loss was not autonomous success.

The [complete six-second film](../../../../previews/embodied_fly/motor_focus_01_all_tasks_v1.mp4)
shows every case at 1×, with actual simulated neural state and observer eyes.
It was fully decoded, visually inspected and opened for the user. The original
review's forbidden-load field inadvertently used only its last control interval;
all cases independently failed posture. That report/video remain preserved. The
fixed evaluator takes the maximum across the entire clip, with an early-contact
regression check. Subsequent results use the corrected reduction.

The Mac/Linux compiled physical arrays differ only by rounding (at most
2.60e-14 absolute); exact fingerprints therefore differ across hosts. Native and
batch models match within each host. Rendering uses the exact captured GPU-host
MJB. Massless non-colliding wings and the custom wing-motion force law are shared
by every task. No candidate is promoted; the earlier ground reference is intact.
