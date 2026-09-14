# Smaller PPO updates improve vertical control but worsen sideways drift

Keep **run11's midpoint** as the preferred hover policy. Run12 changes only the
PPO KL limit from .02 to .005, with the same parent, exploration .0015, rewards,
32 worlds, optimizer/critic state, brain, trainable wing readout and physics.
All three checkpoints stay airborne for all four ten-second tests, but none
meets the combined improvement gate. No new checkpoint replaces the parent.

| Checkpoint | Training time in this trial | Total velocity RMS | Horizontal RMS | Vertical RMS | Net climb | Ten-second flights |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained run11 midpoint | — | **12.76 mm/s** | **9.19 mm/s** | 8.85 mm/s | 60.73 mm | 4/4 |
| Run12 quarter | 149.90 s | 14.23 mm/s | 12.75 mm/s | **6.33 mm/s** | 46.03 mm | 4/4 |
| Run12 midpoint | 301.10 s | 13.61 mm/s | 11.76 mm/s | 6.85 mm/s | **43.74 mm** | 4/4 |
| Run12 final | 604.08 s | 13.69 mm/s | 11.60 mm/s | 7.26 mm/s | 54.87 mm | 4/4 |

The midpoint has the lowest total error and climb among the new checkpoints,
so the comparison video shows it. It reduces climb **28.0%** against the parent,
but horizontal RMS increases **28.0%** and total RMS increases **6.7%**. These
are means over starts 0,1,8,9; the unchanged overall RMS window is 0.2–10 seconds
with the existing 100 ms velocity average. They are development comparisons
from one training seed, not a broad generalization result.

## Startup and subsequent flight

The new read-only audit separates 0–2 seconds from 2–10 seconds. It changes no
reward or force processing and validates every source capture checksum.

| Checkpoint | Minimum startup height | Startup velocity RMS | Later mean climb rate | Later horizontal RMS |
| --- | ---: | ---: | ---: | ---: |
| Retained parent | 8.53 mm | 16.79 mm/s | 8.08 mm/s | **8.21 mm/s** |
| Quarter | **12.89 mm** | **14.21 mm/s** | **5.57 mm/s** | 13.04 mm/s |
| Midpoint | 10.27 mm | 15.45 mm/s | 5.76 mm/s | 11.73 mm/s |
| Final | 12.52 mm | 14.57 mm/s | 6.71 mm/s | 11.62 mm/s |

Vertical improvement is real: every new snapshot dips less initially and climbs
more slowly afterward. The problem is increased lateral motion, not a lower
starting altitude creating a misleading net-climb score. The planned targets
were total RMS below12.76 mm/s, horizontal RMS below10 mm/s, climb below50 mm and
four complete flights without hiding worse startup. No snapshot passes all of
them. The final continuation also gives back some of the earlier vertical gain.

## What ran

- **604.079904 s training (10 min 4 s)**, 108 rollouts, **1,769,472 transitions**,
  3,538,944 physics steps and 3,538.944 aggregate simulated seconds.
- **32 worlds / 16 CPU MuJoCo-mjbatch threads / RTX4090 neural work**. This is
  CPU physics with CUDA brain inference and learning, not MuJoCo Warp.
- Same 1kHz physics, 500Hz actor, full fixed MaleCNS graph and FlyBody
  `wing_motion_agile_v5` contract. All live motor output still passes through
  the graph. Only the existing 105,222-parameter wing readout is updated.
- 217 actor updates, 3,456 critic minibatches and 55,296 imitation presentations.
  The PID is a training-only replay anchor; it never commands these flights.
  There are 283 failures among 601 completed training episodes, separately from
  the deterministic four-case evaluations.
- Collection597.562803s; actor optimization2.312371s, including imitation.146835s;
  critic4.178672s. Approximately **2,929 transitions/s** including learning.
  Peak PyTorch CUDA allocation1,548,353,536bytes (~1.44GiB).
- Separately: setup15.718949s, recurrent replay audit3.367842s, checkpointIO.100526s,
  physical evaluation134.699693s. Parent/PID captures are reused only after exact
  parent-snapshot and physical-contract checks; evaluation time is excluded
  from the training timer. Rendering, transfer and tests are also separate.
- Training source`373dc76`; start **2026-09-14T19:32:44.961128+00:00**;
  report complete **2026-09-14T19:45:07.240496+00:00**.
- Full fly suite: **256 passed**,45 upstream warnings,164.77s. Focused prelaunch
  checks:33 passed. The tighter limit is shared by sampled/analytic early stops
  and post-update backtracking. Snapshot hashes, frozen upstream parameters,
  constant exploration, matching recipe and replay are independently verified.

The full trial cost is retained even though none of its weights are promoted.
Preferred ancestry stays **3,374.129354s (56min14s)**. Twelve full PPO trials,
including failures, total **7,250.304584s (120min50s)** and **18,399,232 transitions**.

[Watch PID / retained policy / run12 midpoint](../../../../previews/embodied_fly/velocity_hover_ppo_12_comparison_v1.mp4).
All four starts, matched cameras and chart scales, real-time 1x. The right pane
is labeled as a trial, not the preferred policy. Full video QA and rendering
timings are recorded in `video_qa.json` and the adjacent video metadata.

```sh
open previews/embodied_fly/velocity_hover_ppo_12_comparison_v1.mp4
```

## Next action and decision gate

Run one bounded **recovery-start PPO** experiment from the retained run11
midpoint:16 normal-start worlds and16 worlds beginning partway through that
policy's own climbing/drifting flights. Restore physical state, previous actions,
full neural memory and reward history together. This targets the established
motion that still needs correction while retaining startup practice. It is an
explicit change to training starts; there are no added gusts, force assistance,
new reward or external controller. [Implementation plan](NEXT_RECOVERY_PLAN.md).

Keep the .005 cap and .0015 exploration to compare against this run's training
configuration. Allow one roughly ten-minute /108-rollout block, with physical
checks halfway and at the end. Promote only if the unchanged cold-start tests
pass the combined hover targets and show no startup regression; recovery tests
must also show the actor reducing its actual drift. If this targeted block
does not improve the physical outcome, pause further PPO extensions and measure
phase-conditioned control authority through the existing learned interfaces.
This experiment supplies a testable path forward; more training time alone is
not the recommendation.
