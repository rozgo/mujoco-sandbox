# Coordinated reward improves sideways control but loses cold-start support

**Do not promote run 14.** Run 11 midpoint remains preferred overall; run 13
final remains the vertical-control candidate. The new midpoint improves total
motion relative to its run 13 parent but trades vertical precision for sideways
control. The final checkpoint fails every cold start at **0.796 s**, despite
completing all six warmed-up recovery cases. This is an unsuccessful overall
continuation with a more specific failure diagnosis, not solved hover.

| Checkpoint | Complete cold flights | Total RMS | Horizontal RMS | Vertical RMS | Net climb |
| --- | ---: | ---: | ---: | ---: | ---: |
| Preferred run 11 midpoint | 4/4 | **12.76 mm/s** | **9.19 mm/s** | 8.85 mm/s | 60.73 mm |
| Experimental parent, run 13 final | 4/4 | 15.53 mm/s | 15.00 mm/s | **4.03 mm/s** | **13.80 mm** |
| Run 14 midpoint | 4/4 | 12.81 mm/s | 9.29 mm/s | 8.82 mm/s | 63.62 mm |
| Run 14 final | **0/4** | — | — | — | — |

Four predefined ten-second development starts, existing 100 ms velocity average
and 0.2–10 s RMS window. Final's raw pre-failure metrics remain in the reports;
its 0.796-second intervals are not comparable full-flight errors. Its negative
height change is a fall, not better altitude regulation. No failed data is hidden.

The midpoint's minimum startup height falls from 12.69 to 9.69 mm. Its later
upward speed rises from 1.71 to 8.23 mm/s. Neither new checkpoint meets the
declared combined-hover and startup-retention criteria.

## What changed

One coordinated velocity reward replaced the separate vertical/horizontal
terms. It retains 5 mm/s precision, maximum rate 3, the causal 100 ms window,
alive/angular/upright terms and terminal penalty. The prerequisite exact-action
audit ranks the saved behaviors correctly on every paired case and passes
recovery/failure counterexamples. [Audit and bank evidence](PREPARATION.md).

The actor starts from run 13 final. A new bank contains that exact parent's
physical and full neural histories, with 24 training and six withheld records.
The split stays 16 normal plus 16 recovery worlds. The first rollout updates
only the critic for the changed reward; all later learning uses the existing
wing readout, unchanged graph, physics, observations and actions, 0.0015
exploration, 0.005 KL limit and the existing light imitation term. There is no
PID actuator control, rescue or new force assistance.

## Recovery and force diagnosis

All three checkpoints complete **6/6** withheld three-second recovery cases.
These histories come from episodes 8–9 and never initialize training worlds.

| Policy | Last-second horizontal RMS | Last-second vertical RMS | Last-second total RMS |
| --- | ---: | ---: | ---: |
| Parent | 15.80 mm/s | **1.39 mm/s** | 15.86 mm/s |
| Midpoint | 9.33 mm/s | 7.88 mm/s | 12.21 mm/s |
| Final | **6.94 mm/s** | 8.30 mm/s | **10.82 mm/s** |

The final actor still sustains flight from established body/brain histories.
However, its greater upward speed remains undesirable. Warm-start success is
not a substitute for the failed cold-start test.

An additional observer records wing force at **every 1 kHz physics tick** while
replaying the exact actions, including complete recovery initialization. All
six replay groups match recorded body positions with **zero error**. No new
neural inference, optimization or physical assistance runs in this diagnostic.

| Phase | Parent raw lift / weight | Midpoint | Final |
| --- | ---: | ---: | ---: |
| Cold start, 0.2–0.6 s | 0.9798 | 0.9752 | 0.9690 |
| Cold start, 0.6–0.796 s | 0.9858 | 0.9757 | **0.9422** |
| Recovery, 2–3 s | 1.0006 | 1.0077 | **1.0081** |

Wings keep sweeping. The late-startup mean absolute sweep speed decreases from
30.81 to 29.45 rad/s; raw lift falls from 98.6% to 94.2% of weight. The final
upward applied wrench, including drag/body response, is 98.3% of weight over
that interval. This is a startup support deficit, not a complete loss of wing
motion. Near-upright orientation persists until failure. The same final weights
produce excess lift after warming up, explaining why simply increasing lift
everywhere would worsen established-flight climb. [Per-tick evidence](force_audit.json).

## Measured work and preservation

- **603.458466 s training (10 min 3 s)**, 108 rollouts, **1,769,472 transitions**.
  32 worlds, 16 CPU MuJoCo/mjbatch threads, RTX 4090 neural work; no Warp.
  1 kHz physics, 500 Hz actor; only the existing 105,222-parameter wing readout
  updates. Full graph inference remains live.
- 212 accepted actor updates; maximum KL **0.004989475**. Upstream weights
  unchanged. Full and cached recurrent replay pass. First-rollout actor updates
  are zero, verifying the explicit critic adaptation.
- Collection 596.875102 s; actor optimization 2.191637 s (includes 0.150712 s
  imitation); critic 4.365754 s. About **2,932 transitions/s** including learning.
  3,456 critic minibatches; 54,784 imitation presentations.
- Normal training episodes: 502 failures among 638 completed, mean 2.684 s.
  Recovery episodes: 10 among 177, mean 9.749 s. These changing/noisy training
  episodes are distinct from deterministic evaluation results.
- Separate setup 15.135393 s, replay audit 3.378252 s, evaluation 54.431851 s,
  checkpoint IO 0.066014 s. Recovery evaluation totals 51.325488 s including
  setup/capture/output. Force audit 37.781572 s. Peak PyTorch CUDA allocation
  1,555,027,456 bytes, approximately 1.45 GiB.
- Source `a690867`; training started **20:48:55.846420 UTC**, report completed
  **20:59:57.192225 UTC**, September 14, 2026. Preparation costs are separate.
- Full fly suite **266 passed**, 45 upstream warnings, 171.60 s. Later read-only
  force instrumentation passes exact physical replay; video changes pass lint,
  full decode and visual inspection.
- Preferred lineage remains **56 min 14 s**. Fourteen full PPO trials, including
  unsuccessful continuations, total **8,457.843753 s (140 min 57 s)** and
  **21,938,176 transitions**. Do not erase unsuccessful experiment cost.

[Watch PID / parent / midpoint / failed final](../../../../previews/embodied_fly/velocity_hover_ppo_14_comparison_v1.mp4).
All four cases at 1x, 43 s, 2,150 frames, 50 fps, 2560×1080. Render 80.914830 s;
full decode 5.907844 s. Inspected and opened **21:08:51 UTC**, **27 min 53 s**
after this effort began. Reports and synchronization follow separately.

```sh
open previews/embodied_fly/velocity_hover_ppo_14_comparison_v1.mp4
```

## Next action

Measure phase-conditioned control authority through the existing wing decoder,
starting with the preserved run 13 parent. The force audit shows that startup
needs more support while warmed-up flight needs less: a uniform lift increase
is inappropriate. Small decoder-parameter probes must establish whether lift
and lateral braking can be adjusted independently in both conditions. Use that
evidence to choose the next learning change; do not extend unchanged PPO or
claim the reward audit alone solved learning. [Next diagnostic plan](NEXT_CONTROL_PLAN.md).
