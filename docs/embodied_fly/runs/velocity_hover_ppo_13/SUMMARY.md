# Recovery-start PPO learns better vertical control; sideways drift still regresses

Keep **run11 midpoint** as the preferred overall hover policy. Preserve run13
final as a useful vertical-control candidate. Its net climb falls77.3%, startup
improves, and it brakes upward motion in all six withheld recovery tests, but
horizontal RMS rises63.3% and total RMS rises21.7%. Neither new checkpoint meets
the combined hover gate. No further training is running.

| Policy | Ten-second flights | Total velocity RMS | Horizontal RMS | Vertical RMS | Net climb |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained run11 midpoint | 4/4 | **12.76 mm/s** | **9.19 mm/s** | 8.85 mm/s | 60.73 mm |
| Run13 midpoint,5min | 4/4 | 14.87 mm/s | 13.81 mm/s | 5.52 mm/s | 13.97 mm |
| Run13 final,10min | 4/4 | 15.53 mm/s | 15.00 mm/s | **4.03 mm/s** | **13.80 mm** |

Means over the same four development starts0,1,8,9. Existing100ms velocity
average and0.2–10s RMS window remain unchanged. The same actor, reward, physics,
exploration and update settings are used as run12; only episode initialization
changes. Thirty saved histories come from the frozen parent's actual flight:
24 training histories and six withheld ones. No teacher acts in these flights.

## Startup and withheld recovery

Final startup minimum altitude improves from8.53 to12.69mm, and first-two-second
velocity RMS from16.79 to13.92mm/s. During2–10s, upward speed averages1.71mm/s
versus8.08mm/s for the parent. This is better vertical regulation, not a lower
initial altitude creating an artificially small net climb. The midpoint dips
slightly farther than the parent (7.85mm minimum) and has poorer vertical
regulation than final; final is the video candidate, not a promoted policy.

All six withheld three-second recoveries complete under parent, midpoint and
final. They start from the same physical/neural histories captured at2,4,6s
in episodes8/9; none of those histories initializes a training world.

| Policy | Mean upward speed, first→last second | Horizontal RMS, first→last second |
| --- | ---: | ---: |
| Retained parent | 8.21→7.99 mm/s | 7.75→8.46 mm/s |
| Midpoint | 5.16→2.31 mm/s | 11.05→14.27 mm/s |
| Final | **5.45→1.37 mm/s** | 11.93→15.86 mm/s |

This shows vertical recovery from the withheld histories, alongside worsening
sideways motion. It is one training seed with a narrow set of development
starts, not proof of broad flight-command generalization or complete hover.

## Why more unchanged training is not the next recommendation

An exact-action replay through the same GPU-machine physical plant reproduces
every recorded body position with zero error and scores the original reward:

| Policy | Total ten-second reward | Vertical contribution | Horizontal contribution |
| --- | ---: | ---: | ---: |
| Retained parent | 33.8093 | 5.7838 | 8.4001 |
| Midpoint | 40.6615 | 14.3020 | 6.8979 |
| Final | **41.4673** | **15.7087** | 6.5450 |

The scorer prefers the new flight, despite its worse total motion error. The
vertical gain earns9.9248 additional reward; the sideways regression costs only
1.8552. This identifies a mismatch between the retained training objective and
the combined-hover selection criterion. It does not prove reward weighting is
the only limitation, but it gives a concrete issue to address before changing
the network or spending more time on the unchanged objective.

Reward stayed fixed intentionally to isolate the recovery-start experiment.
The audit changes no rewards or weights. Its31.379470s physical replay has no
neural inference or optimization and is excluded from training time. A local
Mac attempt stopped at the physical-fingerprint preflight; the scored replay
ran on the original GPU machine with the matching plant. No cross-platform
physical equivalence is assumed.

## Implementation, validation and measured cost

- **604.080703s training (10min4s)**,108rollouts,**1,769,472 transitions**,
  3,538,944physicssteps,3,538.944aggregate simulated seconds. Midpoint at302.428147s.
- **32 worlds**,16 normal starts+16 recovery starts,16 CPU MuJoCo/mjbatch threads,
  RTX4090 neural inference/learning. **CPU physics, not MuJoCo Warp**.
  1kHz physics/500Hz full-connectome actor; only the existing105,222-parameter
  wing readout updates. Same .0015 exploration and .005 KL limit.
- Full integration state, previous actions, sensor/wing bookkeeping, full neural
  memory and causal reward history restore together. Recurrent PPO replay
  supports nonzero resets inside sequences. [Restoration proof](RESTORATION.md)
  includes exact recorded-action replay and two stopped numerical audit attempts.
-216 actor updates,3,456 critic minibatches,55,296 imitation presentations.
  All accepted KL values≤.005 (maximum.004991266); upstream weights unchanged.
  Full-core replay and cached likelihood checks pass with unchanged tolerances.
- Normal worlds:197 failures among344 completed training episodes,mean duration
  4.923s. Recovery worlds:11 among176,mean9.794s. This is training under changing
  weights and exploration, separately from deterministic evaluation. The192
  recovery initializations use training histories only, including initial worlds.
- Collection597.699857s; actor optimization2.247471s (includes imitation.142410s);
  critic4.107126s. Approximately2,929 transitions/s including learning.
  Peak PyTorch CUDA allocation1,559,323,136bytes (~1.45GiB).
- Separately:setup16.340686s,replay audit3.380689s,checkpointIO.066760s,
  ordinary physical evaluation90.848644s. Recovery evaluation takes33.342988s
  capture+13.093565s setup across three policies, plus file output. Preparation,
  tests, reward diagnostic, rendering and transfer are not training time.
- Training source`750f5d6`; start **2026-09-14T20:10:49.296215+00:00**;
  report complete **2026-09-14T20:22:27.702591+00:00**.
- Full suite259passed/45upstream warnings/169.46s; two new recovery-scoring tests
  pass separately. GPU restoration tests3passed. Checkpoint, bank, physical
  contract, graph and matched recipe checks pass.
- Preferred ancestry remains3,374.129354s (56min14s). Thirteen full PPO trials,
  including failed continuations, total7,854.385287s (130min54s) and20,168,704
  transitions. The unsuccessful trial cost is not erased by keeping its parent.

[Watch PID / retained policy / final vertical-control trial](../../../../previews/embodied_fly/velocity_hover_ppo_13_comparison_v1.mp4).
Four complete cases,43s/2,150frames/50fps/1920×1080/1x. Final is clearly labeled
as a trial. Render69.772786s; full decode4.370024s. Inspected and opened
**20:30:20 UTC**,34min6s after this effort began. Reports and synchronization
follow separately. The original policy and all earlier videos are preserved.

```sh
open previews/embodied_fly/velocity_hover_ppo_13_comparison_v1.mp4
```

## Next action

Validate a coordinated3D-velocity reward on these saved flights before more
training. Keep the current5mm/s vertical precision, rather than loosening it
as the earlier20mm/s vector-reward trial did. Require the proposed scoring to
favor low motion in all directions and recoverable flight over early failure.
If that audit passes, resume one bounded PPO block from the new final vertical
candidate with the same body, brain and32-world mix. Promote only on improved
combined hover, with vertical/startup gains retained. If a correctly ordered
objective still fails, keep training paused for phase-conditioned control and
representation diagnostics. [Concrete proposed plan](NEXT_REWARD_PLAN.md).
