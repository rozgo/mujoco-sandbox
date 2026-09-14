# Closed-flight PPO pilot 01: better value fit, no learned return yet

The six routes are implemented and physically possible with the accepted PID.
This first learned pilot does **not** solve them. All seven frozen cases remain
airborne, but zero pass waypoint/return/hold gates. The preferred imitation
checkpoint stays unchanged. Preserve this child as diagnostic evidence only.

## What changed

One shared motor actor receives changing position requests: 32 stationary hover
worlds and 32 worlds cycling through left/right, right/left, forward/backward,
backward/forward, up/down and down/up. Each episode ends with the original
position requested again. Distances are 1.2–1.8 mm, timing scales 0.85–1.15.
The graph, physical body, wing force law, 1,000/500 Hz clocks, action interface,
and bounded reward weights remain unchanged. No teacher acts during PPO.
The original critic inputs now use shuffled individual time/world transitions.

Two critic-only warmup rollouts precede two rollouts with actor updates.
The sensory encoder, sensor-extension map, cell dynamics, motor decoder and
wing readout change. Utility and intention layers, measured graph and fixed
observation statistics stay unchanged. Artifact verification records every
changed/fixed tensor and exact optimizer counts.

## Measured result

| Measure | Preferred parent | PPO child |
|---|---:|---:|
| Mean position RMS over six 12 s routes | 74.103 mm | 75.870 mm |
| Stationary-hover RMS after first second | 77.357 mm | 78.734 mm |
| Complete waypoint/return/hold gates | 0/7 | 0/7 |
| Airborne throughout frozen review | 7/7 | 7/7 |

Route error worsens 2.38%; hover error worsens 1.78%. Final origin errors are
134–137 mm, compared with requested 1.5 mm excursions. This is drift, not
successful directed flight. Both the predeclared progress and acceptance gates
fail. These are matched fixed development cases, not a generalization study.

Critic explained variance on its collected training rollout after each fit is
0.905, 0.874, 0.722 and 0.963. The last preceding trial's final fit was near zero.
That is a useful value-fitting improvement, but it is **in-sample fit** on a
changed curriculum, not proof of general value accuracy or better policy.
The frozen actor result makes this distinction explicit.

## Actual resources and time

- Training: **400.911823 seconds (6 min 41 s)**, plus 7.473046 s setup. The
  requested 300-second allowance completes its current full rollout/update cycle.
- 64 CPU MuJoCo/mjbatch worlds, 16 physics threads; RTX 4090 neural computation.
  This fly trial does not use Warp physics.
- 524,288 world/action transitions; 1,048.576 aggregate simulated seconds
  (17 min 28.6 s); 1,307.739 transitions/s including learning.
- Four rollouts, 64 actor updates and 1,024 critic updates; no KL guard stops.
  Maximum measured actor KL 0.0173124, below the 0.03 guard.
- Collection 139.607833 s; optimization 261.302301 s. Critic warmup is included
  in the reported training time, not added later or hidden.
- 64 complete training episodes, four falls, zero complete tracking successes.
- Actor 2,516,402 parameters; critic 236,033. Peak PyTorch CUDA allocation
  9,073,805,312 bytes (8.45 GiB); this is not total GPU/device usage.
- Parent evaluation: 45.586476 s capture, 7.191927 s setup, 62.052239 s total
  including export. Child: 45.743458 s capture, 7.472256 s setup, 62.424454 s total.
  Each uses seven simultaneous worlds for 12 s, 42,000 transitions, no resets.

See MEASURED_STATS.json for exact values and transition counts per route.
The full package has 191 passing tests; no physics or reward mutation is hidden.

## Review and next decision

[All seven learned cases plus PID hover](../../../../previews/embodied_fly/round_trip_ppo_v1.mp4)
uses recorded states at 1x. Amber marks the requested target, green the actual
axis position; camera following is damped and labeled. The eighth panel is the
accepted stationary PID, not another learned case. Large plotted drift must not
be described as a completed round trip. Full video decode and sampled visual QA
are recorded separately; presentation QA does not change physical failures.

The existing wing rhythm survives, but position correction still needs learning.
Do not extend this same PPO run simply because the critic now fits well. The
next useful step is to measure response to target changes and, if insufficient,
use the accepted PID to teach short corrective movements on this same plant,
then return to PPO. Keep the paired routes as the eventual evaluation and keep
stationary hover in the curriculum. No new training stage is claimed here.

Video render: 70.101679 seconds. Full decode: 600 frames, 50 fps,
1600x1000, 12 seconds at 1x. Sampled frames inspected and opened on the Mac
at 2026-09-14 02:07:35 UTC. Video SHA256 `bcb12be57a46b8bec2e5038063f6392ef0ef159a5525a3c494c0706dc1f8bed7`.
