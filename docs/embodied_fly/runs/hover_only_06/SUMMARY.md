# Tighter vertical reward preserves flight but does not reduce settled bobbing

The actor reacts quickly to sensed disturbances, so the 500 Hz control clock is
not missing wingbeats. The phase probe records small wing-command changes within
0–4 ms, scalar-lift changes within 4–14 ms, and first correctly signed sustained
lift changes within 4–60 ms. Initial correction direction depends on wing phase;
all 16 branches have the correct average lift direction over 200 ms. Height
corrections remain small. See [probe and limitations](../hover_response_01/SUMMARY.md).

One 608.933-second PPO continuation tightens the vertical-speed reward scale from
50 to 20 mm/s, with the same weight and all other physical reward terms unchanged.
The actor, actor Adam, exploration, graph, action interface and physical body
continue from pilot05. The separate critic and its Adam are explicitly reset for
the changed return objective. Four actor-frozen fitting rollouts take 34.430 s,
included in that training time. There is no PID supervision, phase clock, output
mask, extra world-force controller or newly introduced motor stage.

| Nominal measurement | Before, pilot05 | After, pilot06 | PID |
| --- | ---: | ---: | ---: |
| Altitude RMS error, after first second | 2.349 mm | 2.203 mm | 0.036 mm |
| Height span, after first second | 5.590 mm | 5.271 mm | 0.131 mm |
| Vertical-speed RMS, after first second | 57.998 mm/s | 57.972 mm/s | 13.537 mm/s |
| Peak position error, after first second | 58.903 mm | 58.858 mm | 0.197 mm |
| Mean repeated height ripple, seconds6–10 | 2.797 mm | 2.797 mm | 0.101 mm |
| Dominant wing-sweep frequency, seconds6–10 | 9.25 Hz | 9.25 Hz | 30 Hz |

The altitude RMS improvement is about 6.2%, and the full after-startup span falls
about 5.7%. This is mostly settling/mean-height improvement. Settled ripple, wing
range (about96 degrees), vertical speed and horizontal drift barely change.
All three ten-second starts remain airborne; all three still fail the original
accurate-hover gates. PID's complete result dictionary matches earlier captures.
No falls or numerical failures are hidden. This is one development continuation,
not evidence that this reward change or PPO cannot work under other settings.

The tighter vertical-speed score on the frozen nominal trajectory changes only
from .417776 to .418086. The height score improves more, from .950272 to1.014523.
These are selected per-second physical reward terms on saved pre-action states
after one second, not full returns. The [comparison and method](comparison.json)
separate those terms from independent physical measures and windows.

The critic is correctly updated and its final collected-rollout fit RMSE falls
from .05663 to .04078 in that rollout. Its explained variance remains near zero
(-.00027 after fitting). These are bootstrapped training targets, not held-out
future returns; low mean fit error is not proof of useful temporal discrimination.
It remains a possible contributor to the plateau, not an established cause.

## Measured execution

- 64 hover worlds, 16 native CPU MuJoCo/mjbatch physics threads; RTX4090 neural
  inference/training. 1,000 Hz physics, 500 Hz actions, two physics ticks/action.
- 608.932846 s training, 7.851417 s separate setup; 18 rollouts,
  589,824 physical action transitions and 1,179.648 aggregate simulated seconds.
- 112 actor updates, 144 critic updates, four critic-only warmup rollouts;
  no KL stops. Median observed maximum update KL .014626, below the .03 limit.
- All 192 completed five-second training episodes survive. Partial final episodes
  are not counted as completed successes. The harder reset curriculum stays locked.
- Collection/inference152.246475 s, optimization456.666741 s, both included in
  training. Throughput968.62 transitions per training second includes warmup;
  this is not a matched throughput benchmark against pilot05.
- Peak PyTorch CUDA allocation7,506,160,128 bytes; not total device usage.
  Actor2,516,402 parameters, critic236,033; all78 actions remain learned outputs.
- Evaluation36.691565 s, four ten-second worlds,20,000 actions; no optimization.
- Full suite176 passed in148.93 s,45 dependency warnings; focused12 passed
  in5.73 s. Video decoding, visual inspection and opening are recorded separately.

Source `408ff2a`; the raw configuration, timings, gradient audit and exact hashes
are in [training.json](training.json), [statistics](MEASURED_STATS.json) and
[artifact verification](artifact_verification.json). Verification confirms only
the intended reward-scale change, unchanged graph/body/utility/normalization,
finite learned tensors, 112 inherited actor-Adam increments and144 fresh critic
increments. There are no failure trace windows in this run.

New-plant ancestry is pilots01,04,05,06:2,131.976 s (35 min31.976 s). All six
new-plant pilots cost3,041.234 s (50 min41.234 s), including discarded02/03.
The actor also inherits earlier motor training; these are not from-scratch totals.

## Review and next decision

[New policy versus PID](../../../../previews/embodied_fly/hover_only_pid_comparison_v6.mp4)
and [before versus after](../../../../previews/embodied_fly/hover_only_before_after_v6.mp4)
use matched fixed overviews, identical detail magnification and common error
scales at1x. No visual motion scaling or physics changes are applied.

Do not spend another identical training block expecting the clock to fix this.
The next decision should address learning a steadier wing-generated force pattern
and the critic's weak temporal discrimination. A short, explicitly disclosed
initialization from the accepted PID on this exact plant, followed by PPO and
actor-only evaluation, is one option. It is not implemented by this run, and no
runtime PID or oscillator has been added. Accurate hover and the broader learned
fly goal remain open; the previous checkpoint/video are preserved.
