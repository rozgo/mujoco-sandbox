# Longer reward credit preserves flight but does not improve hover

The user-approved trial retains the preferred imitation actor's approximately
29.75 Hz wing rhythm and three ten-second airborne starts. It does not improve
altitude loss or horizontal drift, so it fails the predeclared development
progress criteria. Preserve the earlier imitation checkpoint as the next
development parent. No new motor stage or runtime controller was introduced.

| Nominal result | Preferred imitation parent | Longer-credit PPO |
| --- | ---: | ---: |
| Airborne review starts | 3/3 | 3/3 |
| Original accurate-hover passes | 0/3 | 0/3 |
| Altitude RMS, after first second | 8.724 mm | 8.793 mm |
| Position RMS, after first second | 63.708 mm | 64.009 mm |
| Peak position error | 108.999 mm | 109.484 mm |
| Mean height error, seconds6–10 | -9.861 mm | -9.949 mm |
| Mean repeated height ripple, seconds6–10 | 0.09487 mm | 0.09488 mm |
| Dominant sweep frequency, seconds6–10 | 29.75 Hz | 29.75 Hz |

Both perturbed starts stay within the declared10% regression limit, but neither
nominal position RMS nor altitude RMS improves by the required10%. Small ripple
around the wrong height is not accurate hover. The matched PID still passes its
original quality gate. See [all measurements and progress checks](comparison.json),
[frozen evaluation](evaluation.json) and the [predeclared plan](PLAN.md).

## Exactly what changed

The same399-input/78-output MaleCNS actor, graph routing, physical fly, reward,
1,000 Hz physics and500 Hz action rate remain. All64worlds train hover with
16CPU MuJoCo/mjbatch threads and RTX4090 neural computation; this is not Warp.
The longer rollout grows512 ->2048actions (1.024 ->4.096s). Discount decay grows
about2 ->5s; combined gamma/GAE trace decay grows.333 ->2s. Episodes grow5 ->10s.
The128-action (.256s) recurrent gradient chunk stays fixed for GPU memory.
Neural state persists across chunks; longer return credit does not imply longer
full-graph backpropagation. The [numeric timing audit](timing_audit.json) verifies
the changed decay and that episode boundaries stop cross-episode reward credit.

Same actor LR1e-7, critic LR1e-4, exploration standard deviation/floor.001,
two epochs, KL limit.03 and entropy0. The selected parent is imitation, so PPO
and critic optimizers start fresh; no older PPO optimizer history is inherited.
Four actor-frozen critic-fitting rollouts are included in measured training time.
The reset curriculum is unchanged, and harder starting offsets remain locked.
No PID labels, rehearsal, phase target, wing oscillator or direct force controller.

Training takes **620.901816 seconds**, plus7.720015 s setup. The requested600s
finishes the last rollout/update rather than truncating it. It collects
**1,703,936 physical action transitions**, **3,407.872 aggregate simulated seconds
(56 min47.872 s)** across64worlds:13rollouts,41actor and416critic updates.
All320completed ten-second episodes survive. Collection takes442.609408 s and
optimization178.287360 s; the four critic-only rollouts take135.805563 s inside
the total. Peak PyTorch CUDA allocation is14,281,320,448bytes. These are continued
pretrained actors and aggregate multi-world experience, not from-scratch or
real-time inference figures.

## What the divergence guard did

PPO compares action probabilities from its updated actor with the policy that
collected the rollout. This approximate KL measure triggers a stop before
further actor steps when it exceeds.03. The latest weights remain; this is not
automatic rollback or a physical fall detector. Independent critic fitting
continues after a stop.

Eight of nine actor-learning rollouts hit this guard. The first eight accepted
nine actor updates in total; the final round accepted32more with maximum observed
KL.0168. Across active rollouts the median maximum KL is.0515 and the largest.1496.
Thus the guard limited early learning, but did not prevent all updates. This run
does not establish that loosening it would improve flight. More collected
experience cannot be reported as equivalent to more actor optimization.

The final critic fit reduces RMSE1.214 ->.977 on its fixed bootstrap targets,
with explained variance approximately-.0006. Its weak discrimination between
returns remains an unresolved learning issue. These are training-target metrics,
not independent prediction accuracy. Nonzero physical-reward gradients reach
cell gain/leak/bias; fixed graph, body, routing, normalization and utility context
are verified along with finite tensors and fresh Adam step counts.

## Decision and next question

The earlier user-reviewed imitation actor remains preferred:
`assets/embodied_fly/diagnostics/pid_imitation_01_teacher_stage.pt`, SHA256
`315ef3a259ed0e74757163bf58ab22feabbc2049d24992a5028d6802805df1d3`.
The longer-credit child is retained as diagnostic evidence, SHA256
`be6128d0566aaca6f264ac9d8517bffa3ca28c226a215a03a93cdfc966cc4655`.
This trial alone does not rule out longer credit; it establishes no gain with
this complete recipe. Do not start another identical timed continuation and
expect the elapsed minutes to establish progress.

A useful next investigation is whether exploration and the value signal
distinguish corrective wing changes from the already good periodic motion.
The guard and critic are plausible constraints, not proven sole causes.
The read-only [input range review](input_range_review.json) finds horizontal
feedback clipping only after9.558s of the preserved trajectory, after substantial
drift has already occurred; it does not explain the onset. No clock, input,
architecture, exploration or reward change is silently applied to this run.

Eleven focused timing/critic/hover tests pass in7.66s. The unchanged implementation
previously passed179tests in157.91s, so that entire suite was not repeated for
this CLI-only experiment. The source is6570f52. See [validation](validation.json),
[training](training.json), [statistics](MEASURED_STATS.json) and
[artifact verification](artifact_verification.json).

Both final videos were fully decoded (500frames,50fps,ten seconds,1600x900,1x),
sampled frames visually inspected and opened on macOS at00:22:29 UTC on
September14. That is20min30s after the observed effort start. Frozen capture
takes36.779771s plus6.819405s setup. PID/PPO and before/after rendering take
65.400899s and71.483189s concurrently. Training, evaluation, rendering and
elapsed development time are recorded separately; the latter also includes
implementation review, transfers, discussion and documentation.

- [Preferred parent beside longer-credit PPO](../../../../previews/embodied_fly/hover_credit_before_after_v1.mp4)
- [Longer-credit PPO beside PID](../../../../previews/embodied_fly/hover_credit_pid_comparison_v1.mp4)

```sh
open previews/embodied_fly/hover_credit_before_after_v1.mp4
```
