# Ground motor curriculum (09)

**Not selected.** Full sensory-to-motor training keeps both ground commands
upright for five seconds, but resting-wing accuracy does not beat preferred06.
This is a curriculum stage of the same command-conditioned actor, with hover
still required and evaluated; it is not a separate deployed walking policy.

- Source9569f3d;parent06;seed71009;freshAdam1e-5.
- 181.150672s training plus8.816460s setup;129updates.
- 32worlds:16standing/16walking,16native CPU MuJoCo threads.
- RTX4090 graph/learning;5kHz physics/500Hz control.
- 132,096physicaltransitions/264.192aggregate simulated seconds.
- Collection/forward130.998965s;backward/optimization50.142515s.
- Peak CUDA allocation13,824,745,472bytes.
- Initial wing disturbances:±0.15rad within joint limits and±2rad/s.
  Only training reset state changes; the canonical rest target remains fixed.
- Ground posture targets,wing weight10,paired-response weight1,eight ground
  examples per action. Every executed action comes from the student.
- All128 completed episodes reach2s without falling. This is not full success.

Five-second developmentseed72001 evaluation keeps standing and walking upright
with permitted support. Standing/walking wing RMS is0.1637/0.0751rad versus
06's0.1140/0.0631rad. Standing height loss improves to0.508%, but all complete
tracking/posture gates still fail. Hover, untrained in this stage, falls.
Zero numerical warnings. Evaluation takes33.540868s plus5.861885s setup.

Finite motor gradients reach all intrinsic cell-parameter groups. Utility and
context weights stay frozen; physical fingerprint, graph and actor architecture
are unchanged. Every checkpoint/model/capture hash and causal action sequence
is verified. This is online corrective imitation, not PPO.

[Complete review](../../../../previews/embodied_fly/motor_focus_09_all_tasks_v1.mp4).
15s,750frames,1600×900,50fps,1×. Render/encode/full decode60.432509s.
Sampled frames inspected and video automatically opened on the Mac. The failed
hover remains visible; this is not a successful three-task controller release.
