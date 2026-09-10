# Adaptive locomotion across partial damage

Implementation started September 10, 2026 at 06:53:59 UTC on `experiment/adaptive-dog`. This brief supersedes the earlier leg-count framing and the SWAP-specific architecture proposal.

## Implementation authorization and hard training constraint

> lets implement this into a branch to make sure we get traction before polutin main branch... set as goal and lets try to learn

> we need to prioritize fast RL... anything longer than a few minutes is not acceptable

> fast rl through batching, or GPU, or just from vectorized environment or gym... anything, but if training takes hours then we cant continue this path

Training speed is an acceptance gate. Interpret a few minutes as a maximum five-minute cumulative training budget per final policy, including teacher/pretraining, estimator training, fine-tuning and resumed ancestry. Compare independent bounded experiments and CPU/GPU backends; do not conceal longer training in a chain of short runs. Record setup, compilation and evaluation separately and also report full time-to-result. If the learning gate fails, report it and reassess the approach rather than extending a policy to hours. Do not merge experimental work into main.

## User direction

> what about partial leg? you see the point? im trying to make a general controller that learns to adapt no matter the state of the dogs leg

> ok, this looks like the right direction, expand the plan and lets see if we have gaps and how we would implement this

## Objective

Train one controller to infer changes in its body from recent interaction and adapt its locomotion to complete unfamiliar obstacle courses. Include partial legs, reduced joint travel, weakened or intermittent actuation, missing components, and combinations of these changes. Generalization to untrained conditions is the claim to test; counting working legs is insufficient.

Use physically stepped MuJoCo, sensible documented inertias and actuator limits, current supported dependencies managed by uv, macOS viewing, optional RTX 4090 training, and Git/LFS synchronization. mjbatch is the first execution backend to investigate because the upstream walking example has already been benchmarked locally.

The intended deliverable after implementation is a reproducible environment, trainable and comparable policies, retained success/failure reports, and a complete video with synchronized overview, following/detail and robot-mounted views. Build and show static previews of the course and damaged bodies before adding movement.

## Scope and evidence

The broad goal remains adaptation across body conditions. Initial implementation should include partial geometry from the outset, then expand difficulty according to measured performance. It must not quietly become motor disabling alone or a library of separately selected gaits.

Immediate history-conditioned adaptation and additional training that changes weights are different experiments. Neither universal mobility under arbitrary destruction nor complete generalist training in one minute is established. Physical feasibility and the coverage of evaluation define the demonstrated scope.

See [the implementation plan](ADAPTIVE_LOCOMOTION_PLAN.md) for architecture, missing pieces, staged acceptance criteria and proposed code layout; [the mjbatch audit](MJBATCH_REVIEW.md) for the experiment already completed.
