# mjbatch and rapid adaptation after body changes

Reviewed 2026-09-10 UTC. This direction supersedes the proposed SWAP architecture. The completed work is a dependency/source audit and a bounded benchmark of an upstream example, not a three-legged policy or a new simulation goal.

## User direction

> SWAP doesnt seem mature enough, and im not convinced the simmetry is a good thing... what if we want the dog to learn to walk with three legsm after it lost one... i want something more generic, and I want something it can learn in a few minutes... check https://github.com/kevinzakka/mjbatch it says it can train fast

## What the repository provides

[mjbatch](https://github.com/kevinzakka/mjbatch) is a CPU MuJoCo batching library with a C++ thread pool and per-simulation parameter arrays. Its README claims Go1 walking in under a minute on an M1 laptop. The inspected revision is `77966f85bcd8f7ef4351cb4a1a6f42e133d19725`; the repository was created on September 10 UTC, with one commit and package version 0.1.0 at inspection. It is new software wrapping the established MuJoCo engine.

The [Go1 example](https://github.com/kevinzakka/mjbatch/blob/77966f85bcd8f7ef4351cb4a1a6f42e133d19725/examples/go1_joystick.py) uses PPO, a 2 Hz trot-phase reward, and mirrored-action averaging. These are example choices, not engine requirements. Our compiled model audit found foot/floor collisions only, with all twelve actuator force limits active. The example is a narrow training benchmark, not evidence of damage adaptation or general parkour.

## Measured on this Mac

An isolated checkout built successfully using uv 0.12.12, Python 3.14.7, MuJoCo 3.13.0, PyTorch 2.14.0, NumPy 2.5.3 and Menagerie 2026.9.0. Changes were limited to the MuJoCo/PyTorch/Python dependency pins and removal of the old CUDA 12.8 package index. No C++ changes were necessary. All **38 upstream tests passed** in 9.47 seconds. The main demo environment and lockfile were untouched.

On the Apple M3 Max with 16 CPU cores and 128 GiB memory:

| Measurement | Result |
| --- | --- |
| Initialization | Random weights; supplied checkpoint was not loaded |
| Simulation and actor | CPU |
| Learner | Apple GPU through PyTorch MPS |
| Training loop wall time | 60.0123 seconds, excluding setup and evaluation |
| PPO iterations | 294 of the example's default 600 |
| Control transitions | 7,225,344 across 1,024 environments |
| Physics substeps | Five per control transition |
| Ten-second evaluation trials | 16 predetermined cases, four command types |
| Stayed upright throughout | 12/16 initially, 16/16 after training |
| Mean forward-velocity RMSE | 0.7200 to 0.1532 m/s across four complete forward trials |

Both forward comparisons use the full ten seconds and a 0.7 m/s command; all four forward cases survived in both evaluations. Other per-survival errors in the report are conditional and must not be treated as complete-run scores. The initial network matches the learner initialization actually copied to the rollout actor. Upright means body-up z above 0.5, trunk height above 0.12 m, and finite state throughout; it is not a parkour success criterion.

This is one training seed and a small evaluation, with the example's task assumptions preserved. Rendered trained-policy poses were inspected during a separate six-second forward run. See [benchmark report](MJBATCH_BENCHMARK.json); raw progress, checkpoint and images are local ignored artifacts under `outputs/research/mjbatch/`. The isolated audit checkout is temporary; it is not an installed project dependency.

## Proposed next experiment

Use mjbatch as the simulator execution layer with an unconstrained policy and causal proprioceptive history. Start with task progress, balance and physical effort objectives; omit fixed gait timing and left/right action tying. Preserve full meaningful collision geometry and torque limits, and benchmark their cost. A small policy is the initial candidate for short iteration cycles; architecture and optimizer remain replaceable.

Measure three distinct cases: a normal policy used without updates after damage; that checkpoint fine-tuned on the changed body; and fresh training on the changed body. Save results at one, three and five minutes of training. These are proposed budgets, not predicted convergence times. Pretrained immediate robustness and actual weight-changing adaptation must be reported separately.

Represent a missing leg with an explicit three-leg model, removing its mass, joints and collision shapes. Keep a stable semantic observation/action interface with documented padding and availability masks. Batch model variants separately where their topologies differ. A motor-disabled leg is a separate failure case: it remains massive and can still provide passive contact. A locked or folded leg is another distinct condition. Do not label zero motor torque as amputation.

Begin with flat ground, then low ramps and steps once adaptation is measurable. The first video should show normal walking, the changed-body failure, short training checkpoints, and recovered progress on a held-out simple course. Use a labeled transition between four- and three-leg models rather than silently changing morphology inside a trajectory. Dynamic detachment would require its own modeled mechanism.

Conditioned symmetry is not inherently incompatible with damage if the damage state is transformed too, but no such architectural constraint is needed for this experiment. Genericity comes from the environment/model interface and evaluation across body conditions; it should not be claimed from a single successful tripod gait.
