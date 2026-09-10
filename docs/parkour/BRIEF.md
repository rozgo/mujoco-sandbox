# SWAP-inspired quadruped parkour investigation

Status: research and proposed environment design, 2026-09-10 UTC. No parkour environment or policy has been implemented or trained.

## User direction

> SWAP for real quadruped parkor is absolutely what we want now. This is the type of innovation im looking for. Lets dig deeper here. This affects the environments we want to build.

Earlier constraints remain applicable: use current tools and modules, MuJoCo, uv, a usable macOS viewer, RTX 4090 training, Git/Git LFS synchronization, and documented physical behavior. The user is interested in frontier learning strategies and comparisons between trained policies. The amphibious demo's scripted gait is not a learned parkour policy.

## Proposed objective

Build a visual parkour research environment that measures how a symmetry-aware predictive representation changes learned locomotion. Show the policy seeing an obstacle, making contact, taking off or climbing, landing, and continuing. Evaluate frozen policies on predetermined unseen terrain, with failures retained.

The immediate first milestone should be a static preview of a configurable gap lane, platform-climbing lane, and mirrored terrain pair. Developing a new simulation and producing its final video requires a subsequent implementation phase; this investigation does not claim those deliverables exist.

See [environment proposal](ENVIRONMENT_PROPOSAL.md) for the specification, evidence and outstanding dependencies.
