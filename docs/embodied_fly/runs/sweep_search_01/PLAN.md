# Sweep search 01 — declared before execution

Parent: `state_hover_retention_02` (`f5d383485f37478b02b74cf9502c603f35b24e2f08b35e0ca6035b76b0bd87c6`). Preserve it regardless of this result.

Search a common gain and bias on the two existing wing-sweep decoder logits. Grid: gain [0.85, 1.0, 1.15, 1.3] × bias [-0.02, 0, 0.02, 0.04]. The unmodified parent (1,0) is included. All other actor state and all physical parameters remain fixed. There are no labels, runtime task masks, aerodynamic changes or direct body-force commands.

Use 32 worlds, 16 CPU MuJoCo threads, RTX neural inference, 5 kHz physics/500 Hz actions. Two batches of eight candidates; four cases per candidate: stand, walk, two independently initialized airborne starts. Repeat the same four initial states for every candidate. Search seed 96001; each case runs two complete physical seconds without resetting failures. Retain all states, actions, observations, wrenches and metrics.

Select among candidates retaining ground stability and permitted support. Ground wing, speed and yaw RMS must stay below 1.25× parent plus tolerances (0.005 rad wings, 0.05 cm/s speed, 0.05 rad/s yaw). Peak wing error may grow by no more than 0.05 rad. Hover cost averages root RMS in cm plus twice failed-frame fraction across both starts. Require >=5% improvement to export changed weights; otherwise retain exact parent state. These are development selection criteria, not relaxed acceptance gates.

Verify candidate batched arithmetic agrees with an ordinary exported actor, all unselected state is unchanged, live inputs are causal, and the canonical physical fingerprint matches. A selected changed actor then receives the existing full five-second three-command evaluation on seed 72003, including neural view and all failures. No short search success establishes learned hover.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.sweep_search \
  --resume assets/embodied_fly/diagnostics/state_hover_retention_02.pt \
  --graph ../fly-survival/outputs/fly_survival/malecns \
  --output outputs/embodied_fly/sweep_search_01 --seconds 2 --seed 96001 --threads 16
```
