# Surgical robot project brief

Requested October 5, 2026. This first deliverable is research and a setup proposal.

Find the best surgical robot foundation for MuJoCo, assess how complex the robot
and its tasks can become, and plan reinforcement learning with PufferLib 5.0.
Report the proposed experience, implementation requirements, and practical limits.

## Working assumptions

- Start with simulation of surgical instrument skills on a training fixture.
- Prioritize a recognizable surgical mechanism and reproducible physical behavior.
- Use PufferLib **5.0**, with its current native environment interface.
- Develop and inspect on macOS; plan training on Linux with an NVIDIA GPU.
- Robot acquisition, hardware deployment, and clinical validation are outside this
  initial simulation proposal.
- Research does not start training or imply that a model has been compiled or
  dynamically validated. Estimates in the proposal are planning estimates.

## Proposed experience

Two dVRK Si patient-side manipulators with needle drivers and a stereo endoscope.
Begin with needle pickup, orientation, handoff, ring passage, and placement.
Extend to thread handling and closure of a compliant pad through existing eyelets;
investigate puncture and knot mechanics separately before promising real suturing.

Required views: fixed overview, close tool/needle view, stereo endoscopic cameras,
and observer detail cameras near each instrument. Initial learning uses state
observations; displayed camera feeds are observer output until a vision policy is
explicitly implemented and tested. Final motion videos use 1x playback.

## Completion criteria

For this research stage: identify and cite the preferred asset source, compare
alternatives, verify the PufferLib version/interface, distinguish available physics
from new research, and give an actionable setup and staged validation plan.

For the proposed first implementation: compile and review the static scene before
motion; validate linkages, support, clearance, actuator limits, and contact grasps;
then complete one reference-controller mission before RL. Proposed learned-task
acceptance is at least 90/100 successful held-out full missions per training seed,
evaluated over three predetermined training seeds, with all failures retained.
Numerical stability and prohibited-contact checks are separate mandatory gates.
Detailed proposed tolerances appear in [the project proposal](DVRK_PROPOSAL.md).
