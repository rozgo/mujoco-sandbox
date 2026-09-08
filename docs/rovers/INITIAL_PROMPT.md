# Rover goal request

Public task brief:

Build a separate scene with RC-like rovers communicating over simulated LoRa, using MuJoCo for physical vehicle dynamics.

User authorization:

> Lets set this as new goal. Implement now

The accepted implementation goal is a separate six-rover inspection-yard scene alongside the existing hexapod: physical wheel motion, independent local observations and beliefs, simulated LoRa delivery, evidence that a delivered report changes another rover's behavior, healthy/degraded/disabled comparisons, camera/network/belief visualization and a complete recording. The real Reticulum path is an optional local-stack integration over the modeled radio channel.

Existing project requirements carry forward: uv-managed Python, practical macOS support, Git repository, Git LFS for large binaries, sensible documented physical/actuator limits, testing during development, saved prompts and elapsed-time tracking.

Goal start: **2026-09-07 21:57:21 PDT** / **2026-09-08 04:57:21 UTC**, from the clock reading when this goal was created. The earlier hexapod prompt remains in `docs/INITIAL_PROMPT.md`.
