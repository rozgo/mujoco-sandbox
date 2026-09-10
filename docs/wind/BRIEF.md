# Wind-delivery demo brief

Goal started **2026-09-09 17:28:08 PDT / 2026-09-10 00:28:08 UTC**.

The user supplied a neural-operator research brief and asked:

> Read this and let me know what we can build in sim, to make an impressive video.

The accepted proposal was a MuJoCo quadrotor carrying a suspended parcel through changing crosswinds, with a side-by-side comparison of frozen-wind and learned-wind predictive control. MuJoCo retains rigid-body, cable, actuation and contact dynamics; a neural operator forecasts an external fluid field.

The user then instructed:

> Perfect, make this the new goal. Implement fully.

During implementation the user offered an RTX 4090 machine through an existing RustDesk session. After verifying the remote host, training moved to that GPU; macOS remained the flight evaluation and rendering machine.

## Deliverables

- Separate scene, preserving the existing hexapod, rover and amphibious demos.
- Static preview before motion; primitive quadrotor, suspended package, pickup/drop platforms and two gates.
- Physical takeoff, transport, lowering, contact-confirmed release and departure.
- Independent numerical Navier–Stokes wind used as the common evaluation environment.
- Actually trained FNO and PINO predictors, measured on held-out trajectories and different grid resolutions.
- Identical predictive controller and actuator limits across comparisons; only the forecast changes.
- Complete video with synchronized physical flights, cameras, wind fields and measured results.
- Tests, documented assumptions and specifications, uv/macOS support, Git/LFS and elapsed-time record.

The research brief is inspiration, not evidence of a particular published result. This demo does not reproduce a paper's numerical benchmark and does not replace MuJoCo with an FNO. See [README](README.md) for the implemented equations and limitations. The repository's original user prompt is preserved separately in [INITIAL_PROMPT.md](../INITIAL_PROMPT.md).
