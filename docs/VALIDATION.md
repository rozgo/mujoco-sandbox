# Scene checkpoint validation

Executed on Apple Silicon macOS, Python 3.12.12, MuJoCo 3.12.0, using the committed uv lockfile.

- `uv run pytest -q`: **5 passed**. Covers upstream asset hashes, model structure/independent grippers, six initial foot contacts with no other contacts, 150 sampled leg configurations, and 2,500 physics steps (5 simulated seconds) checking finite state, actuator bounds, absence of warnings, base displacement below 3 cm, settled speed below 0.02 and both targets remaining on the source table.
- `uv run sixlegs render`: **passed**. Six 1600 × 1000 camera PNGs and a 1600 × 1095 contact sheet generated and visually inspected.
- `uv run sixlegs inspect`: **passed**. 97 position coordinates, 90 velocity coordinates, 34 actuators, six cameras, approximately 58.831 kg robot mass. Initial contacts are the six feet at floor level.
- `uv run sixlegs view --seconds 2`: **passed**, opened and closed the native passive viewer without stepping physics.

## Issue fixed during testing

Direct `uv run mjpython ...` initially failed to find `@rpath/libpython3.12.dylib`. The project's `view` command now adds the running interpreter's library directory to its child process's `DYLD_FALLBACK_LIBRARY_PATH`, then starts the official `mjpython` launcher. This passed the native GUI check without modifying the virtual environment, system libraries or global shell configuration.

## Scope

This validates the static scene and a bounded-duration stationary physics hold. Full-range collision freedom, locomotion, grasping, carried-object stability, route execution and release success are not yet tested or implemented. Movement is deliberately deferred until the user has reviewed the scene.
