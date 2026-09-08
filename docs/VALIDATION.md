# Validation results

Executed on Apple Silicon macOS, Python 3.12.12 and MuJoCo 3.12.0, using the committed uv lockfile.

## Full transfer

The shared production simulation completed the task in **112.912 simulated seconds**, approximately **23.5 wall seconds** headlessly on this machine. The live viewer runs at real time by default.

| Result | Mug, right gripper | Block, left gripper |
| --- | ---: | ---: |
| Final position (m) | (0.123907, 3.368824, 0.839990) | (0.140986, 3.807645, 0.869987) |
| Placement error | 6.2 mm | 16.5 mm |
| Destination-table support | Yes | Yes |
| Gripper released | Yes | Yes |
| Upright | Yes | Yes |
| Minimum body-origin height during carrying | 1.107 m | 1.138 m |

Both grippers established opposing-pad contact before lift. The robot passed west of the barrier at X = −2.7 m. The completed run had zero MuJoCo warnings and no robot/table/barrier penetrations above 1 mm, checked at every physics step. Peak normalized actuator torque was 1.0: a torque cap was reached but never exceeded. A separate inspection of the saved 20 Hz trajectory found no robot self-penetrations above 1 mm.

The final success check requires physical support, release, low velocity, upright orientation and proximity to the placement marks. See [TRANSFER_REPORT.json](TRANSFER_REPORT.json) for the complete measured report and phase times.

## Checks run

- `uv run pytest -q`: **7 passed** (about 25 seconds). Covers vendored asset hashes, model structure, independent grippers, initial contacts, sampled leg clearance, stationary physics hold, analytic leg IK against MuJoCo forward kinematics, and the complete task.
- The transfer test also confirms the controller never changes physical `qpos`, no external force arrays are used, the model has no mocap bodies or weld constraints, both objects clear the table during carrying, actuator caps hold and destination contacts/release are correct.
- `uv run sixlegs run`: **passed** with the production stepping loop and saved trajectory/report.
- `uv run sixlegs view --seconds 4 --speed 2 --camera left_wrist`: **passed** on the native macOS viewer, with physics advancing and wrist camera selected.
- `uv run sixlegs view --static --seconds 1`: **passed** for the preserved static review mode.
- Six static camera PNGs render successfully. Dynamic lift, barrier passage, release and final-placement images were rendered and visually inspected.
- Multi-camera MP4: **1280 × 1024, H.264, 20 fps, 56.45 seconds at 2× playback**, showing third-person, head and both wrist views. Frames are rendered from a recorded physical trajectory, not an independently animated sequence.

## Issues fixed during development

1. Direct `uv run mjpython ...` initially failed to find `@rpath/libpython3.12.dylib`. The launcher now adds the interpreter's actual library directory before starting the official macOS trampoline, without editing system libraries or global shell settings.
2. An initial open-loop foot schedule accumulated approximately half a meter of body-position error. Measured touchdown positions and a bounded body reference corrected this; the final controller reaches all route waypoints.
3. The first horizontal grasp orientation put the Gen3 wrist vision housing through the table edge. Rolling the tool frame keeps that housing above the table, preserving collision geometry and allowing both grasps.
4. The Kinova-specific gripper mounting was corrected during the static milestone to remove the redundant Robotiq mechanical coupling.

## Limits

This is a deterministic demonstration with fixed task coordinates and state feedback. Cameras produce actual scene images but are not used for visual object detection. The nominal task is validated; robustness to arbitrary object relocation, changed friction or masses, external pushes, and general route planning is not claimed. Joint limits and torque caps are simulation parameters, not hardware certification.


## Full-length all-view video

`previews/transfer_all_views.mp4` contains the complete validated trajectory at real time, plus a four-second final-state hold. All six views are synchronized: scene, following detail, overhead, head, left wrist and right wrist.

Verified with FFprobe: **1920 × 1080, H.264, 20 fps, 2,338 frames, 116.9 seconds, 29,239,955 bytes**. The entire file decoded without errors. Encoded frames from the lift and final placement were visually inspected; labels, camera panels and completion status are readable. Rendering uses the saved successful physical run, so no new physics behavior was introduced.

Reproduce with:

```sh
uv run sixlegs record --trajectory outputs/transfer.npz --layout all --speed 1 --output previews/transfer_all_views.mp4
```
