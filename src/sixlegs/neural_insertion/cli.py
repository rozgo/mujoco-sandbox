"""Static inspection of the surgical insertion robot."""

import argparse
import json
import os
from pathlib import Path
from queue import SimpleQueue
import sys
import sysconfig
import time

import mujoco

from .scene import CAMERAS, ROOT, load_scene
from .visuals import preview


def view(args):
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        env = os.environ.copy()
        paths = [sysconfig.get_config_var("LIBDIR"), str(Path(sys.base_prefix) / "lib")]
        paths += env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib").split(":")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
        os.execve(sys.executable, [sys.executable, str(Path(sys.executable).parent / "mjpython"),
                                  "-m", "sixlegs.neural_insertion.cli", *sys.argv[1:]], env)
    import mujoco.viewer

    model, data = load_scene()
    events = SimpleQueue()
    selected = CAMERAS.index(args.camera)
    with mujoco.viewer.launch_passive(model, data, key_callback=events.put,
                                     show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.sitegroup[:] = 0
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera(CAMERAS[selected]).id
        start = time.monotonic()
        if args.task == "tour":
            # Same stepping/servo loop as headless runs; the callback only syncs the view.
            from .motion import run, start_ready, tour_reference
            start_ready(model, data)
            reference = tour_reference(model, data)
            viewer.set_texts((None, None, "SURGICAL INSERTION | Six-target approach | programmed servo | 1.0x", None))

            def pace(state):
                if round(state.time*1000) % 16 == 0:
                    viewer.sync()
                    lag = state.time-(time.monotonic()-start)
                    if lag > 0:
                        time.sleep(lag)
                if not viewer.is_running() or (args.seconds is not None and time.monotonic()-start > args.seconds):
                    raise KeyboardInterrupt

            try:
                run(model, data, reference, record_every=1000, callback=pace)
            except KeyboardInterrupt:
                pass
        while viewer.is_running() and (args.seconds is None or time.monotonic() - start < args.seconds):
            while not events.empty():
                key = events.get()
                if 49 <= key <= 54:
                    selected = key - 49
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = model.camera(CAMERAS[selected]).id
                elif key in (70, 102):
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                    viewer.cam.lookat[:] = (0, -.01, .24)
                    viewer.cam.distance = 1.15
                    viewer.cam.azimuth = 125
                    viewer.cam.elevation = -25
            viewer.set_texts((None, None, "SURGICAL INSERTION | Static design review | v1",
                             "1 Overview | 2 Mechanism | 3 Field | 4 Tool | 5 Microscope | 6 Cassette | F Free camera"))
            viewer.sync()
            time.sleep(.01)


def approach(args):
    import hashlib
    import numpy as np
    from .motion import approach_suite
    started = time.perf_counter()
    report, trace, reference = approach_suite()
    out = args.run_output
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "tour.npz", **trace)
    report.update(scene_sha256=hashlib.sha256((ROOT / "build/neural_insertion/static_v1.xml").read_bytes()).hexdigest(),
                  trajectory_sha256=hashlib.sha256((out / "tour.npz").read_bytes()).hexdigest(),
                  mujoco=mujoco.__version__, controller="programmed servo (PD + feedforward + bounded integral)",
                  run_wall_s=time.perf_counter() - started)
    if args.command == "record":
        from .motion_media import record
        report["video"] = record(report, trace, reference, args.output)
    (out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "hold_tip_drift_m": report["hold"]["tip_drift_m"],
                      "tour": {k: report["tour"][k] for k in ("tracking_rms_while_moving_m", "robot_environment_contacts",
                                                              "min_needle_phantom_gap_m")},
                      "video": report.get("video", {}).get("video")}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "view", "inspect", "approach", "record"), nargs="?", default="view")
    parser.add_argument("--task", choices=("tour",), help="Run the measured six-target approach in the viewer.")
    parser.add_argument("--run-output", type=Path, default=ROOT / "outputs/neural_insertion/approach/latest",
                        help="Telemetry directory for approach/record.")
    parser.add_argument("--output", type=Path, default=ROOT / "previews/neural_insertion/static_v1")
    parser.add_argument("--static", action="store_true", help="Static inspection (the only mode in v1).")
    parser.add_argument("--camera", choices=CAMERAS, default="overview")
    parser.add_argument("--seconds", type=float, help="Close the viewer after this many wall-clock seconds.")
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    if args.command == "view":
        view(args)
    elif args.command == "preview":
        print(json.dumps(preview(args.output), indent=2))
    elif args.command in ("approach", "record"):
        approach(args)
    else:
        from .validation import inspect_scene
        report = inspect_scene()
        print(json.dumps(report, indent=2))
        if not report["success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
