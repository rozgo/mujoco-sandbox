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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preview", "view", "inspect"), nargs="?", default="view")
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
    else:
        from .validation import inspect_scene
        report = inspect_scene()
        print(json.dumps(report, indent=2))
        if not report["success"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
