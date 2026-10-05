"""Native static inspection of the experimental DER fixture."""

import argparse
import os
from pathlib import Path
import sys
import sysconfig
import time

import mujoco

from .der import load_der


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static", action="store_true", help="Frozen inspection; this viewer never advances physics")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.seconds is not None and args.seconds <= 0:
        parser.error("--seconds must be positive")
    # Same native macOS launch pattern as neural_insertion.cli.
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        env = os.environ.copy()
        paths = [sysconfig.get_config_var("LIBDIR"), str(Path(sys.base_prefix)/"lib")]
        paths += env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib").split(":")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
        os.execve(sys.executable, [sys.executable, str(Path(sys.executable).parent/"mjpython"),
                                  "-m", "sixlegs.neural_insertion.der_view", *sys.argv[1:]], env)
    import mujoco.viewer

    model, data = load_der()
    model.vis.quality.shadowsize = 0
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("cable").id
        viewer.opt.sitegroup[:] = 0
        start = time.monotonic()
        while viewer.is_running() and (args.seconds is None or time.monotonic()-start < args.seconds):
            viewer.set_texts((None, None, "DER / direct force projection / static inspection",
                             "44 mm length | 40 micrometre diameter | t = 0 | acceptance under evaluation"))
            viewer.sync()
            time.sleep(.01)


if __name__ == "__main__":
    main()
