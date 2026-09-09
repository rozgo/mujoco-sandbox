"""Unarmed land-water-land attachment demonstration."""

import argparse
import json
import os
import sys
import sysconfig
import time
from pathlib import Path
from queue import SimpleQueue

from sixlegs.scene import ROOT

from .simulation import Simulation, run
from .visuals import follow, preview, record


def view(args):
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        env = os.environ.copy()
        paths = [sysconfig.get_config_var("LIBDIR"), str(Path(sys.base_prefix) / "lib")]
        paths += env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib").split(
            ":"
        )
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            dict.fromkeys(p for p in paths if p)
        )
        os.execve(
            sys.executable,
            [
                sys.executable,
                str(Path(sys.executable).parent / "mjpython"),
                "-m",
                "sixlegs.amphibious.cli",
                *sys.argv[1:],
            ],
            env,
        )
    import mujoco.viewer

    sim = Simulation()
    events = SimpleQueue()
    paused, selected, budget = args.static, 1, 0.0
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=events.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.opt.geomgroup[3] = 0
        viewer.opt.sitegroup[:] = 0
        start = last = time.monotonic()
        while viewer.is_running() and (
            args.seconds is None or time.monotonic() - start < args.seconds
        ):
            now = time.monotonic()
            elapsed, last = min(0.1, now - last), now
            while not events.empty():
                key = events.get()
                if key == 32:
                    paused = not paused
                elif 49 <= key <= 52:
                    selected = key - 48
            if not paused and (
                sim.done_at is None or sim.data.time < sim.done_at + 3.0
            ):
                budget += elapsed * args.speed
                while budget >= sim.model.opt.timestep:
                    sim.step(False)
                    budget -= sim.model.opt.timestep
            else:
                budget = 0.0
            if selected in (1, 2):
                cam = follow(sim.data, selected == 2)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = cam.lookat
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                    cam.distance,
                    cam.azimuth,
                    cam.elevation,
                )
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = sim.model.camera(
                    "overview" if selected == 3 else "front"
                ).id
            viewer.set_texts(
                (
                    None,
                    None,
                    f"AMPHIBIOUS | {sim.control.phase} | {sim.data.time:.1f}s | {args.speed:g}x\nBuoyancy {sum(sim.water.buoyancy):.0f} N / Weight {sim.weight:.0f} N",
                    "1 Follow | 2 Side | 3 Overview | 4 Front | Space Pause / Resume",
                )
            )
            viewer.sync()
            time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("view", "preview", "run", "record"),
        default="view",
        nargs="?",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/amphibious/crossing"
    )
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--duration", type=float, default=650.0)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--static", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0 or args.duration <= 0:
        parser.error("Speed and duration must be positive")
    if args.command == "view":
        view(args)
    elif args.command == "preview":
        preview(args.output or ROOT / "previews/amphibious")
    elif args.command == "run":
        result = run(args.output or args.run_dir, args.duration)
        print(json.dumps(result, indent=2))
        if not result["success"]:
            raise SystemExit(1)
    else:
        if not (args.run_dir / "trajectory.npz").exists():
            run(args.run_dir, args.duration)
        print(
            json.dumps(
                record(
                    args.run_dir,
                    args.output or ROOT / "previews/amphibious/crossing.mp4",
                    args.speed,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
