"""Native MuJoCo viewer for the end-to-end cycle: watch it compute live, or replay a recorded run.

live    runs the cycle (same runner, same checks) and refreshes the window every 0.25 ms of simulated
        time. Physics costs about 600 s per simulated second, so this is extreme slow motion.
replay  plays a run's recorded states (1 ms apart). Space pauses, Right/Left step 1 ms,
        Up/Down double or halve the speed. Replay of recorded states, not simulation.
Mouse orbits, pans and zooms; the camera starts on the eyelet.

  uv run --locked python -m sixlegs.neural_insertion.e2e_view live --output DIR [--tubes] [--stop-after lift]
  uv run --locked python -m sixlegs.neural_insertion.e2e_view replay DIR [--speed 0.05]
"""

import argparse
import json
import os
from pathlib import Path
from queue import SimpleQueue
import sys
import sysconfig
import time

import mujoco
import numpy as np


def relaunch_under_mjpython(module="sixlegs.neural_insertion.e2e_view"):
    """macOS passive viewer needs mjpython; same launcher as the project's other viewers."""
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        env = os.environ.copy()
        paths = [sysconfig.get_config_var("LIBDIR"), str(Path(sys.base_prefix)/"lib")]
        paths += env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib").split(":")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
        os.execve(sys.executable, [sys.executable, str(Path(sys.executable).parent/"mjpython"),
                                  "-m", module, *sys.argv[1:]], env)


def focus(viewer, eyelet):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = eyelet
    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.0, 215, -22


def live(args):
    import mujoco.viewer
    from . import e2e
    if args.tube:
        from .tube_cycle import TubeCycle
        cycle = TubeCycle()
    elif args.modern:
        from .modern import ModernCycle
        cycle = ModernCycle(args.target, fast=args.fast)
    else:
        cycle = e2e.Cycle(args.target, tubes=args.tubes)
    if args.resume:
        cycle.restore(args.resume)
    with mujoco.viewer.launch_passive(cycle.m, cycle.d, show_left_ui=False, show_right_ui=False) as viewer:
        focus(viewer, cycle.eyelet)
        follow = [True]

        def show(name):
            if not viewer.is_running():
                raise KeyboardInterrupt
            gap = float(np.linalg.norm(cycle.eyelet-cycle.tip))
            viewer.set_texts((None, None, f"LIVE | {name} | t = {cycle.d.time:.4f} s simulated | "
                              f"loop to needle tip {gap*1e3:.0f} um | step {cycle.m.opt.timestep*1e6:.0f} us", None))
            if follow[0]:
                viewer.cam.lookat[:] = cycle.eyelet
            viewer.sync()

        cycle.on_step = show
        status = "completed"
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            cycle.run(checkpoints=args.output/"checkpoints", stop_after=args.stop_after)
            if args.stop_after:
                status = f"stopped after {args.stop_after} (requested)"
        except (RuntimeError, FloatingPointError) as error:
            status = f"stopped: {error}"
        except KeyboardInterrupt:
            status = "viewer closed"
        e2e.save(cycle, args.output, status, started)
        print(status, flush=True)
        while viewer.is_running():  # keep the last state on screen until closed
            show(f"done: {status}")
            time.sleep(.05)


def replay(args):
    import mujoco.viewer
    from .e2e_scene import load_e2e
    run = Path(args.run)
    report = json.loads((run/"report.json").read_text())
    trace = dict(np.load(run/"trace.npz", allow_pickle=True))
    if (run/"scene.xml").exists():  # the run's own scene: exact replay
        from .der import plugin
        plugin()
        m = mujoco.MjModel.from_xml_path(str(run/"scene.xml"))
        d = mujoco.MjData(m)
    else:
        m, d, _ = load_e2e(tubes=report.get("variant") == "tubes")
    times, qpos, phases = trace["time"], trace["qpos"], trace["phase"]
    keys = SimpleQueue()
    with mujoco.viewer.launch_passive(m, d, key_callback=keys.put, show_left_ui=False, show_right_ui=False) as viewer:
        d.qpos[:] = qpos[0]
        mujoco.mj_forward(m, d)
        focus(viewer, trace["eyelet"][0])
        t, speed, playing, last = times[0], args.speed, True, time.monotonic()
        while viewer.is_running():
            while not keys.empty():
                k = keys.get()
                if k == 32:
                    playing = not playing
                elif k == 262:
                    t, playing = t+1e-3, False
                elif k == 263:
                    t, playing = t-1e-3, False
                elif k == 265:
                    speed *= 2
                elif k == 264:
                    speed /= 2
            now = time.monotonic()
            if playing:
                t += (now-last)*speed
            last = now
            t = float(np.clip(t, times[0], times[-1]))
            i = min(int(np.searchsorted(times, t)), len(times)-1)
            d.qpos[:] = qpos[i]  # replay of a recorded state
            mujoco.mj_forward(m, d)
            gap = float(np.linalg.norm(trace["eyelet"][i]-trace["tip"][i]))
            viewer.set_texts((None, None, f"REPLAY | {phases[i]} | t = {times[i]:.4f} s | playback {speed:g}x | "
                              f"eyelet to needle tip {gap*1e3:.0f} um | {'playing' if playing else 'paused'} "
                              f"(space, arrows)", None))
            viewer.sync()
            time.sleep(1/60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("live")
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--target", type=int, default=0)
    a.add_argument("--tubes", action="store_true")
    a.add_argument("--modern", action="store_true", help="design v3 cycle (cannula, latch)")
    a.add_argument("--fast", action="store_true", help="fast thread settings (with --modern)")
    a.add_argument("--tube", action="store_true", help="thread-tube design (tube_cycle.py)")
    a.add_argument("--stop-after")
    a.add_argument("--resume", type=Path)
    b = sub.add_parser("replay")
    b.add_argument("run", type=Path)
    b.add_argument("--speed", type=float, default=.05)
    args = parser.parse_args()
    relaunch_under_mjpython()
    live(args) if args.mode == "live" else replay(args)


if __name__ == "__main__":
    main()
