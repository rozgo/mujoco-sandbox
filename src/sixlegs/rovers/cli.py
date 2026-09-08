"""RC rover inspection with simulated LoRa and independent local beliefs."""

import argparse
import json
import os
import sys
import sysconfig
import time
from pathlib import Path
from queue import SimpleQueue

from sixlegs.scene import ROOT

from .simulation import RoverSimulation, run
from .visuals import add_links, preview, record


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
                "sixlegs.rovers.cli",
                *sys.argv[1:],
            ],
            env,
        )
    import mujoco.viewer

    sim = RoverSimulation(args.case, args.seed, args.duration, args.reticulum)
    m, d = sim.model, sim.data
    events = SimpleQueue()
    paused = args.static
    selected = 1
    budget = 0.0
    try:
        with mujoco.viewer.launch_passive(
            m, d, key_callback=events.put, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = m.camera("overview").id
            viewer.opt.sitegroup[:] = 0
            start = last = time.monotonic()
            while viewer.is_running() and (
                args.seconds is None or time.monotonic() - start < args.seconds
            ):
                now = time.monotonic()
                elapsed = min(0.1, now - last)
                last = now
                while not events.empty():
                    key = events.get()
                    if key == 32:
                        paused = not paused
                    elif 49 <= key <= 54:
                        selected = key - 49
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = m.camera(f"r{selected}_front").id
                    elif key in (48, 79, 84):
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = m.camera(
                            "overhead" if key == 84 else "overview"
                        ).id
                    elif key == 70:
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                        viewer.cam.trackbodyid = m.body(f"r{selected}").id
                        viewer.cam.distance = 2.5
                        viewer.cam.elevation = -25
                        viewer.cam.azimuth = 135
                if not paused and d.time < args.duration:
                    budget += elapsed * args.speed
                    while budget >= m.opt.timestep and d.time < args.duration:
                        sim.step(False)
                        budget -= m.opt.timestep
                else:
                    budget = 0.0
                viewer.user_scn.ngeom = 0
                add_links(viewer.user_scn, sim.positions(), sim.network.events, d.time)
                a = sim.agents[selected]
                status = (
                    "PAUSED"
                    if paused
                    else ("FINISHED" if d.time >= args.duration else "RUNNING")
                )
                viewer.set_texts(
                    (
                        None,
                        None,
                        f"ROVER COMMS | {args.case} | {d.time:.1f}s | {status}\nInspected {len(sim.inspected)}/8 | R{selected} knows {len(a.belief)}/8, completed {len(a.completed)}/8",
                        "1-6 Front cameras | 0/O Overview | T Top | F Follow\nSpace Pause / Resume | green: delivered, red: lost",
                    )
                )
                viewer.sync()
                time.sleep(0.005)
    finally:
        if sim.bridge:
            sim.bridge.close()


def compare(args):
    folder = args.output or ROOT / "outputs/rovers/comparison"
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    reports = []
    for seed in range(args.seeds):
        for case in ("healthy", "degraded", "disabled"):
            reports.append(
                run(
                    case,
                    seed,
                    args.duration,
                    folder / f"{case}-{seed}",
                    record=args.save_runs,
                    reticulum=args.reticulum,
                )
            )
    result = {"duration_s": args.duration, "radio_seeds": args.seeds, "runs": reports}
    (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "case",
                        "seed",
                        "unique_inspections",
                        "completion_time_s",
                        "duplicate_inspections",
                        "team_completion_knowledge_fraction",
                    )
                }
                for r in reports
            ],
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("view", "preview", "run", "compare", "record"),
        nargs="?",
        default="view",
    )
    parser.add_argument(
        "--case", choices=("healthy", "degraded", "disabled"), default="healthy"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--duration", type=float, default=150)
    parser.add_argument("--seconds", type=float, help="Viewer smoke-test wall duration")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--reticulum", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--save-runs", action="store_true")
    args = parser.parse_args()
    if args.speed <= 0 or args.duration <= 0 or args.seeds < 1:
        parser.error("Speed, duration and seeds must be positive")
    if args.command == "view":
        view(args)
    elif args.command == "preview":
        preview(args.output or ROOT / "previews/rovers")
    elif args.command == "run":
        result = run(
            args.case,
            args.seed,
            args.duration,
            args.output or ROOT / f"outputs/rovers/{args.case}",
            reticulum=args.reticulum,
        )
        print(json.dumps(result, indent=2))
    elif args.command == "compare":
        compare(args)
    elif args.command == "record":
        directory = args.run_dir or ROOT / f"outputs/rovers/{args.case}"
        if not (directory / "trajectory.npz").exists():
            run(
                args.case, args.seed, args.duration, directory, reticulum=args.reticulum
            )
        record(
            directory,
            args.output or ROOT / f"previews/rovers/{args.case}.mp4",
            args.speed,
        )


if __name__ == "__main__":
    main()
