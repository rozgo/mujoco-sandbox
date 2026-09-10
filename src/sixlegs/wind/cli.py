"""Neural wind forecasting and physical drone delivery on macOS or CUDA."""

import argparse
import hashlib
import json
import os
import sys
import sysconfig
import time
from pathlib import Path
from queue import SimpleQueue

from sixlegs.scene import ROOT

from .flow import dataset
from .scene import preview
from .simulation import Simulation, run
from .weather import Weather, prepare


def ensure_weather(args, seed=None):
    seed = args.seed if seed is None else seed
    folder = ROOT / f"outputs/wind/weather-{seed}"
    valid = False
    if (folder / "weather.json").exists():
        meta = json.loads((folder / "weather.json").read_text())
        valid = all(
            meta["models"][k].get("sha256")
            == hashlib.sha256((args.models / f"{k}.pt").read_bytes()).hexdigest()
            for k in ("fno", "pino")
        )
    if not valid:
        prepare(seed, args.models, folder, backend=args.device)
    return folder


def compare(args):
    result = []
    folder = args.output or ROOT / "outputs/wind/comparison"
    folder.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed, args.seed + args.seeds):
        weather = ensure_weather(args, seed)
        for kind in ("persistence", "fno", "pino", "oracle"):
            report = run(
                Weather(weather, kind), folder / f"{seed}-{kind}", args.duration
            )
            report.update(seed=seed, forecast=kind)
            result.append(report)
            (folder / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    return result


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
                "sixlegs.wind.cli",
                *sys.argv[1:],
            ],
            env,
        )
    import mujoco.viewer

    from .visuals import add_flow, follow

    folder = ensure_weather(args)
    weather = Weather(folder, args.kind)
    sim = Simulation(weather)
    events = SimpleQueue()
    paused = args.static
    camera = 1
    budget = 0.0
    with mujoco.viewer.launch_passive(
        sim.model,
        sim.data,
        key_callback=events.put,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.opt.sitegroup[4] = 0
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
                elif 49 <= key <= 53:
                    camera = key - 48
            if not paused and sim.data.time < args.duration:
                budget += elapsed * args.speed
                while (
                    budget >= sim.model.opt.timestep and sim.data.time < args.duration
                ):
                    sim.step(False)
                    budget -= sim.model.opt.timestep
            else:
                budget = 0.0
            if camera == 1:
                cam = follow(sim.data)
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = cam.lookat
                viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = (
                    cam.distance,
                    cam.elevation,
                    cam.azimuth,
                )
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = sim.model.camera(
                    {2: "overview", 3: "front", 4: "loadcam", 5: "overhead"}[camera]
                ).id
            viewer.user_scn.ngeom = 0
            add_flow(viewer.user_scn, weather, sim.data.time)
            viewer.set_texts(
                (
                    None,
                    None,
                    f"NEURAL WIND | {args.kind.upper()} | {sim.control.phase} | {sim.data.time:.1f}s",
                    "1 Follow | 2 Overview | 3 Front | 4 Payload | 5 Top | Space Pause",
                )
            )
            viewer.sync()
            time.sleep(0.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "view",
            "preview",
            "dataset",
            "validate",
            "prepare",
            "run",
            "compare",
            "record",
        ),
        nargs="?",
        default="view",
    )
    parser.add_argument("--seed", type=int, default=300)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument(
        "--kind", choices=("persistence", "fno", "pino", "oracle"), default="pino"
    )
    parser.add_argument("--models", type=Path, default=ROOT / "assets/wind")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--duration", type=float, default=43.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--run-dir", type=Path, default=ROOT / "outputs/wind/comparison"
    )
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    if args.duration <= 0 or args.speed <= 0 or args.seeds < 1:
        parser.error("Duration, speed and seed count must be positive")
    if args.command == "preview":
        print(preview(args.output or ROOT / "previews/wind"))
    elif args.command == "dataset":
        print(dataset(args.output or ROOT / "outputs/wind/training.npz"))
    elif args.command == "validate":
        from .validation import validate

        print(
            json.dumps(
                validate(
                    args.models,
                    args.output or ROOT / "outputs/wind/validation",
                    args.device,
                ),
                indent=2,
            )
        )
    elif args.command == "prepare":
        print(
            prepare(
                args.seed,
                args.models,
                args.output or ROOT / f"outputs/wind/weather-{args.seed}",
                backend=args.device,
            )
        )
    elif args.command == "run":
        result = run(
            Weather(ensure_weather(args), args.kind),
            args.output or ROOT / f"outputs/wind/{args.seed}-{args.kind}",
            args.duration,
        )
        print(json.dumps(result, indent=2))
        if not result["success"]:
            raise SystemExit(1)
    elif args.command == "compare":
        if not all(r["success"] for r in compare(args)):
            raise SystemExit(1)
    elif args.command == "record":
        from .visuals import record

        print(
            record(
                args.run_dir,
                ensure_weather(args),
                args.output or ROOT / "previews/wind/comparison.mp4",
                args.seed,
            )
        )
    else:
        view(args)


if __name__ == "__main__":
    main()
