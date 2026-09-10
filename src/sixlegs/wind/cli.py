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
from .profiles import PROFILES
from .scene import preview
from .simulation import Simulation, run
from .weather import Weather, prepare


def ensure_weather(args, seed=None):
    seed = args.seed if seed is None else seed
    folder = experiment_dir(args) / f"weather-{seed}"
    duration = max(46.0, args.profile.wind_scale * (args.duration + 3))
    horizon = 2.5 * args.profile.wind_scale
    valid = False
    if (folder / "weather.json").exists():
        meta = json.loads((folder / "weather.json").read_text())
        valid = (
            meta["duration"] >= duration
            and meta.get("forecast_horizon_s", 2.5) >= horizon
            and all(
                meta["models"][k].get("sha256")
                == hashlib.sha256((args.models / f"{k}.pt").read_bytes()).hexdigest()
                for k in ("fno", "pino")
            )
        )
    if not valid:
        prepare(
            seed,
            args.models,
            folder,
            backend=args.device,
            duration=duration,
            horizon=horizon,
        )
    return folder


def experiment_dir(args):
    return (
        ROOT
        / "outputs/wind"
        / ("" if args.profile.name == "standard" else args.profile.name)
    )


def compare(args):
    result = []
    folder = args.output or experiment_dir(args) / "comparison"
    folder.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed, args.seed + args.seeds):
        weather = ensure_weather(args, seed)
        for kind in ("persistence", "fno", "pino", "oracle"):
            report = run(
                Weather(weather, kind, args.profile.wind_scale),
                folder / f"{seed}-{kind}",
                args.duration,
                args.profile,
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
    weather = Weather(folder, args.kind, args.profile.wind_scale)
    sim = Simulation(weather, args.profile)
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
                cam = follow(sim.data, profile=args.profile)
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
    parser.add_argument("--profile", choices=tuple(PROFILES), default="standard")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument(
        "--kind", choices=("persistence", "fno", "pino", "oracle"), default="pino"
    )
    parser.add_argument("--models", type=Path, default=ROOT / "assets/wind")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    args.profile = PROFILES[args.profile]
    args.duration = args.profile.duration if args.duration is None else args.duration
    args.seed = (
        (300 if args.profile.name == "standard" else 400)
        if args.seed is None
        else args.seed
    )
    args.run_dir = args.run_dir or experiment_dir(args) / "comparison"
    preview_dir = (
        ROOT
        / "previews/wind"
        / ("" if args.profile.name == "standard" else args.profile.name)
    )
    if args.duration <= 0 or args.speed <= 0 or args.seeds < 1:
        parser.error("Duration, speed and seed count must be positive")
    if args.command == "preview":
        print(preview(args.output or preview_dir))
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
                args.output or experiment_dir(args) / f"weather-{args.seed}",
                backend=args.device,
                duration=max(46.0, args.profile.wind_scale * (args.duration + 3)),
                horizon=2.5 * args.profile.wind_scale,
            )
        )
    elif args.command == "run":
        result = run(
            Weather(ensure_weather(args), args.kind, args.profile.wind_scale),
            args.output or experiment_dir(args) / f"{args.seed}-{args.kind}",
            args.duration,
            args.profile,
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
                args.output or preview_dir / "comparison.mp4",
                args.seed,
                profile=args.profile,
            )
        )
    else:
        view(args)


if __name__ == "__main__":
    main()
