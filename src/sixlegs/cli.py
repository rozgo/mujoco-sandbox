"""Live physical transfer demo, static previews, and recordings."""
import argparse
import json
import os
import sys
import sysconfig
from queue import SimpleQueue
from pathlib import Path
import time

import mujoco
from PIL import Image, ImageDraw, ImageFont

from sixlegs.scene import ROOT, load_scene


def render(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, data = load_scene()
    option = mujoco.MjvOption()
    option.geomgroup[3] = 0  # Upstream collision meshes duplicate the visual meshes.
    option.sitegroup[:] = 0
    with mujoco.Renderer(model, height=1000, width=1600) as renderer:
        for name in ("third_person", "robot_detail", "overhead", "head", "left_wrist", "right_wrist"):
            renderer.update_scene(data, camera=name, scene_option=option)
            Image.fromarray(renderer.render()).save(output / f"{name}.png")
    canvas = Image.new("RGB", (1600, 1095), "#101820")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 23)
        title = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 38)
    except OSError:
        font = ImageFont.load_default(size=23)
        title = ImageFont.load_default(size=38)
    draw.text((32, 24), "SIXLEGS / SCENE PREVIEW", font=title, fill="#edf5f7")
    draw.text((32, 77), "Dual Kinova Gen3 + Robotiq 2F-85   /   Static design checkpoint", font=font, fill="#78c6c8")
    panels = [
        ("third_person", 0, 125, 800, 500, "01   SOURCE / BARRIER / DESTINATION"),
        ("robot_detail", 800, 125, 800, 500, "02   SIX LEGS / TWO ARMS"),
        ("head", 0, 695, 533, 333, "03   HEAD"),
        ("left_wrist", 533, 695, 533, 333, "04   LEFT WRIST"),
        ("right_wrist", 1066, 695, 534, 333, "05   RIGHT WRIST"),
    ]
    for name, x, y, width, height, label in panels:
        tile = Image.open(output / f"{name}.png").resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(tile, (x, y+40))
        draw.text((x+24, y+8), label, font=font, fill="#edf5f7")
    canvas.save(output / "preview.png")
    print(f"Rendered six cameras and contact sheet to {output}")


def inspect():
    model, data = load_scene()
    contacts = []
    for contact in data.contact:
        contacts.append({"geoms":[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in contact.geom], "distance":float(contact.dist)})
    robot_id = model.body("chassis").id
    print(json.dumps({"nq":model.nq,"nv":model.nv,"actuators":model.nu,
                      "cameras":[model.camera(i).name for i in range(model.ncam)],
                      "robot_mass_kg":float(model.body_subtreemass[robot_id]),
                      "initial_contacts":contacts},indent=2))


def view(seconds=None, static=False, speed=1., camera="third_person"):
    # uv's standalone CPython dylib is outside the venv. Supply its actual
    # location before mjpython execs its Cocoa launcher (upstream issue #1923).
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        env = os.environ.copy()
        paths = [sysconfig.get_config_var("LIBDIR"), str(Path(sys.base_prefix)/"lib")]
        paths += env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib").split(":")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
        launcher = Path(sys.executable).parent / "mjpython"
        argv = [sys.executable, str(launcher), "-m", "sixlegs.cli", "view", "--speed", str(speed), "--camera", camera]
        if static:
            argv += ["--static"]
        if seconds is not None:
            argv += ["--seconds", str(seconds)]
        os.execve(sys.executable, argv, env)
    import mujoco.viewer
    from sixlegs.simulation import Simulation
    sim = None if static else Simulation()
    model, data = load_scene() if static else (sim.model, sim.data)
    events = SimpleQueue()
    paused = False
    cameras = {49:"third_person", 50:"head", 51:"left_wrist", 52:"right_wrist", 53:"overhead"}
    with mujoco.viewer.launch_passive(model, data, key_callback=events.put,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera(camera).id
        viewer.opt.geomgroup[3] = 0
        viewer.opt.sitegroup[:] = 0
        start = last = time.monotonic()
        budget = 0.
        reported = False
        while viewer.is_running() and (seconds is None or time.monotonic()-start < seconds):
            now = time.monotonic()
            elapsed = min(now-last, .1)
            last = now
            while not events.empty():
                key = events.get()
                if key == 32:
                    paused = not paused
                elif key in cameras:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    viewer.cam.fixedcamid = model.camera(cameras[key]).id
                elif key == 54:
                    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                    viewer.cam.trackbodyid = model.body("chassis").id
                    viewer.cam.distance = 3.6
                    viewer.cam.azimuth = 135
                    viewer.cam.elevation = -25
                elif key in (82,114) and sim is not None:
                    from sixlegs.task import TransferDemo
                    mujoco.mj_resetDataKeyframe(model,data,0)
                    mujoco.mj_forward(model,data)
                    sim.demo=TransferDemo(model,data)
                    sim.ticks=0
                    sim.maximum_torque_fraction=0.
                    paused=False
                    reported=False
                    budget=0.
            if sim is not None and not paused and not sim.demo.done:
                budget += elapsed*speed
                while budget >= model.opt.timestep and not sim.demo.done:
                    sim.step()
                    budget -= model.opt.timestep
            else:
                budget=0.
            status = "STATIC PREVIEW" if static else sim.demo.phase.name
            if sim is not None and sim.demo.done:
                status = "PASS - both objects placed and released" if sim.demo.success else "FAILED - inspect terminal report"
                if not reported:
                    print(json.dumps(sim.report(),indent=2),flush=True)
                    reported=True
            viewer.set_texts((None,None,
                f"SIXLEGS | {data.time:.1f}s | {'PAUSED' if paused else status}",
                "1 Scene   2 Head   3 Left wrist   4 Right wrist   5 Overhead   6 Follow\nSpace Pause / Resume   R Restart"))
            viewer.sync()
            time.sleep(.005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render","view","inspect","run","record"), default="render", nargs="?")
    parser.add_argument("--seconds", type=float, help="Close the viewer after this duration (smoke test)")
    parser.add_argument("--output", type=Path, help="Output directory (render/run), or MP4 path (record)")
    parser.add_argument("--static", action="store_true", help="Freeze the viewer for scene inspection")
    parser.add_argument("--speed", type=float, default=1., help="Live simulation / video playback speed")
    parser.add_argument("--camera", default="third_person", choices=("third_person","robot_detail","overhead","head","left_wrist","right_wrist"))
    parser.add_argument("--layout", choices=("overview","all"), default="overview", help="Video: overview + wrists/head, or all six views")
    parser.add_argument("--trajectory", type=Path, help="Saved NPZ to record instead of rerunning simulation")
    args = parser.parse_args()
    if args.speed <= 0:
        parser.error("--speed must be positive")
    if args.command == "render":
        render(args.output or ROOT/"previews")
    elif args.command == "view":
        view(args.seconds,args.static,args.speed,args.camera)
    elif args.command == "run":
        from sixlegs.simulation import run_headless
        run_headless(args.output or ROOT/"outputs")
    elif args.command == "record":
        from sixlegs.simulation import run_headless
        from sixlegs.recording import render_video
        trajectory=args.trajectory
        if trajectory is None:
            run_headless(ROOT/"outputs")
            trajectory=ROOT/"outputs/transfer.npz"
        render_video(trajectory,args.output or ROOT/"previews/transfer.mp4",speed=args.speed,layout=args.layout)
    else:
        inspect()

if __name__ == "__main__":
    main()
