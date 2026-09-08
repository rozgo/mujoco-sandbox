"""Static preview tools. No locomotion or manipulation controller yet."""
import argparse
import json
import os
import sys
import sysconfig
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


def view(seconds=None):
    # uv's standalone CPython dylib is outside the venv. Supply its actual
    # location before mjpython execs its Cocoa launcher (upstream issue #1923).
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        env = os.environ.copy()
        paths = [sysconfig.get_config_var("LIBDIR"), str(Path(sys.base_prefix)/"lib")]
        paths += env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib").split(":")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(dict.fromkeys(p for p in paths if p))
        launcher = Path(sys.executable).parent / "mjpython"
        argv = [sys.executable, str(launcher), "-m", "sixlegs.cli", "view"]
        if seconds is not None:
            argv += ["--seconds", str(seconds)]
        os.execve(sys.executable, argv, env)
    import mujoco.viewer
    model, data = load_scene()
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = model.camera("third_person").id
        viewer.opt.geomgroup[3] = 0
        viewer.opt.sitegroup[:] = 0
        start = time.monotonic()
        while viewer.is_running() and (seconds is None or time.monotonic()-start < seconds):
            # Explicitly frozen for design review; no mj_step or state animation.
            viewer.sync()
            time.sleep(1/60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("render","view","inspect"), default="render", nargs="?")
    parser.add_argument("--seconds", type=float, help="Close the viewer after this duration (smoke test)")
    parser.add_argument("--output", type=Path, default=ROOT/"previews")
    args = parser.parse_args()
    if args.command == "render":
        render(args.output)
    elif args.command == "view":
        view(args.seconds)
    else:
        inspect()

if __name__ == "__main__":
    main()
