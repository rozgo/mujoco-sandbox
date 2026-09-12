"""Render saved physical states with synchronized inspector data."""

import argparse
import json
import shutil
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .paths import OUTPUTS, PREVIEWS
from .utility import ACTIONS


def font(size):
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def writer(path, size, fps=25):
    w = imageio_ffmpeg.write_frames(
        str(path),
        size,
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
        macro_block_size=1,
    )
    w.send(None)
    return w


def record(name, output="fly_lab_development_v1", selected=0, publish=False):
    start = time.perf_counter()
    source = OUTPUTS / name
    states = np.load(source / "physics.npz")
    telemetry = json.loads((source / "telemetry.json").read_text())
    frames = telemetry["frames"]
    if not (source / "scene.mjb").exists():
        raise ValueError(
            "Capture has no saved physical model; refusing to silently use a changed scene"
        )
    m = mj.MjModel.from_binary_path(str(source / "scene.mjb"))
    d = mj.MjData(m)
    body_ids = [
        m.body(f"fly_{i:02}/c_thorax").id for i in range(int(states["n_flies"]))
    ]
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0, 0, 1]
    camera.distance = 66
    camera.azimuth = 100
    camera.elevation = -57
    detail = mj.MjvCamera()
    detail.type = mj.mjtCamera.mjCAMERA_FREE
    detail.distance = 9.0
    detail.azimuth = 120
    detail.elevation = -25
    brain = Image.open(PREVIEWS / "brain_atlas_v1.png").resize((328, 328))
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    inspector = PREVIEWS / "inspector"
    raw = writer(source / "arena.mp4", (1200, 800))
    movie = writer(PREVIEWS / f"{output}.mp4", (1600, 1000))
    renderer = mj.Renderer(m, width=1200, height=800)
    close = mj.Renderer(m, width=360, height=252)
    duration = float(frames[-1]["t"])
    count = round(duration * 25)
    for out_index in range(count):
        at = min(
            int(np.searchsorted(states["times"], out_index / 25, side="right")),
            len(frames) - 1,
        )
        frame = frames[at]
        f = frame["flies"][selected]
        d.qpos[:] = states["qpos"][at]
        d.qvel[:] = states["qvel"][at]
        d.ctrl[:] = states["ctrl"][at]
        d.time = frame["t"]
        mj.mj_forward(m, d)
        renderer.update_scene(d, camera=camera)
        cam = renderer.scene.camera[0]
        forward = np.asarray(cam.forward)
        up = np.asarray(cam.up)
        right = np.cross(forward, up)
        center = (renderer.scene.camera[0].pos + renderer.scene.camera[1].pos) / 2
        tangent = cam.frustum_top / cam.frustum_near
        for i, fly in enumerate(frame["flies"]):
            delta = d.xpos[body_ids[i]] - center
            depth = max(float(np.dot(delta, forward)), 1e-6)
            fly["screen"] = [
                0.5 + float(np.dot(delta, right)) / (2 * depth * tangent * 1.5),
                0.5 - float(np.dot(delta, up)) / (2 * depth * tangent),
            ]
        world = renderer.render().copy()
        raw.send(world)
        canvas = Image.new("RGB", (1600, 1000), "#111313")
        canvas.paste(Image.fromarray(world), (24, 100))
        draw = ImageDraw.Draw(canvas)
        draw.text((28, 25), "FLY LAB", font=font(30), fill="#e6e1d8")
        draw.text(
            (225, 34),
            "INDEPENDENT NEEDS / SHARED UTILITY",
            font=font(15),
            fill="#959c93",
        )
        draw.text(
            (1250, 31), f"{frame['t']:05.2f} s   /   1×", font=font(18), fill="#ffc31f"
        )
        draw.rectangle((1244, 100, 1575, 965), fill="#1c1f1e", outline="#353a36")
        draw.text(
            (1262, 119),
            f"SELECTED / FLY {selected + 1:02}",
            font=font(17),
            fill="#e6e1d8",
        )
        action = ACTIONS[f["action"]].upper() if f["alive"] else "DEAD"
        draw.text(
            (1262, 152),
            action,
            font=font(23),
            fill="#ffc31f" if f["alive"] else "#d05c43",
        )
        for i, (label, score) in enumerate(zip(ACTIONS, f["scores"])):
            y = 205 + i * 31
            draw.text((1262, y), label, font=font(12), fill="#b9c0b5")
            draw.rectangle((1352, y + 4, 1535, y + 10), fill="#363c36")
            draw.rectangle(
                (1352, y + 4, 1352 + score * 183, y + 10),
                fill="#ffc31f" if i == f["action"] else "#6b7763",
            )
        for i, (label, value) in enumerate(
            zip(("Energy", "Water", "Fatigue", "Heat", "Health"), f["needs"])
        ):
            y = 412 + i * 32
            draw.text((1262, y), label, font=font(12), fill="#b9c0b5")
            draw.rectangle((1352, y + 4, 1535, y + 11), fill="#363c36")
            draw.rectangle(
                (1352, y + 4, 1352 + value * 183, y + 11),
                fill="#c64b3c" if i == 3 else "#8cac60",
            )
        canvas.paste(brain, (1246, 595))
        draw = ImageDraw.Draw(canvas)
        for x, y, alpha in f["brain"]:
            px = 1246 + (30 + x * 840) * 328 / 900
            py = 595 + (30 + y * 840) * 328 / 900
            color = (int(120 + 135 * alpha), int(90 + 105 * alpha), 25)
            draw.ellipse((px, py, px + 1.6, py + 1.6), fill=color)
        draw.text(
            (1262, 927),
            f"MaleCNS / {f['spikes']} spikes",
            font=font(12),
            fill="#b9c0b5",
        )
        detail.lookat[:] = d.xpos[body_ids[selected]]
        close.update_scene(d, camera=detail)
        canvas.paste(Image.fromarray(close.render()), (48, 625))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((48, 625, 408, 877), outline="#8e936f", width=1)
        draw.text(
            (60, 638), f"FLY {selected + 1:02} / FOLLOW", font=font(12), fill="#ffc31f"
        )
        for i, fly in enumerate(frame["flies"]):
            x = 30 + i * 145
            draw.rectangle(
                (x, 918, x + 129, 968),
                outline="#ffc31f" if i == selected else "#353a36",
            )
            draw.text(
                (x + 12, 926),
                f"{i + 1:02} / {ACTIONS[fly['action']] if fly['alive'] else 'dead'}",
                font=font(11),
                fill="#e6e1d8",
            )
            draw.text(
                (x + 12, 946), f"{fly['speed']:.1f} mm/s", font=font(10), fill="#929b94"
            )
        movie.send(np.asarray(canvas))
    raw.close()
    movie.close()
    renderer.close()
    close.close()
    if publish:
        inspector.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "arena.mp4", inspector / "arena.mp4")
        # Fill intermediate telemetry frames with the preceding rendered-frame projection.
        previous = None
        for frame in frames:
            if "screen" in frame["flies"][0]:
                previous = [f["screen"] for f in frame["flies"]]
            elif previous is not None:
                for fly, xy in zip(frame["flies"], previous):
                    fly["screen"] = xy
        (inspector / "telemetry.json").write_text(json.dumps(telemetry))
        shutil.copytree(source / "eyes", inspector / "eyes", dirs_exist_ok=True)
        shutil.copy2(
            Path(__file__).with_name("inspector.html"), inspector / "index.html"
        )
    result = {
        "source": name,
        "output": output,
        "fps": 25,
        "frames": count,
        "duration": count / 25,
        "render_seconds": time.perf_counter() - start,
        "mode": "saved MuJoCo state replay",
        "selected": selected,
    }
    (PREVIEWS / f"{output}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--output", default="fly_lab_development_v1")
    p.add_argument("--selected", type=int, default=0)
    p.add_argument("--publish", action="store_true")
    a = p.parse_args()
    record(a.name, a.output, a.selected, a.publish)
