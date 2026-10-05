"""Static inspection views; all panels share one unchanged MuJoCo state."""

import hashlib
import json
from pathlib import Path
import time

import mujoco
from PIL import Image, ImageDraw, ImageFont

from .scene import CAMERAS, ROOT, SCENE, load_scene

LABELS = {
    "overview": ("01  /  SURGICAL WORKCELL", "Cartesian gantry  /  tissue platform  /  thread cassette"),
    "mechanism": ("02  /  POSITIONING + TOOL HEAD", "XYZ carriage  /  insertion slide  /  retainer  /  microscope"),
    "surgical_field": ("03  /  WORK AREA", "Six target sites  /  vessel map  /  four thread samples"),
    "tool_clearance": ("04  /  NEEDLE + RETAINER", "150 micrometre needle diameter  /  surface approach clearance"),
    "microscope": ("05  /  MOUNTED MICROSCOPE", "Tool-relative view of the target field"),
    "thread_fixture": ("06  /  THREAD PRESENTATION", "Cassette geometry  /  40 micrometre thread samples"),
}


def font(size):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def options():
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    return opt


def labeled(raw, camera):
    w, h = raw.size
    frame = Image.new("RGB", (w, h + 102), "#101c25")
    frame.paste(raw, (0, 64))
    draw = ImageDraw.Draw(frame)
    title, detail = LABELS[camera]
    draw.rectangle((22, 21, 27, 43), fill="#43c8b5")
    draw.text((42, 19), title, font=font(25), fill="#eef4f5")
    draw.text((22, h + 75), detail, font=font(18), fill="#b2c5cf")
    return frame


def preview(output=ROOT / "previews/neural_insertion/static_v1"):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model, data = load_scene()
    renders = {}
    with mujoco.Renderer(model, height=1000, width=1600) as renderer:
        for name in CAMERAS:
            renderer.update_scene(data, camera=name, scene_option=options())
            frame = labeled(Image.fromarray(renderer.render()), name)
            frame.save(output / f"{name}.png")
            renders[name] = frame
    sheet = Image.new("RGB", (2400, 1765), "#101c25")
    draw = ImageDraw.Draw(sheet)
    draw.text((30, 22), "SURGICAL INSERTION ROBOT", font=font(36), fill="#eef4f5")
    draw.text((30, 70), "STATIC DESIGN REVIEW  /  v1", font=font(20), fill="#67d7c6")
    for i, name in enumerate(CAMERAS):
        tile = renders[name].resize((1200, 826), Image.Resampling.LANCZOS)
        # A separate two-column sheet gives useful size to tool and field details.
        if i < 4:
            sheet.paste(tile, ((i % 2) * 1200, 113 + (i // 2) * 826))
    sheet.save(output / "review.png")
    # Full six-view sheet is also useful for the whole inspection in one window.
    contact = Image.new("RGB", (2400, 1234), "#101c25")
    draw = ImageDraw.Draw(contact)
    draw.text((26, 20), "SURGICAL INSERTION ROBOT  /  STATIC v1", font=font(30), fill="#eef4f5")
    draw.text((26, 62), "5 actuated axes  /  6 target sites  /  4 thread samples", font=font(20), fill="#67d7c6")
    for i, name in enumerate(CAMERAS):
        contact.paste(renders[name].resize((800, 551), Image.Resampling.LANCZOS), ((i % 3) * 800, 118 + (i // 3) * 558))
    contact.save(output / "all_views.png")
    report = {
        "preset": "static_v1", "mujoco_version": mujoco.__version__,
        "scene_sha256": hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        "simulated_seconds": float(data.time), "training_seconds": 0,
        "render_elapsed_seconds": time.perf_counter() - started,
        "cameras": list(CAMERAS), "qpos": data.qpos.tolist(),
        "images": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.glob("*.png"))},
    }
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
