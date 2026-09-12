"""Preview the native theme, or render the accepted cached moving trace with it."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import ROOT, BodySpec, build_model, initialize
from adaptive_locomotion.graphite import (
    BG,
    MUTED,
    ORANGE,
    TEAL,
    WHITE,
    configure,
    decorate,
    layout,
)
from adaptive_locomotion.moving_record import camera
from adaptive_locomotion.record import font, model_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def option():
    opt = mujoco.MjvOption()
    opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    return opt


def preview(output):
    canvas = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((30, 25, 36, 87), fill=TEAL)
    draw.text((57, 18), "GRAPHITE / NATIVE MUJOCO", font=font(42), fill=WHITE)
    draw.text(
        (58, 76),
        "STATIC DESIGN PREVIEW   /   Unbranded shell, neutral lighting, focused color",
        font=font(24),
        fill=MUTED,
    )
    cases = (
        (BodySpec(), "flat", 90, "01 / RIGHT SIDE", "Unbranded curved side panel"),
        (BodySpec(), "flat", -90, "02 / LEFT SIDE", "Original vendor assets preserved"),
        (
            BodySpec("whole_fr", absent_legs=("FR",)),
            "flat",
            50,
            "03 / LIMB REMOVAL",
            "Orange marks the physical cut",
        ),
        (
            BodySpec(),
            "moving",
            50,
            "04 / MOVING SUPPORT",
            "Teal identifies the task surface",
        ),
    )
    for i, (body, terrain, azimuth, title, note) in enumerate(cases):
        model = build_model(body, terrain=terrain)
        data = mujoco.MjData(model)
        initialize(model, data)
        configure(model)
        model.vis.global_.offwidth, model.vis.global_.offheight = 960, 380
        with mujoco.Renderer(model, height=380, width=960) as renderer:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = data.qpos[:3] - [0, 0, 0.04 if terrain == "flat" else 0.2]
            cam.azimuth, cam.elevation = azimuth, -20 if terrain == "flat" else -25
            cam.distance = 1.3 if terrain == "flat" else 2.5
            renderer.update_scene(data, camera=cam, scene_option=option())
            decorate(renderer.scene, model, data, body)
            x, y = i % 2 * 960, 130 + i // 2 * 460
            canvas.paste(Image.fromarray(renderer.render()), (x, y + 40))
            draw.text((x + 28, y), title, font=font(25), fill=WHITE)
            draw.text(
                (x + 28, y + 423), note, font=font(21), fill=ORANGE if i == 2 else MUTED
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def replay(output, trace_root):
    source = ROOT / "previews/locomotion/moving/moving_supports_v1.json"
    manifest = json.loads(source.read_text())
    case = next(c for c in manifest["cases"] if c["key"] == "combined_2")
    path = trace_root / case["trajectory"]
    if sha(path) != case["trajectory_sha256"]:
        raise ValueError("Recorded trajectory changed")
    model = mujoco.MjModel.from_binary_path(str(trace_root / case["model_binary"]))
    if model_hash(model) != case["model_mjb_sha256"]:
        raise ValueError("Recorded physical model changed")
    with np.load(path, allow_pickle=False) as arrays:
        states = {k: arrays[k] for k in arrays.files}
    assert len(states["time"]) == 500 and np.isclose(states["time"][-1], 10)
    configure(model)
    data = mujoco.MjData(model)
    model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1920, 1080),
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    started = time.perf_counter()
    try:
        with (
            mujoco.Renderer(model, height=840, width=1260) as large,
            mujoco.Renderer(model, height=404, width=592) as small,
        ):
            for frame in range(250):
                k = 2 * frame + 1
                data.qpos[:], data.qvel[:], data.ctrl[:] = (
                    states["qpos"][k],
                    states["qvel"][k],
                    states["ctrl"][k],
                )
                data.time = states["time"][k]
                mujoco.mj_forward(model, data)
                images = []
                for kind, renderer in (
                    ("follow", large),
                    ("overview", small),
                    ("head", small),
                ):
                    renderer.update_scene(
                        data, camera=camera(model, data, kind), scene_option=option()
                    )
                    decorate(renderer.scene, model, data, BodySpec())
                    images.append(Image.fromarray(renderer.render()))
                canvas = layout(
                    *images,
                    title="BALANCING ON A MOVING WORLD",
                    subtitle="Healthy robot / combined platform motion / one frozen policy",
                    seconds=float(data.time),
                    drift=float(states["relative_drift_m"][k]),
                    contact=states["tip_forces"][k],
                )
                writer.send(np.asarray(canvas))
    finally:
        writer.close()
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "theme": "graphite",
        "generated_media": False,
        "source_manifest": str(source.relative_to(ROOT)),
        "source_manifest_sha256": sha(source),
        "checkpoint_sha256": manifest["checkpoint_hashes"][1],
        "trajectory_sha256": case["trajectory_sha256"],
        "model_mjb_sha256": case["model_mjb_sha256"],
        "case": "combined_2",
        "trial": 0,
        "seed": manifest["seed"],
        "physics_backend": manifest["physics_backend"],
        "new_simulation_steps": 0,
        "new_training_seconds": 0,
        "render_encode_seconds": time.perf_counter() - started,
        "frames": 250,
        "fps": 25,
        "duration_s": 10,
        "dimensions": [1920, 1080],
        "playback_speed": 1,
        "cameras": ["third person", "overhead", "head"],
        "state_indices": "1,3,...,499 at 50 Hz; actual 0.04–10.00 s timestamps",
        "cosmetic_panels": "MjvScene-only curved side covers and small forward covers; no added physical geometry",
        "video_sha256": sha(output),
        "visual_inspection": "pending",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preview-only", action="store_true")
    p.add_argument("--trace-root", type=Path, default=ROOT)
    args = p.parse_args()
    if args.preview_only:
        preview(args.output)
    else:
        replay(args.output, args.trace_root)


if __name__ == "__main__":
    main()
