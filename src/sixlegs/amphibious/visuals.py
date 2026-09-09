"""Synchronized views of the saved physical trajectory."""

import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sixlegs.recording import font

from .scene import load_scene


def options():
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    return opt


def follow(data, side=False):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.qpos[:3]
    cam.lookat[2] += 0.03
    cam.distance = 1.65 if side else 1.85
    cam.azimuth = 90 if side else 135
    cam.elevation = -8 if side else -22
    return cam


def preview(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    m, d = load_scene()
    with mujoco.Renderer(m, 900, 1600) as renderer:
        for camera in ("overview", "detail", "side", "front"):
            renderer.update_scene(d, camera=camera, scene_option=options())
            Image.fromarray(renderer.render()).save(output / f"{camera}.png")
    print(f"Saved previews to {output}")


def dashboard(main, small, model, data, sample, speed):
    frame = Image.new("RGB", (1920, 1080), "#101e28")
    draw = ImageDraw.Draw(frame)
    main.update_scene(data, camera=follow(data), scene_option=options())
    frame.paste(Image.fromarray(main.render()), (0, 72))
    for camera, y, label in (
        (follow(data, True), 72, "SIDE / LEG & FLOAT POSTURE"),
        ("overview", 450, "OVERVIEW / COMPLETE CROSSING"),
    ):
        small.update_scene(data, camera=camera, scene_option=options())
        frame.paste(Image.fromarray(small.render()), (1200, y))
        draw.rectangle((1212, y + 10, 1906, y + 43), fill="#101e28")
        draw.text((1222, y + 15), label, font=font(19), fill="#c4e3e5")
    draw.text(
        (22, 18), "AMPHIBIOUS / LEG-MOUNTED FLOATS", font=font(30), fill="#eff7f8"
    )
    draw.text(
        (1230, 22),
        f"{sample['phase']}   |   {sample['time']:.1f}s   |   {speed:g}x replay",
        font=font(23),
        fill="#74d5c8",
    )
    draw.text(
        (22, 840),
        "CONTACT-DRIVEN WALK  →  FLOAT  →  WALK OUT",
        font=font(27),
        fill="#f2b65b",
    )
    draw.text(
        (22, 888),
        "Go2 physical model · scripted joint control · assumed sealed hull",
        font=font(23),
        fill="#bdd0da",
    )
    draw.text(
        (22, 927),
        "Sampled-volume buoyancy + drag · idealized twin water thrusters",
        font=font(23),
        fill="#bdd0da",
    )
    draw.text(
        (22, 1005),
        "MuJoCo rigid-body simulation  /  Concept demonstration, not a calibrated hardware replica",
        font=font(21),
        fill="#86a5b6",
    )
    draw.text((1230, 844), "PHYSICAL SUPPORT", font=font(25), fill="#eff7f8")
    weight = sample["weight_n"]
    buoy = sum(sample["buoyancy_n"])
    draw.text(
        (1230, 887),
        f"Buoyancy {buoy:.0f} N / weight {weight:.0f} N",
        font=font(23),
        fill="#a8e5dc",
    )
    draw.rectangle((1230, 923, 1885, 941), fill="#284350")
    draw.rectangle(
        (1230, 923, 1230 + 655 * min(1.0, buoy / weight), 941), fill="#55c8b6"
    )
    status = "GROUND CONTACT" if sample["ground_contact"] else "NO GROUND CONTACT"
    draw.text((1230, 960), status, font=font(23), fill="#f2b65b")
    draw.text(
        (1230, 1001),
        f"Feet touching: {sum(sample['foot_contacts'])}/4   |   Thrust: {sum(sample['thrust_n']):.1f} N",
        font=font(21),
        fill="#bdd0da",
    )
    return frame


def record(run_dir, output, speed=8.0, fps=25):
    directory, output = Path(run_dir), Path(output)
    report = json.loads((directory / "report.json").read_text())
    if not report["success"]:
        raise ValueError(
            "Refusing to publish an incomplete crossing as the final video"
        )
    trajectory = np.load(directory / "trajectory.npz")
    samples = json.loads((directory / "telemetry.json").read_text())
    times = trajectory["time"]
    output.parent.mkdir(parents=True, exist_ok=True)
    m, d = load_scene()
    # Give the attachment rotation and return to ground contact time to read.
    enter = next(e["time"] for e in report["phases"] if e["phase"] == "FLOATING")
    leave = next(e["time"] for e in report["phases"] if e["phase"] == "EXIT")
    replay = [(float(times[0]), speed)] * (2 * fps)
    t = float(times[0])
    while t < times[-1]:
        rate = min(speed, 2.0) if enter - 2 <= t <= leave + 6 else speed
        replay.append((t, rate))
        t += rate / fps
    replay.extend([(float(times[-1]), speed)] * (3 * fps))
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1920, 1080),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=2,
        output_params=["-movflags", "+faststart", "-threads", "4"],
    )
    writer.send(None)
    captured = set()
    with mujoco.Renderer(m, 750, 1200) as main, mujoco.Renderer(m, 366, 720) as small:
        try:
            for n, (t, rate) in enumerate(replay):
                i = max(0, int(np.searchsorted(times, t, side="right")) - 1)
                d.qpos[:] = trajectory["qpos"][i]
                mujoco.mj_forward(m, d)
                frame = dashboard(main, small, m, d, samples[i], rate)
                writer.send(np.asarray(frame))
                phase = samples[i]["phase"]
                if phase not in captured and (
                    phase != "FLOATING"
                    or samples[i]["time"]
                    > next(
                        e["time"] for e in report["phases"] if e["phase"] == "FLOATING"
                    )
                    + 4
                ):
                    frame.save(
                        output.parent / f"frame-{phase.lower().replace(' ', '-')}.png"
                    )
                    captured.add(phase)
                    if phase == "FLOATING":
                        frame.save(output.with_suffix(".png"))
                if n % 100 == 0:
                    print(f"{n}/{len(replay)} video frames", flush=True)
        finally:
            writer.close()
    return {
        "path": str(output),
        "fps": fps,
        "frames": len(replay),
        "duration_s": len(replay) / fps,
        "speed": speed,
        "attachment_transition_speed": min(speed, 2.0),
    }
