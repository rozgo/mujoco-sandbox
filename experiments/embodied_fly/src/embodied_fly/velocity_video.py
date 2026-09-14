"""Continuous teacher review: body motion, heading and all commanded velocities."""

import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.record import font
from embodied_fly.velocity_exercise import STAGES, rolling_velocity


def record(source, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("Preserve earlier teacher videos")
    report = json.loads((source / "report.json").read_text())
    speed_scale = report.get("physical_command_speed_scale", 1.0)
    tracking_seconds = 0.12 if speed_scale > 1 else 0.4
    assert sha256(source / "capture.npz") == report["capture_sha256"]
    assert sha256(source / "model.mjb") == report["model_sha256"]
    with np.load(source / "capture.npz") as data:
        arrays = {key: data[key] for key in data.files}
    model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    positions = arrays["qpos"][:, :3]
    tracked = positions[0].copy()
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -24, 1.6
    overhead = mujoco.MjvCamera()
    overhead.azimuth, overhead.elevation = 90, -85
    overhead.lookat[:] = (positions.min(0) + positions.max(0)) / 2
    overhead.distance = max(1.5, float(np.ptp(positions, axis=0).max()) * 2.5)
    raw = np.column_stack((arrays["measured_velocity"] * 10, np.degrees(arrays["yaw_rate"])))
    mean = rolling_velocity(raw)
    requested = arrays["command"] * (10, 10, 10, 180 / np.pi)
    ranges = np.maximum((3, 3, 3, 35), np.ceil(np.abs(raw).max(0) * 1.1))
    fps, size = 50, (1600, 1000)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        size,
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    frame_count = 0
    try:
        with (
            mujoco.Renderer(model, height=570, width=1080) as main,
            mujoco.Renderer(model, height=330, width=480) as top,
        ):
            for step in range(0, len(arrays["time"]), 10):
                t = float(arrays["time"][step])
                stage = int(arrays["stage"][step])
                board = Image.new("RGB", size, "#111519")
                draw = ImageDraw.Draw(board)
                draw.text(
                    (20, 12),
                    "FLIGHT SCHOOL / PID TEACHER"
                    + (" / FAST FLIGHT" if speed_scale > 1 else ""),
                    font=font(30),
                    fill="#ffc31f",
                )
                draw.text(
                    (20, 54),
                    "One continuous exercise  /  Velocity + turn-rate commands  /  Physical wing control  /  1x",
                    font=font(20),
                    fill="#e6e1db",
                )
                for key in ("qpos", "qvel", "act", "ctrl"):
                    getattr(data, key)[:] = arrays[key][step]
                data.time = t
                mujoco.mj_forward(model, data)
                tracked += (1 - np.exp(-1 / fps / tracking_seconds)) * (
                    positions[step] - tracked
                )
                camera.lookat[:] = tracked
                main.update_scene(data, camera=camera, scene_option=option)
                board.paste(Image.fromarray(main.render()), (12, 125))
                top.update_scene(data, camera=overhead, scene_option=option)
                board.paste(Image.fromarray(top.render()), (1104, 125))
                draw.rectangle((12, 88, 1584, 122), fill="#22272b")
                draw.text(
                    (22, 91),
                    f"{stage + 1:02d}/{len(STAGES)}   {STAGES[stage][0]}",
                    font=font(23),
                    fill="#ffc31f",
                )
                draw.text((1114, 134), "FIXED OVERVIEW", font=font(18), fill="#e6e1db")
                draw.rectangle((1104, 468, 1584, 694), fill="#1b2025")
                yaw = float(arrays["heading"][step])
                center = np.array([1180, 551])
                draw.ellipse((1135, 506, 1225, 596), outline="#727a80", width=2)
                tip = center + np.array([np.cos(yaw), -np.sin(yaw)]) * 39
                draw.line((tuple(center), tuple(tip)), fill="#ffc31f", width=5)
                draw.text((1245, 488), "BODY HEADING", font=font(17), fill="#a8b0b5")
                draw.text(
                    (1245, 515), f"{np.degrees(yaw):+.1f} deg", font=font(29), fill="#e6e1db"
                )
                draw.text(
                    (1245, 555),
                    f"Height {positions[step, 2] * 10:.2f} mm",
                    font=font(20),
                    fill="#e6e1db",
                )
                draw.text(
                    (1120, 608),
                    "1,000 Hz physics / 500 Hz controls",
                    font=font(19),
                    fill="#a8b0b5",
                )
                draw.text(
                    (1120, 640),
                    "PID reference / no student training",
                    font=font(19),
                    fill="#a8b0b5",
                )
                draw.text(
                    (24, 661),
                    (
                        f"{speed_scale:g}x commanded speeds / 1x playback / physical wing control"
                        if speed_scale > 1
                        else "Damped observer camera / continuous physical state"
                    ),
                    font=font(19),
                    fill="#e6e1db",
                    stroke_width=1,
                    stroke_fill="#111519",
                )
                draw.text(
                    (20, 708),
                    "Amber: command    Green: measured velocity, 100 ms mean    Gray strip: raw physics velocity",
                    font=font(19),
                    fill="#a8b0b5",
                )
                start_time = max(0, t - 8)
                history = np.arange(max(0, step - 4000), step + 1, 5)
                for axis, label in enumerate(
                    (
                        "FORWARD / BACK  mm/s",
                        "LEFT / RIGHT  mm/s",
                        "UP / DOWN  mm/s",
                        "TURN RATE  deg/s",
                    )
                ):
                    x, y = 12 + 396 * axis, 734
                    draw.rectangle((x, y, x + 384, y + 208), fill="#1b2025")
                    draw.text((x + 10, y + 8), label, font=font(18), fill="#e6e1db")
                    zero = y + 140
                    draw.line((x + 10, zero, x + 374, zero), fill="#555e65")
                    draw.text(
                        (x + 10, y + 30),
                        f"Raw +/-{ranges[axis]:g}",
                        font=font(13),
                        fill="#8b9398",
                    )
                    chart_x = x + 10 + (arrays["time"][history] - start_time) / 8 * 364
                    raw_points = list(
                        zip(chart_x, y + 64 - raw[history, axis] / ranges[axis] * 18)
                    )
                    if len(raw_points) > 1:
                        draw.line(raw_points, fill="#777f85", width=1)
                    mean_range = max(
                        35 if axis == 3 else 3,
                        float(np.ceil(np.abs(mean[history, axis]).max() * 1.1)),
                        float(np.ceil(np.abs(requested[history, axis]).max() * 1.1)),
                    )
                    draw.text(
                        (x + 10, y + 88),
                        f"Mean +/-{mean_range:g}",
                        font=font(13),
                        fill="#8b9398",
                    )
                    for values, color, width in (
                        (requested, "#ffc31f", 2),
                        (mean, "#70a88a", 2),
                    ):
                        points = list(
                            zip(
                                chart_x,
                                zero - values[history, axis] / mean_range * 35,
                            )
                        )
                        if len(points) > 1:
                            draw.line(points, fill=color, width=width)
                    draw.text(
                        (x + 10, y + 184),
                        f"Goal {requested[step, axis]:+.2f}   Actual {mean[step, axis]:+.2f}",
                        font=font(17),
                        fill="#e6e1db",
                    )
                draw.rectangle((12, 950, 1584, 957), fill="#33393e")
                draw.rectangle(
                    (12, 950, 12 + 1572 * t / report["duration_seconds"], 957), fill="#ffc31f"
                )
                draw.text(
                    (20, 966),
                    f"{t:05.2f} / {report['duration_seconds']:.2f} s   |   No episode resets   |   Velocity zero = brake and hover",
                    font=font(19),
                    fill="#a8b0b5",
                )
                writer.send(np.asarray(board))
                frame_count += 1
    finally:
        writer.close()
    manifest = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "controller": report["controller"],
        "learned_actor": False,
        "source_report_sha256": sha256(source / "report.json"),
        "capture_sha256": report["capture_sha256"],
        "model_sha256": report["model_sha256"],
        "sha256": sha256(output),
        "frames": frame_count,
        "fps": fps,
        "size": size,
        "duration_s": frame_count / fps,
        "playback_multiplier": 1,
        "physical_command_speed_scale": speed_scale,
        "command_speed_mm_s": report["command_speed_mm_s"],
        "yaw_command_rad_s": report["yaw_command_rad_s"],
        "render_seconds": time.perf_counter() - started,
        "measurement_display": "Raw velocity and trailing 100 ms mean; mean is observer-only and never feeds PID or physics",
        "raw_chart_ranges": ranges.tolist(),
        "mean_chart_range": "Labeled per trailing window; minimum +/-3 mm/s and +/-35 deg/s; expands to include cold-start transient without clipping",
        "continuous_episode": True,
        "teacher_metric_gate_passed": report["passed"],
        "camera": {
            "main": f"damped position tracking, {tracking_seconds:g} s time constant, fixed azimuth",
            "overview": "fixed for entire episode",
        },
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    record(args.source, args.output)
