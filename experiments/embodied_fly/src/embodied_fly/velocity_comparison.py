"""Timestamp-matched teacher/student video, including explicit failed-episode ends."""

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


def record(args):
    started = time.perf_counter()
    if args.output.exists():
        raise FileExistsError("Preserve previous comparison videos")
    report = json.loads((args.student / "report.json").read_text())
    training = json.loads(args.training_report.read_text())
    long_run = training["requested_training_seconds"] >= 120
    training_title = (
        f"{training['requested_training_seconds'] / 60:.0f}-MINUTE TRAINING"
        if long_run
        else "FIRST LEARNING BURST"
    )
    training_caption = (
        f"MaleCNS  /  {training['training_wall_seconds'] / 60:.1f} min this run"
        f"  /  {training['cumulative_training_seconds'] / 60:.1f} min total  /  Same physics  /  1x"
        if long_run
        else f"Fresh MaleCNS student  /  {training['training_wall_seconds']:.1f}s training  /  Same flight physics  /  1x"
    )
    teacher_report = json.loads((args.dataset / "report.json").read_text())
    assert report["physical_contract"] == teacher_report["physical_contract"]
    assert report["checkpoint_sha256"] == training["checkpoint_sha256"]
    model = mujoco.MjModel.from_binary_path(str(args.student / "model.mjb"))
    assert sha256(args.student / "model.mjb") == report["model_sha256"]
    model.vis.global_.offwidth = max(800, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -20, 3.8
    fps, size = 50, (1600, 960)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
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
    frames = 0
    sections = []
    try:
        with (
            mujoco.Renderer(model, height=440, width=776) as renderer,
            mujoco.Renderer(model, height=170, width=240) as detail,
        ):
            for case in report["cases"]:
                world = case["episode"]
                teacher_path = args.dataset / f"episode_{world:02d}.npz"
                student_path = args.student / case["file"]
                assert sha256(teacher_path) == case["teacher_capture_sha256"]
                assert sha256(student_path) == case["capture_sha256"]
                with np.load(teacher_path) as f:
                    teacher = {k: f[k] for k in f.files}
                with np.load(student_path) as f:
                    student = {k: f[k] for k in f.files}
                duration = min(71.2, max(6, case["duration_seconds"]))
                sections.append(
                    {
                        "episode": world,
                        "video_start_seconds": frames / fps,
                        "video_duration_seconds": duration,
                        "student_duration_seconds": case["duration_seconds"],
                    }
                )
                tracked = [teacher["qpos"][0, :3].copy(), student["qpos"][0, :3].copy()]
                arrays_list = (teacher, student)
                means = [rolling_velocity(a["measured_velocity"] * 10) for a in arrays_list]
                yaw = [
                    np.degrees(rolling_velocity(a["yaw_rate"][:, None])[:, 0])
                    for a in arrays_list
                ]
                for frame in range(round(duration * fps)):
                    t = frame / fps
                    step = round(t * 500)
                    board = Image.new("RGB", size, "#111519")
                    draw = ImageDraw.Draw(board)
                    draw.text(
                        (22, 15),
                        "FLIGHT SCHOOL / " + training_title,
                        font=font(31),
                        fill="#ffc31f",
                    )
                    draw.text(
                        (22, 57),
                        training_caption,
                        font=font(22),
                        fill="#e6e1db",
                    )
                    stage = int(teacher["stage"][min(step, len(teacher["time"]) - 1)])
                    draw.text(
                        (22, 94),
                        f"{'CANONICAL START' if world == 0 else 'HELD-OUT START'}  /  {STAGES[stage][0]}  /  {t:.2f}s",
                        font=font(22),
                        fill="#aab3b8",
                    )
                    for column, arrays in enumerate(arrays_list):
                        x = 12 + 800 * column
                        ended = column == 1 and step >= len(arrays["time"])
                        draw.rectangle((x, 130, x + 776, 170), fill="#252a2e")
                        draw.text(
                            (x + 12, 136),
                            "PID TEACHER" if column == 0 else "MALECNS STUDENT",
                            font=font(25),
                            fill="#ffc31f" if column == 0 else "#70a88a",
                        )
                        if ended:
                            draw.rectangle((x, 176, x + 776, 616), fill="#1c2226")
                            draw.text(
                                (x + 36, 286), "EPISODE ENDED", font=font(35), fill="#df6b54"
                            )
                            draw.text(
                                (x + 36, 344),
                                case["failure"] or "Completed",
                                font=font(24),
                                fill="#e6e1db",
                            )
                            failure_time = case["first_failure_seconds"]
                            draw.text(
                                (x + 36, 390),
                                f"First failure: {failure_time:.3f}s"
                                if failure_time is not None
                                else f"Capture ended: {case['duration_seconds']:.3f}s",
                                font=font(22),
                                fill="#aab3b8",
                            )
                            draw.text(
                                (x + 36, 430),
                                "No resets or assisted recovery",
                                font=font(22),
                                fill="#aab3b8",
                            )
                        else:
                            for key in ("qpos", "qvel", "act", "ctrl"):
                                getattr(data, key)[:] = arrays[key][step]
                            data.time = t
                            mujoco.mj_forward(model, data)
                            tracked[column] += (1 - np.exp(-1 / fps / 0.2)) * (
                                arrays["qpos"][step, :3] - tracked[column]
                            )
                            camera.lookat[:] = tracked[column]
                            camera.lookat[2] = max(0.9, tracked[column][2] - 0.8)
                            renderer.update_scene(data, camera=camera, scene_option=option)
                            board.paste(Image.fromarray(renderer.render()), (x, 176))
                            if args.wing_detail:
                                close = mujoco.MjvCamera()
                                close.azimuth, close.elevation, close.distance = 135, -24, 1.25
                                close.lookat[:] = arrays["qpos"][step, :3]
                                detail.update_scene(data, camera=close, scene_option=option)
                                board.paste(Image.fromarray(detail.render()), (x + 532, 182))
                                draw.text(
                                    (x + 540, 186),
                                    "BODY-FOLLOWING DETAIL",
                                    font=font(13),
                                    fill="#e6e1db",
                                )
                            draw.text(
                                (x + 16, 574),
                                f"Altitude {arrays['qpos'][step, 2] * 10:.2f} mm",
                                font=font(23),
                                fill="#e6e1db",
                            )
                    # Matched time and units, with no extrapolation after a failure.
                    charts = [
                        ("ALTITUDE / mm", None),
                        ("FORWARD / mm/s", 0),
                        ("SIDEWAYS / mm/s", 1),
                        ("YAW / deg/s", 3),
                    ]
                    for axis, (label, index) in enumerate(charts):
                        x, y, width = 12 + 400 * axis, 642, 376
                        draw.rectangle((x, y, x + width, y + 238), fill="#20262b")
                        draw.text((x + 10, y + 10), label, font=font(19), fill="#e6e1db")
                        values = [
                            a["qpos"][:, 2] * 10
                            if index is None
                            else yaw[i]
                            if index == 3
                            else means[i][:, index]
                            for i, a in enumerate(arrays_list)
                        ]
                        extent = max(
                            22 if index is None else 20 if index != 3 else 280,
                            max(
                                float(np.abs(v[: min(len(v), step + 1)]).max()) for v in values
                            )
                            * 1.05,
                        )
                        low = 0 if index is None else -extent
                        high = extent
                        bottom, top = y + 195, y + 51
                        draw.text(
                            (x + 10, y + 209),
                            f"{low:.0f} to {high:.0f}  |  {max(0, t - 6):.1f}-{max(6, t):.1f}s",
                            font=font(15),
                            fill="#aab3b8",
                        )
                        for i, (a, v) in enumerate(zip(arrays_list, values, strict=True)):
                            selected = np.flatnonzero(
                                (a["time"] >= max(0, t - 6)) & (a["time"] <= t)
                            )[::5]
                            points = [
                                (
                                    x + 10 + (a["time"][j] - max(0, t - 6)) / 6 * (width - 20),
                                    bottom - (v[j] - low) / (high - low) * (bottom - top),
                                )
                                for j in selected
                            ]
                            if len(points) > 1:
                                draw.line(
                                    points, fill="#ffc31f" if i == 0 else "#70a88a", width=2
                                )
                    draw.text(
                        (22, 900),
                        "AMBER: TEACHER   /   GREEN: STUDENT   /   Velocities: trailing 100 ms mean for display",
                        font=font(21),
                        fill="#aab3b8",
                    )
                    writer.send(np.asarray(board))
                    frames += 1
            board = Image.new("RGB", size, "#111519")
            draw = ImageDraw.Draw(board)
            draw.text(
                (70, 150),
                training_title if long_run else "ONE-MINUTE CHECKPOINT",
                font=font(44),
                fill="#ffc31f",
            )
            lines = [
                f"{training['updates_this_burst']} optimizer updates / {training['supervised_targets']:,} supervised targets",
                f"Held-out wing MSE: {training['validation_before']['wing_mse']:.5f} -> {training['validation_after']['wing_mse']:.5f}",
                f"Full flight exercises completed: {sum(c['completed_full_exercise'] for c in report['cases'])}/{len(report['cases'])}",
                "Fixed connectome wiring / fresh trainable encoder, readouts and cell dynamics",
                "Checkpoint + optimizer preserved; physical flight determines the outcome"
                if long_run
                else "Checkpoint + optimizer saved for the next one-minute burst",
            ]
            for line, text in enumerate(lines):
                draw.text((70, 255 + line * 74), text, font=font(27), fill="#e6e1db")
            for _ in range(3 * fps):
                writer.send(np.asarray(board))
                frames += 1
    finally:
        writer.close()
    manifest = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "sha256": sha256(args.output),
        "source_report_sha256": sha256(args.student / "report.json"),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "model_sha256": report["model_sha256"],
        "size": size,
        "fps": fps,
        "frames": frames,
        "duration_s": frames / fps,
        "playback_multiplier": 1,
        "sections": sections,
        "render_seconds": time.perf_counter() - started,
        "failure_display": "Student replaced with explicitly labeled end card after recorded failure continuation; no frozen body or synthetic recovery",
        "velocities": "Trailing 100 ms mean, display only",
        "result_card_seconds": 3,
        "wing_detail_inset": args.wing_detail,
        "training_wall_seconds": training["training_wall_seconds"],
        "cumulative_training_seconds": training["cumulative_training_seconds"],
    }
    args.output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--wing-detail", action="store_true")
    record(parser.parse_args())
