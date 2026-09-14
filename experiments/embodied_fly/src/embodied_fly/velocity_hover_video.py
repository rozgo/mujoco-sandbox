"""Three-way, real-time PID/parent/PPO hover comparison from physical captures."""

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
from embodied_fly.velocity_exercise import rolling_velocity
from embodied_fly.velocity_hover import EVALUATION_EPISODES


def record(args):
    if args.output.exists():
        raise FileExistsError("Preserve prior videos")
    begin = time.perf_counter()
    training = json.loads((args.run / "report.json").read_text())
    reports = [training["evaluations"][label] for label in ("pid", "parent", "final")]
    roots = [args.run] * 3
    original = None
    if args.original_run:
        original = json.loads((args.original_run / "report.json").read_text())
        if original["physical_contract"] != training["physical_contract"]:
            raise ValueError("Original comparison must use the same physical contract")
        reports[1] = original["evaluations"]["parent"]
        roots[1] = args.original_run
    model = mujoco.MjModel.from_binary_path(str(args.run / "model.mjb"))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, 1920)
    model.vis.global_.offheight = max(model.vis.global_.offheight, 1080)
    data = mujoco.MjData(model)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -20, 3.8
    continued = training["recipe"].get("resumed_ppo_optimizer_and_critic", False)
    titles = (
        ("PID REFERENCE", "BEFORE CONTINUATION", "AFTER CONTINUATION")
        if continued
        else ("PID REFERENCE", "BEFORE PPO", "AFTER PPO")
    )
    if original:
        titles = ("PID REFERENCE", "ORIGINAL IMITATION", "AFTER PPO")
    colors = ("#b7c6d3", "#ffc31f", "#82b89b")
    fps, size = 50, (1920, 1080)
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
    frames, sections = 0, []
    with (
        mujoco.Renderer(model, height=460, width=612) as renderer,
        mujoco.Renderer(model, height=150, width=220) as detail,
    ):
        try:
            for case_index, episode in enumerate(EVALUATION_EPISODES):
                captures, cases = [], []
                for label, report, root in zip(
                    ("pid", "parent", "final"), reports, roots, strict=True
                ):
                    case = next(c for c in report["cases"] if c["episode"] == episode)
                    file = root / label / case["file"]
                    assert sha256(file) == case["sha256"]
                    with np.load(file) as saved:
                        captures.append({k: saved[k] for k in saved.files})
                    cases.append(case)
                for capture in captures[1:]:
                    np.testing.assert_array_equal(capture["qpos"][0], captures[0]["qpos"][0])
                    np.testing.assert_array_equal(capture["qvel"][0], captures[0]["qvel"][0])
                velocities = [rolling_velocity(c["measured_velocity"] * 10) for c in captures]
                heights = [c["post_position"][:, 2] * 10 for c in captures]
                speed_limit = max(
                    5, max(float(np.linalg.norm(v, axis=1).max()) for v in velocities) * 1.1
                )
                height_limit = max(25, max(float(h.max()) for h in heights) * 1.1)
                tracked = [c["qpos"][0, :3].copy() for c in captures]
                sections.append(
                    {
                        "episode": episode,
                        "video_start_seconds": frames / fps,
                        "duration_seconds": 10,
                    }
                )
                for frame in range(500):
                    t, step = frame / fps, frame * 10
                    board = Image.new("RGB", size, "#101213")
                    draw = ImageDraw.Draw(board)
                    draw.text(
                        (24, 18),
                        "FLIGHT SCHOOL  /  LEARNING TO HOVER",
                        font=font(34),
                        fill="#ffc31f",
                    )
                    draw.text(
                        (24, 66),
                        "Original imitation to learned flight  |  32 worlds  |  1,000 Hz physics / 500 Hz brain control  |  1x"
                        if original
                        else f"{training['training_wall_seconds'] / 60:.1f} min {'additional ' if continued else ''}PPO + light imitation  |  32 worlds  |  1,000 Hz physics / 500 Hz brain control  |  1x",
                        font=font(23),
                        fill="#e6e1db",
                    )
                    draw.text(
                        (24, 107),
                        f"CASE {case_index + 1}/4  /  {'HELD-OUT START' if episode >= 8 else 'TRAINING START'} {episode}  /  {t:.2f} s  /  COMMAND: ZERO TRANSLATION + ZERO TURN",
                        font=font(22),
                        fill="#aab3b8",
                    )
                    for col, (capture, case) in enumerate(zip(captures, cases, strict=True)):
                        x = 20 + 634 * col
                        draw.rectangle((x, 150, x + 612, 660), fill="#202426")
                        draw.text((x + 14, 159), titles[col], font=font(25), fill=colors[col])
                        if step < len(capture["time"]):
                            for key in ("qpos", "qvel", "act", "ctrl"):
                                getattr(data, key)[:] = capture[key][step]
                            data.time = t
                            mujoco.mj_forward(model, data)
                            tracked[col] += (1 - np.exp(-1 / fps / 0.2)) * (
                                capture["qpos"][step, :3] - tracked[col]
                            )
                            camera.lookat[:] = tracked[col]
                            camera.lookat[2] = max(0.9, tracked[col][2] - 0.8)
                            renderer.update_scene(data, camera=camera, scene_option=option)
                            board.paste(Image.fromarray(renderer.render()), (x, 200))
                            if args.detail:
                                close = mujoco.MjvCamera()
                                close.azimuth, close.elevation, close.distance = 135, -24, 1.0
                                close.lookat[:] = capture["qpos"][step, :3]
                                detail.update_scene(data, camera=close, scene_option=option)
                                board.paste(Image.fromarray(detail.render()), (x + 388, 205))
                                draw.text(
                                    (x + 394, 210),
                                    "WING DETAIL",
                                    font=font(14),
                                    fill="#e6e1db",
                                )
                        else:
                            draw.text(
                                (x + 28, 320), "EPISODE ENDED", font=font(32), fill="#db705d"
                            )
                            draw.text(
                                (x + 28, 371),
                                f"Physical failure at {case['first_failure_seconds']:.3f} s",
                                font=font(23),
                                fill="#e6e1db",
                            )
                            draw.text(
                                (x + 28, 410),
                                "No resets or assisted recovery",
                                font=font(23),
                                fill="#aab3b8",
                            )
                        at = min(step, len(capture["time"]) - 1)
                        failed_now = (
                            case["first_failure_seconds"] is not None
                            and t >= case["first_failure_seconds"]
                        )
                        status = "FAILED" if failed_now else "AIRBORNE"
                        draw.text(
                            (x + 12, 670),
                            f"{status}  |  ALTITUDE {heights[col][at]:.1f} mm",
                            font=font(23),
                            fill="#db705d" if failed_now else colors[col],
                        )
                        for chart, (label, values, maximum) in enumerate(
                            (
                                ("ALTITUDE / mm", heights[col], height_limit),
                                (
                                    "SPEED / mm/s · 100 ms average",
                                    np.linalg.norm(velocities[col], axis=1),
                                    speed_limit,
                                ),
                            )
                        ):
                            y = 715 + 145 * chart
                            draw.rectangle((x, y, x + 612, y + 132), fill="#202426")
                            draw.text((x + 12, y + 8), label, font=font(19), fill="#e6e1db")
                            draw.text(
                                (x + 495, y + 8),
                                f"0–{maximum:.0f}",
                                font=font(18),
                                fill="#aab3b8",
                            )
                            draw.line(
                                (x + 12, y + 113, x + 600, y + 113), fill="#596166", width=1
                            )
                            selected = np.arange(0, at + 1, 5)
                            points = [
                                (
                                    x + 12 + capture["time"][j] / 10 * 588,
                                    y + 113 - float(values[j]) / maximum * 74,
                                )
                                for j in selected
                            ]
                            if len(points) > 1:
                                draw.line(points, fill=colors[col], width=2)
                    draw.text(
                        (24, 1020),
                        "Full MaleCNS controls all 78 outputs  ·  Training the existing wing readout  ·  Same fly, physics and starts"
                        if training["recipe"].get("wing_readout_only")
                        else "Same fly / same flight dynamics / same starts  ·  Actor controls all 78 outputs  ·  Damped tracking cameras, shared chart scales",
                        font=font(22),
                        fill="#aab3b8",
                    )
                    writer.send(np.asarray(board))
                    frames += 1
            board = Image.new("RGB", size, "#101213")
            draw = ImageDraw.Draw(board)
            draw.text(
                (50, 55),
                "LEARNING PROGRESS / MEASURED OUTCOME"
                if original
                else "PPO CONTINUATION / MEASURED OUTCOME"
                if continued
                else "TEN-MINUTE PPO PILOT / MEASURED OUTCOME",
                font=font(39),
                fill="#ffc31f",
            )
            draw.text(
                (50, 120),
                "Airborne duration and motion error are both required; survival alone is not stable hover.",
                font=font(28),
                fill="#e6e1db",
            )
            for col, report in enumerate(reports):
                x = 50 + 625 * col
                draw.text((x, 220), titles[col], font=font(29), fill=colors[col])
                for row, case in enumerate(report["cases"]):
                    y = 310 + row * 115
                    draw.text(
                        (x, y),
                        f"Start {case['episode']}: {case['airborne_seconds']:.2f} s airborne",
                        font=font(25),
                        fill="#e6e1db",
                    )
                    draw.text(
                        (x, y + 42),
                        f"Speed RMS {case['velocity_rms_mm_s']:.2f} mm/s",
                        font=font(23),
                        fill="#aab3b8",
                    )
            draw.text(
                (50, 900),
                "Error metrics cover each pre-failure interval; shorter failed episodes are not comparable successes.",
                font=font(26),
                fill="#aab3b8",
            )
            for _ in range(150):
                writer.send(np.asarray(board))
                frames += 1
        finally:
            writer.close()
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "video_sha256": sha256(args.output),
        "fps": fps,
        "frames": frames,
        "duration_seconds": frames / fps,
        "dimensions": list(size),
        "playback_speed": 1,
        "wing_detail_inset": args.detail,
        "render_wall_seconds": time.perf_counter() - begin,
        "checkpoint_sha256": training["checkpoint_sha256"],
        "sections": sections,
        "camera": "0.2 s damped following, identical settings; plots share scales within case",
        "source_training_report_sha256": sha256(args.run / "report.json"),
        "original_comparison_report_sha256": sha256(args.original_run / "report.json")
        if original
        else None,
        "original_comparison_checkpoint_sha256": original["parent_checkpoint_sha256"]
        if original
        else None,
    }
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--detail", action="store_true")
    p.add_argument("--original-run", type=Path)
    record(p.parse_args())
