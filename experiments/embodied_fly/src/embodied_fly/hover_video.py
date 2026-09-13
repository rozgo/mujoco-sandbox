"""Fixed-camera PID/PPO comparison from the same physical capture."""

import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.record import expand_floor_display, font, replay_clock


def record(source, output, previous=None):
    if output.exists():
        raise FileExistsError("Preserve existing video versions")
    started, provenance = time.perf_counter(), evidence()
    report = json.loads((source / "report.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
    floor_display = expand_floor_display(model)
    data = mujoco.MjData(model)
    reports = [report, report]
    sources = [source / "pid.npz", source / "hover.npz"]
    names = ("PID REFERENCE", "PPO / MALECNS ACTOR")
    cases = ("pid", "hover")
    title = "HOVER / REFERENCE AND LEARNING"
    if previous is not None:
        reports[0] = json.loads((previous / "report.json").read_text())
        if (
            reports[0]["physical_contract"] != report["physical_contract"]
            or reports[0]["model_sha256"] != report["model_sha256"]
        ):
            raise ValueError("Before/after comparison requires the same physical model")
        sources[0] = previous / "hover.npz"
        names = ("PPO / BEFORE", "PPO / TIGHTER VERTICAL REWARD")
        cases = ("hover", "hover")
        title = "HOVER / LEARNING BEFORE AND AFTER"
    captures = [np.load(path) for path in sources]
    indices, timestamps, control_hz = replay_clock(captures[0], report)
    if control_hz != 500 or report["physics_hz"] != 1000:
        raise ValueError("Comparison requires the accepted 1 kHz / 500 Hz clocks")
    if not np.array_equal(captures[0]["time"], captures[1]["time"]):
        raise ValueError("Comparison captures must share timestamps")
    np.testing.assert_array_equal(captures[0]["qpos"][0], captures[1]["qpos"][0])
    target = captures[0]["qpos"][0, :3].copy()
    target[2] = captures[0]["requested_height_cm"][0]
    # Both cameras share one fixed fit to all recorded root positions. No
    # smoothing, per-controller zoom, state interpolation or pose edits.
    positions = np.concatenate([s["qpos"][:, :3] for s in captures])
    low, high = positions.min(axis=0), positions.max(axis=0)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 135, -25
    camera.lookat[:] = (low + high) / 2
    camera.distance = max(1.6, float(np.linalg.norm(high - low)) * 2.8 + 1)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    detail_camera = mujoco.MjvCamera()
    detail_camera.azimuth, detail_camera.elevation = camera.azimuth, camera.elevation
    detail_camera.distance = 1.1
    thorax = model.body("walker/thorax").id
    errors = [(s["qpos"][:, 2] - target[2]) * 10 for s in captures]
    chart_range = max(1, float(np.ceil(max(np.abs(e).max() for e in errors))))
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1600, 900),
        fps=50,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        with (
            mujoco.Renderer(model, height=540, width=780) as renderer,
            mujoco.Renderer(model, height=180, width=260) as detail,
        ):
            for frame, step in enumerate(indices):
                board = Image.new("RGB", (1600, 900), "#111519")
                draw = ImageDraw.Draw(board)
                draw.text((24, 18), title, font=font(26), fill="#ffc31f")
                draw.text(
                    (24, 56),
                    "Same body / Same start / Same physics / 1x playback",
                    font=font(18),
                    fill="#e6e1db",
                )
                for i, (states, name, color) in enumerate(
                    zip(
                        captures,
                        names,
                        ("#ffc31f", "#70a88a"),
                    )
                ):
                    x = 10 + i * 800
                    for key, saved in [
                        ("qpos", "qpos"),
                        ("qvel", "qvel"),
                        ("act", "activation"),
                        ("ctrl", "ctrl"),
                    ]:
                        getattr(data, key)[:] = states[saved][step]
                    data.time = timestamps[step]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    board.paste(Image.fromarray(renderer.render()), (x, 140))
                    detail_camera.lookat[:] = data.xpos[thorax]
                    detail.update_scene(data, camera=detail_camera, scene_option=option)
                    board.paste(Image.fromarray(detail.render()), (x + 510, 160))
                    draw.text(
                        (x + 518, 346), "DETAIL / follows body", font=font(13), fill="#a8b0b5"
                    )
                    draw.text((x + 14, 100), name, font=font(23), fill=color)
                    error = float(np.linalg.norm(data.qpos[:3] - target) * 10)
                    draw.text(
                        (x + 14, 690),
                        f"Altitude {data.qpos[2] * 10:.2f} / {target[2] * 10:.2f} mm   |   Error {error:.2f} mm",
                        font=font(17),
                        fill="#e6e1db",
                    )
                    draw.text(
                        (x + 14, 721),
                        f"ALTITUDE ERROR / +/-{chart_range:g} mm",
                        font=font(14),
                        fill="#a8b0b5",
                    )
                    draw.rectangle(
                        (x + 14, 745, x + 765, 815), fill="#1b2025", outline="#33383c"
                    )
                    draw.line((x + 14, 780, x + 765, 780), fill="#646c73")
                    history = indices[: frame + 1]
                    points = list(
                        zip(
                            x + 14 + 751 * history / max(len(states["qpos"]) - 1, 1),
                            780 - 35 * errors[i][history] / chart_range,
                        )
                    )
                    if len(points) > 1:
                        draw.line(points, fill=color, width=2)
                    result = next(r for r in reports[i]["results"] if r["case"] == cases[i])
                    failed_at = result["first_failure_seconds"]
                    if failed_at is not None and data.time >= failed_at:
                        draw.text(
                            (x + 24, 630),
                            f"HOVER LOST AT {failed_at:.2f} s",
                            font=font(22),
                            fill="#ef715a",
                            stroke_width=1,
                            stroke_fill="#111519",
                        )
                draw.text(
                    (24, 848),
                    f"t = {timestamps[step]:.2f} s  |  1,000 Hz physics / 500 Hz control  |  Fixed matched cameras",
                    font=font(19),
                    fill="#a8b0b5",
                )
                writer.send(np.asarray(board))
    finally:
        writer.close()
    manifest = {
        "provenance": provenance,
        "completed_utc": utc_now(),
        "frames": len(indices),
        "fps": 50,
        "duration_s": len(indices) / 50,
        "playback_multiplier": 1,
        "size": [1600, 900],
        "source_report_sha256": sha256(source / "report.json"),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "model_sha256": report["model_sha256"],
        "state_sha256": {name: sha256(path) for name, path in zip(names, sources)},
        "panel_labels": names,
        "previous_checkpoint_sha256": reports[0]["checkpoint_sha256"] if previous else None,
        "previous_report_sha256": sha256(previous / "report.json") if previous else None,
        "camera": {
            "fixed": True,
            "shared": True,
            "lookat_cm": camera.lookat.tolist(),
            "distance_cm": camera.distance,
            "detail_inset": "same fixed magnification, follows measured thorax; main overview remains fixed",
        },
        "altitude_chart_range_mm": chart_range,
        "observer_only_floor_display": floor_display,
        "render_seconds": time.perf_counter() - started,
        "sha256": sha256(output),
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument(
        "--previous", type=Path, help="Prior matched hover capture for before/after"
    )
    a = p.parse_args()
    record(a.source, a.output, a.previous)
