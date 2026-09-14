"""Seven synchronized round-trip views; optional learned actor/PID comparison."""

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


def record(source, output, reference=None):
    if output.exists():
        raise FileExistsError("Preserve prior reference videos")
    started = time.perf_counter()
    report = json.loads((source / "report.json").read_text())
    assert sha256(source / "capture.npz") == report["capture_sha256"]
    assert sha256(source / "model.mjb") == report["model_sha256"]
    # NpzFile decompresses an array on every key lookup. Load each needed array
    # once; seven panels must share memory rather than decompress per frame.
    with np.load(source / "capture.npz") as archive:
        states = {
            key: archive[key] for key in ("time", "qpos", "qvel", "act", "ctrl", "target")
        }
    learned = "checkpoint_sha256" in report
    reference_report = None
    if learned and reference is None:
        raise ValueError("Learned review requires the preserved PID reference capture")
    if reference is not None:
        if not learned:
            raise ValueError("Additional reference is for learned reviews only")
        reference_report = json.loads((reference / "report.json").read_text())
        assert sha256(reference / "capture.npz") == reference_report["capture_sha256"]
        assert reference_report["model_sha256"] == report["model_sha256"]
        with np.load(reference / "capture.npz") as archive:
            np.testing.assert_allclose(states["time"], archive["time"])
            for key, value in states.items():
                if key != "time":
                    states[key] = np.concatenate((value, archive[key][:, 6:7]), axis=1)
        report["cases"].append(reference_report["cases"][6])
        report["origins_cm"].append(reference_report["origins_cm"][6])
    camera_targets = np.asarray(report["origins_cm"]).copy()
    chart_range = 2.0
    if learned:
        chart_range = max(
            2.0,
            float(
                np.ceil(
                    max(
                        np.abs(
                            (
                                states["qpos"][:, i, case["axis"]]
                                - report["origins_cm"][i][case["axis"]]
                            )
                            * 10
                        ).max()
                        for i, case in enumerate(report["cases"])
                    )
                )
            ),
        )
    model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
    data = mujoco.MjData(model)
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -22, 1.15
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1600, 1000),
        fps=50,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    labels = (
        "LEFT / RIGHT",
        "RIGHT / LEFT",
        "FORWARD / BACK",
        "BACK / FORWARD",
        "UP / DOWN",
        "DOWN / UP",
        "STATIONARY HOVER",
    )
    if learned:
        labels += ("PID REFERENCE / HOVER",)
    frame_count = 0
    try:
        with mujoco.Renderer(model, height=260, width=384) as renderer:
            for frame, step in enumerate(range(0, len(states["time"]), 10)):
                board = Image.new("RGB", (1600, 1000), "#111519")
                draw = ImageDraw.Draw(board)
                draw.text(
                    (16, 12),
                    "LEARNED FLIGHT / ROUND-TRIP TEST"
                    if learned
                    else "ROUND TRIPS / RETURN TO START",
                    font=font(28),
                    fill="#ffc31f",
                )
                draw.text(
                    (16, 52),
                    (
                        "Development trial / One MaleCNS actor / Six routes + hover / PID hover reference at right / 1x"
                        if learned
                        else "PID reference / Wing-driven flight / No learned actor / 1x playback"
                    ),
                    font=font(20),
                    fill="#e6e1db",
                )
                t = float(states["time"][step])
                for i, case in enumerate(report["cases"]):
                    x, y = 8 + (i % 4) * 400, 90 + (i // 4) * 450
                    draw.rectangle((x, y, x + 384, y + 432), fill="#1b2025")
                    draw.text((x + 8, y + 8), labels[i], font=font(21), fill="#ffc31f")
                    for key in ("qpos", "qvel", "act", "ctrl"):
                        getattr(data, key)[:] = states[key][step, i]
                    data.time = t
                    mujoco.mj_forward(model, data)
                    origin = np.asarray(report["origins_cm"][i])
                    if learned:
                        camera_targets[i] += 0.08 * (
                            states["qpos"][step, i, :3] - camera_targets[i]
                        )
                        camera.lookat[:] = camera_targets[i]
                        camera.distance = 1.4
                    else:
                        camera.lookat[:] = origin
                    renderer.update_scene(data, camera=camera, scene_option=option)
                    scene = renderer.scene
                    for point, size, color in (
                        (origin, 0.012, (0.85, 0.85, 0.85, 0.55)),
                        (states["target"][step, i], 0.018, (1.0, 0.72, 0.07, 0.65)),
                    ):
                        geom = scene.geoms[scene.ngeom]
                        mujoco.mjv_initGeom(
                            geom,
                            mujoco.mjtGeom.mjGEOM_SPHERE,
                            np.array([size] * 3),
                            point,
                            np.eye(3).flatten(),
                            np.array(color),
                        )
                        scene.ngeom += 1
                    board.paste(Image.fromarray(renderer.render()), (x, y + 38))
                    axis = case["axis"]
                    history = np.arange(0, step + 1, 10)
                    desired = (states["target"][history, i, axis] - origin[axis]) * 10
                    actual = (states["qpos"][history, i, axis] - origin[axis]) * 10
                    chart_y = y + 338
                    draw.line((x + 8, chart_y, x + 376, chart_y), fill="#646c73")
                    for values, color in ((desired, "#ffc31f"), (actual, "#70a88a")):
                        points = list(
                            zip(
                                x + 8 + 368 * states["time"][history] / 12,
                                chart_y - 40 * np.clip(values / chart_range, -1, 1),
                            )
                        )
                        if len(points) > 1:
                            draw.line(points, fill=color, width=2)
                    error = (
                        np.linalg.norm(states["qpos"][step, i, :3] - states["target"][step, i])
                        * 10
                    )
                    home = np.linalg.norm(states["qpos"][step, i, :3] - origin) * 10
                    draw.text(
                        (x + 8, y + 382),
                        f"Target error {error:.2f} mm",
                        font=font(17),
                        fill="#e6e1db",
                    )
                    draw.text(
                        (x + 8, y + 408),
                        f"From start {home:.2f} mm",
                        font=font(17),
                        fill="#a8b0b5",
                    )
                if not learned:
                    x, y = 1216, 570
                    for j, (text, color) in enumerate(
                        (
                            ("ONE PHYSICAL FLY", "#ffc31f"),
                            ("Same body in every world", "#e6e1db"),
                            ("Targets: +/-1.5 mm", "#e6e1db"),
                            ("Out / reverse / return / hold", "#e6e1db"),
                            ("Amber: requested position", "#ffc31f"),
                            ("Green: actual axis position", "#70a88a"),
                            ("White marker: starting point", "#e6e1db"),
                            ("1,000 Hz physics", "#a8b0b5"),
                            ("500 Hz wing commands", "#a8b0b5"),
                            ("Reference only; no RL updates", "#a8b0b5"),
                        )
                    ):
                        draw.text((x, y + 32 * j), text, font=font(17), fill=color)
                draw.text(
                    (16, 968),
                    (
                        f"t={t:.2f}/12 s | Amber: target / Green: actual | Axis scale +/-{chart_range:g} mm | Damped following cameras | Return, then hold"
                        if learned
                        else f"t = {t:.2f} / 12.00 s   |   Fixed observer cameras   |   Final four seconds hold the original position"
                    ),
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
        "source_report_sha256": sha256(source / "report.json"),
        "capture_sha256": report["capture_sha256"],
        "model_sha256": report["model_sha256"],
        "frames": frame_count,
        "fps": 50,
        "size": [1600, 1000],
        "duration_s": frame_count / 50,
        "playback_multiplier": 1,
        "render_seconds": time.perf_counter() - started,
        "sha256": sha256(output),
        "controller": "same learned MaleCNS actor in seven worlds; PID hover in eighth"
        if learned
        else "PID reference only",
        "checkpoint_sha256": report.get("checkpoint_sha256"),
        "reference_capture_sha256": reference_report["capture_sha256"]
        if reference_report
        else None,
        "axis_chart_range_mm": chart_range,
        "actor_training": False,
        "observer_only_markers": True,
        "camera": {
            "azimuth": 135,
            "elevation": -22,
            "distance_cm": 1.4 if learned else 1.15,
            "mode": "damped following, alpha .08 at 50 fps" if learned else "fixed",
        },
        "case_order": labels,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--reference", type=Path)
    args = p.parse_args()
    record(args.source, args.output, args.reference)
