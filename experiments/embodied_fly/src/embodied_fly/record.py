"""Replay captured states with source-derived labels and synchronized eye cameras."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from embodied_fly.brain import ACTIVITIES
from embodied_fly.neural_view import colorize
from embodied_fly.provenance import evidence, utc_now


def font(size):
    for path in (
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def record(source, case, output):
    started = time.perf_counter()
    if output.exists():
        raise FileExistsError("Choose a new video version; preserve the existing capture")
    run_evidence = evidence()
    teacher = (source / "manifest.json").exists()
    evaluation = json.loads((source / "report.json").read_text()) if not teacher else {}
    passive_mask = evaluation.get("walking_action_mask", False)
    case_result = next((r for r in evaluation.get("results", []) if r["case"] == case), None)
    title = (
        "REFERENCE TEACHER / inherited walking policy"
        if teacher
        else "MALECNS STUDENT / motor-learning diagnostic"
    )
    model = mujoco.MjModel.from_binary_path(str(source / "model.mjb"))
    data = mujoco.MjData(model)
    states = np.load(source / f"{case}.npz", allow_pickle=False)
    thorax = model.body("walker/thorax").id
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 135, -25, 0.95
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
    frame_count = 0
    with (
        mujoco.Renderer(model, height=700, width=1100) as renderer,
        mujoco.Renderer(model, height=180, width=230) as eye,
    ):
        for step in range(0, len(states["qpos"]), 10):
            # Replay only: exact saved physical poses, not a live controller.
            data.qpos[:] = states["qpos"][step]
            data.qvel[:] = states["qvel"][step]
            data.act[:] = states["activation"][step]
            data.ctrl[:] = states["ctrl"][step]
            data.time = step * 0.002
            mujoco.mj_forward(model, data)
            camera.lookat[:] = data.xpos[thorax]
            renderer.update_scene(data, camera=camera, scene_option=option)
            board = Image.new("RGB", (1600, 900), "#111519")
            board.paste(Image.fromarray(renderer.render()), (10, 100))
            draw = ImageDraw.Draw(board)
            if "neural_map" in states:
                stride = int(states["neural_map_stride"])
                values = states["neural_map"][step // stride].astype(np.float32)
                neural_image = Image.fromarray(colorize(values, states["neural_occupancy"]))
                draw.rectangle((806, 110, 1096, 446), fill="#111519", outline="#33383c")
                draw.text(
                    (818, 120), "CNS / learned latent state", font=font(15), fill="#e6e1db"
                )
                board.paste(
                    neural_image.resize((256, 256), Image.Resampling.NEAREST), (823, 149)
                )
                draw.text((818, 410), "teal −  /  amber +", font=font(14), fill="#a8b0b5")
                draw.text((818, 427), "Measured cell locations", font=font(13), fill="#a8b0b5")
            draw.text((24, 22), title, font=font(25), fill="#ffc31f")
            draw.text(
                (24, 60),
                f"{case.upper()}  |  physical capture  |  1x playback",
                font=font(19),
                fill="#e6e1db",
            )
            draw.text(
                (1125, 105), "EYE CAMERAS / observer only", font=font(16), fill="#e6e1db"
            )
            for i, side in enumerate(("left", "right")):
                eye.update_scene(data, camera=f"walker/eye_{side}", scene_option=option)
                board.paste(Image.fromarray(eye.render()), (1125 + i * 235, 140))
            if "utility" in states:
                draw.text((1125, 350), "LEARNED UTILITY SCORES", font=font(20), fill="#ffc31f")
                scores = states["utility"][step]
                for i, (name, score) in enumerate(zip(ACTIVITIES, scores)):
                    y = 390 + i * 53
                    draw.text((1125, y), name, font=font(18), fill="#e6e1db")
                    draw.rectangle((1250, y + 3, 1565, y + 21), fill="#33383c")
                    draw.rectangle(
                        (1250, y + 3, 1250 + 315 * float(score), y + 21), fill="#e88107"
                    )
                draw.text(
                    (1125, 730), "Rest/explore warm start only", font=font(16), fill="#a8b0b5"
                )
                draw.text(
                    (1125, 755),
                    "59 active / 19 passive channels"
                    if passive_mask
                    else "All 78 actuator channels active",
                    font=font(16),
                    fill="#a8b0b5",
                )
                if case_result:
                    draw.text(
                        (1125, 790),
                        "Posture: " + ("STABLE" if case_result["stable"] else "UNSTABLE"),
                        font=font(18),
                        fill="#a8b0b5",
                    )
                    draw.text(
                        (1125, 818),
                        "Task test: " + ("PASS" if case_result["success"] else "FAIL"),
                        font=font(18),
                        fill="#70a88a" if case_result["success"] else "#ce6654",
                    )
            else:
                draw.text(
                    (1125, 380), "Demonstration for imitation", font=font(19), fill="#ffc31f"
                )
                draw.text(
                    (1125, 430),
                    "No MaleCNS learner in this clip",
                    font=font(16),
                    fill="#a8b0b5",
                )
            draw.text(
                (24, 840),
                f"t = {data.time:.2f} s   |   5 kHz physics / 500 Hz control   |   complete anatomy",
                font=font(20),
                fill="#a8b0b5",
            )
            writer.send(np.asarray(board))
            frame_count += 1
    writer.close()
    manifest = {
        "provenance": run_evidence,
        "completed_utc": utc_now(),
        "frames": frame_count,
        "fps": 50,
        "duration_s": frame_count / 50,
        "playback_multiplier": 1,
        "teacher": teacher,
        "walking_action_mask": passive_mask,
        "checkpoint_sha256": evaluation.get("checkpoint_sha256"),
        "neural_view": evaluation.get("neural_view"),
        "case": case,
        "state_sha256": hashlib.sha256((source / f"{case}.npz").read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256((source / "model.mjb").read_bytes()).hexdigest(),
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "render_seconds": time.perf_counter() - started,
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("case")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    record(args.source, args.case, args.output)
