"""Record genuine rollouts and render synchronized observer cameras."""

import hashlib
import json
import os
import sys
import sysconfig
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .bodies import CONTROL_DT, ROOT
from .evaluate import lane_command, make_case, rollout
from .train import load_checkpoint


def model_hash(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, buffer=buffer)
    return hashlib.sha256(buffer.tobytes()).hexdigest()


def font(size):
    for path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def record(
    checkpoint,
    output,
    cases="short_fl,unseen_pair,unseen_weak,short_steps",
    seconds=12,
    fps=25,
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    net, saved = load_checkpoint(checkpoint)
    records = []
    count = 0
    trajectory_dir = ROOT / "outputs/locomotion/recordings" / output.stem
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    captions = {
        "short_fl": "Front-left calf: 70% remaining",
        "unseen_pair": "Two shortened calves: 75% and 65% remaining",
        "unseen_weak": "Front-right thigh torque drops to 25% at t = 3 s",
        "short_steps": "Shortened calf over physical 4 / 6 / 4 cm steps",
        "unseen_steps": "Shortened calf over physical 6 / 9 / 6 cm steps",
        "missing_calf": "Front-left calf and its joint removed",
    }
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1280, 720),
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    title_font, body_font = font(27), font(20)
    try:
        for case in cases.split(","):
            result, frames, model = rollout(
                net, case, trials=1, seconds=seconds, capture=True
            )
            result["model_mjb_sha256"] = model_hash(model)
            records.append(result)
            data = mujoco.MjData(model)
            option = mujoco.MjvOption()
            option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
            with (
                mujoco.Renderer(model, height=480, width=854) as big,
                mujoco.Renderer(model, height=240, width=426) as small,
            ):
                for frame_no in range(round(seconds * fps)):
                    k = min(round(frame_no / fps / CONTROL_DT), len(frames) - 1)
                    state = frames[k]
                    data.qpos[:] = state["qpos"]
                    data.qvel[:] = state["qvel"]
                    data.time = state["time"]
                    mujoco.mj_forward(model, data)
                    follow = mujoco.MjvCamera()
                    follow.lookat[:] = data.qpos[:3]
                    follow.distance, follow.azimuth, follow.elevation = 1.6, 50, -18
                    big.update_scene(data, camera=follow, scene_option=option)
                    canvas = Image.new("RGB", (1280, 720), (14, 23, 30))
                    canvas.paste(Image.fromarray(big.render()), (0, 70))
                    small.update_scene(data, camera="head", scene_option=option)
                    canvas.paste(Image.fromarray(small.render()), (854, 70))
                    overview = mujoco.MjvCamera()
                    overview.lookat[:] = [3.0, 0, 0.1]
                    overview.distance, overview.azimuth, overview.elevation = (
                        7.4,
                        90,
                        -58,
                    )
                    small.update_scene(data, camera=overview, scene_option=option)
                    canvas.paste(Image.fromarray(small.render()), (854, 310))
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (24, 20),
                        "ADAPTIVE DOG  /  "
                        + captions.get(case, case.replace("_", " ")),
                        font=title_font,
                        fill="white",
                    )
                    draw.text(
                        (20, 80),
                        "FOLLOW / PHYSICAL ROLLOUT",
                        font=body_font,
                        fill="white",
                    )
                    draw.text(
                        (866, 80),
                        "HEAD / OBSERVER CAMERA",
                        font=body_font,
                        fill="white",
                    )
                    draw.text(
                        (866, 320), "COURSE OVERVIEW", font=body_font, fill="white"
                    )
                    draw.text(
                        (24, 575),
                        f"{'REACTIVE' if saved['mode'] == 'blind' else saved['mode'].upper()} POLICY   |   {saved['cumulative_training_seconds']:.1f} s total training",
                        font=title_font,
                        fill=(105, 215, 199),
                    )
                    draw.text(
                        (24, 620),
                        f"t = {state['time']:.2f} s    x = {data.qpos[0]:.2f} m    peak current torque = {np.max(np.abs(state['torque'])):.1f} Nm",
                        font=body_font,
                        fill="white",
                    )
                    status = "5 m COMPLETED" if state["completed"] else "RUNNING"
                    if not state["alive"]:
                        status = "TRIAL FAILED — NO RESET"
                    draw.text(
                        (24, 660),
                        f"{status}  |  1x playback  |  Learned locomotion; scripted lane commands",
                        font=body_font,
                        fill="white",
                    )
                    writer.send(np.asarray(canvas))
                    count += 1
            np.savez_compressed(
                trajectory_dir / (case + ".npz"),
                **{key: np.asarray([f[key] for f in frames]) for key in frames[0]},
            )
            print(
                json.dumps({"recorded": case, "result": result["completed_5m"]}),
                flush=True,
            )
    finally:
        writer.close()
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "contact_profile": "firm: solref=.006 1; solimp=.95 .99 .001",
        "policy_training_seconds": saved["cumulative_training_seconds"],
        "fps": fps,
        "frames": count,
        "duration_seconds": count / fps,
        "cases": records,
        "camera_pixels_used_by_policy": False,
        "navigation": "scripted lane follower with ideal localization",
        "physics": "CPU MuJoCo through mjbatch, no pose control; recorded states replayed for rendering",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(left, right, output, case="short_steps", seconds=12, fps=25):
    """Matched initial conditions and cameras, including unsuccessful rollouts."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    runs = []
    directory = ROOT / "outputs/locomotion/recordings" / output.stem
    directory.mkdir(parents=True, exist_ok=True)
    for i, checkpoint in enumerate((left, right)):
        net, saved = load_checkpoint(checkpoint)
        outcome, frames, model = rollout(
            net, case, trials=1, seconds=seconds, capture=True
        )
        outcome["model_mjb_sha256"] = model_hash(model)
        np.savez_compressed(
            directory / f"policy_{i}.npz",
            **{key: np.asarray([f[key] for f in frames]) for key in frames[0]},
        )
        runs.append((saved, outcome, frames, model, mujoco.MjData(model)))
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1280, 720),
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    renderers = [mujoco.Renderer(r[3], height=480, width=640) for r in runs]
    try:
        for i in range(round(seconds * fps)):
            canvas = Image.new("RGB", (1280, 720), (14, 23, 30))
            draw = ImageDraw.Draw(canvas)
            title = (
                "MISSING FRONT-LEFT CALF"
                if case == "missing_calf"
                else "SHORTENED CALF / 4, 6, 4 CM STEPS"
            )
            draw.text(
                (24, 15),
                title + "  |  SAME START, SAME PHYSICS",
                font=font(25),
                fill="white",
            )
            for j, (saved, outcome, frames, model, data) in enumerate(runs):
                state = frames[min(round(i / fps / CONTROL_DT), len(frames) - 1)]
                data.qpos[:] = state["qpos"]
                data.qvel[:] = state["qvel"]
                data.time = state["time"]
                mujoco.mj_forward(model, data)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.qpos[:3]
                camera.distance, camera.azimuth, camera.elevation = 1.6, 50, -18
                renderers[j].update_scene(data, camera=camera, scene_option=option)
                canvas.paste(Image.fromarray(renderers[j].render()), (j * 640, 105))
                label = (
                    "REACTIVE" if saved["mode"] == "blind" else saved["mode"].upper()
                )
                draw.text(
                    (j * 640 + 24, 64),
                    f"{label} / {saved['cumulative_training_seconds']:.1f} s training",
                    font=font(26),
                    fill=(105, 215, 199),
                )
                status = (
                    "5 m COMPLETED"
                    if state["completed"]
                    else "5 m TARGET NOT YET REACHED"
                )
                if not state["alive"]:
                    status = "FAILED / NO RESET"
                draw.text(
                    (j * 640 + 24, 610),
                    f"x = {data.qpos[0]:.2f} m  |  {status}",
                    font=font(20),
                    fill="white",
                )
            draw.text(
                (24, 672),
                f"t = {state['time']:.2f} s  |  1x playback  |  Learned joint control; scripted lane commands",
                font=font(21),
                fill="white",
            )
            writer.send(np.asarray(canvas))
    finally:
        writer.close()
        for renderer in renderers:
            renderer.close()
    report = {
        "case": case,
        "seconds": seconds,
        "fps": fps,
        "contact_profile": "firm",
        "seed": 9137,
        "policies": [
            {
                "checkpoint": str(path),
                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "training_seconds": r[0]["cumulative_training_seconds"],
                "outcome": r[1],
            }
            for path, r in zip((left, right), runs, strict=True)
        ],
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def view(checkpoint, case="short_fl", seconds=0):
    if sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        env = os.environ.copy()
        paths = [sysconfig.get_config_var("LIBDIR"), str(Path(sys.base_prefix) / "lib")]
        paths += env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib").split(
            ":"
        )
        env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(
            dict.fromkeys(p for p in paths if p)
        )
        os.execve(
            sys.executable,
            [
                sys.executable,
                str(Path(sys.executable).parent / "mjpython"),
                "-m",
                "adaptive_locomotion.cli",
                *sys.argv[1:],
            ],
            env,
        )
    import mujoco.viewer

    torch.set_num_threads(1)
    net, _ = load_checkpoint(checkpoint)
    env = make_case(case, trials=1)
    group = env.groups[0]
    data = mujoco.MjData(group.model)
    start = time.perf_counter()
    try:
        with mujoco.viewer.launch_passive(
            group.model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 50, -18
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
            while viewer.is_running() and (
                not seconds or time.perf_counter() - start < seconds
            ):
                tick = time.perf_counter()
                lane_command(env)
                with torch.no_grad():
                    action, _ = net(
                        torch.as_tensor(env.obs()),
                        torch.as_tensor(env.context),
                        torch.as_tensor(env.history),
                    )
                env.step(action.numpy())
                data.qpos[:] = group.qpos[0]
                data.qvel[:] = group.qvel[0]
                data.ctrl[:] = group.ctrl[0]
                data.time = float(env.steps[0] * CONTROL_DT)
                mujoco.mj_forward(group.model, data)
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.sync()
                time.sleep(max(0, CONTROL_DT - (time.perf_counter() - tick)))
    finally:
        env.close()
