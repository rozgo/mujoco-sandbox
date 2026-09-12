"""Measured moving-platform trajectories, matched film and native live viewer."""

import hashlib
import json
import os
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from .bodies import CONTROL_DT, ROOT
from .moving_env import MovingEnv
from .moving_evaluate import BODY_MAP, run_case
from .presentation import configure, damage_markers
from .record import font, model_hash
from .train import load_checkpoint

# Predetermined before the first trained candidate was evaluated.
VIDEO_MOTIONS = ("translate", "yaw", "heave", "rock", "combined")


def camera(model, data, kind="follow"):
    if kind == "head":
        return "head"
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.57]
    cam.distance = 2.7 if kind == "overview" else 2.3
    cam.azimuth, cam.elevation = (60, -65) if kind == "overview" else (50, -22)
    return cam


def view(checkpoint, motion="combined", body="healthy", seconds=0, relative=True):
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
                "adaptive_locomotion.moving_cli",
                *sys.argv[1:],
            ],
            env,
        )
    import mujoco.viewer

    torch.set_num_threads(1)
    net, _ = load_checkpoint(checkpoint)
    net.eval()
    env = MovingEnv(
        1,
        9411,
        cases=[(BODY_MAP[body], "moving")],
        motion=motion,
        relative=relative,
        schedule=False,
        randomize=True,
        substep_support=True,
    )
    g = env.groups[0]
    configure(g.model)
    data = mujoco.MjData(g.model)
    start = time.perf_counter()
    try:
        with mujoco.viewer.launch_passive(
            g.model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
            while viewer.is_running() and (
                not seconds or time.perf_counter() - start < seconds
            ):
                tick = time.perf_counter()
                t = env.steps[0] * CONTROL_DT
                env.set_commands([0, 0, 0])
                with torch.no_grad():
                    action, _ = net(
                        torch.as_tensor(env.obs()),
                        support=torch.as_tensor(env.support_obs()),
                    )
                env.step(action.numpy())
                data.qpos[:], data.qvel[:], data.ctrl[:] = (
                    g.qpos[0],
                    g.qvel[0],
                    g.ctrl[0],
                )
                data.time = t + CONTROL_DT
                mujoco.mj_forward(g.model, data)
                viewer.cam.lookat[:] = [0, 0, 0.57]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                    2.7,
                    50,
                    -25,
                )
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    damage_markers(viewer.user_scn, g.model, data, BODY_MAP[body])
                viewer.sync()
                time.sleep(max(0, CONTROL_DT - (time.perf_counter() - tick)))
    finally:
        env.close()


def record(
    checkpoint, baseline, output, seed=9411, physics_backend="mjbatch", replay_from=None
):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_hashes = [
        hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (baseline, checkpoint)
    ]
    variants = (
        ("Frozen standing", baseline, False),
        ("Relative inputs only", baseline, True),
        ("Trained on moving supports", checkpoint, True),
    )
    trace_dir = ROOT / "outputs/locomotion/moving/recordings" / output.stem
    trace_dir.mkdir(parents=True, exist_ok=True)
    replay = json.loads(Path(replay_from).read_text()) if replay_from else None
    if replay and (
        replay["checkpoint_hashes"] != checkpoint_hashes
        or replay["seed"] != seed
        or replay["physics_backend"] != physics_backend
    ):
        raise ValueError("Replay provenance mismatch")
    records, runs = [], {}
    capture_start = time.perf_counter()
    for motion in VIDEO_MOTIONS:
        for variant, (label, path, relative) in enumerate(variants):
            key = f"{motion}_{variant}"
            if replay:
                result = dict(next(r for r in replay["cases"] if r["key"] == key))
                trace_path, model_path = (
                    ROOT / result["trajectory"],
                    ROOT / result["model_binary"],
                )
                if (
                    hashlib.sha256(trace_path.read_bytes()).hexdigest()
                    != result["trajectory_sha256"]
                ):
                    raise ValueError("Trajectory changed")
                model = mujoco.MjModel.from_binary_path(str(model_path))
                if model_hash(model) != result["model_mjb_sha256"]:
                    raise ValueError("Model changed")
                with np.load(trace_path, allow_pickle=False) as data:
                    arrays = {k: data[k] for k in data.files}
                frames = [
                    {k: v[i] for k, v in arrays.items()}
                    for i in range(len(arrays["time"]))
                ]
            else:
                net, _ = load_checkpoint(path)
                result, frames, model = run_case(
                    net,
                    surface=motion,
                    trials=4,
                    seconds=10,
                    seed=seed,
                    physics_backend=physics_backend,
                    relative=relative,
                    capture=True,
                )
                trace_path, model_path = (
                    trace_dir / f"{key}.npz",
                    trace_dir / f"{key}.mjb",
                )
                np.savez_compressed(
                    trace_path,
                    **{k: np.asarray([f[k] for f in frames]) for k in frames[0]},
                )
                mujoco.mj_saveModel(model, str(model_path), None)
                result.update(
                    key=key,
                    label=label,
                    model_binary=str(model_path.relative_to(ROOT)),
                    trajectory=str(trace_path.relative_to(ROOT)),
                    trajectory_sha256=hashlib.sha256(
                        trace_path.read_bytes()
                    ).hexdigest(),
                    model_mjb_sha256=model_hash(model),
                )
            records.append(result)
            runs[key] = (result, frames, model)
            print(
                json.dumps(
                    {"captured": key, "strict_passes": result["passed"], "trials": 4}
                ),
                flush=True,
            )
    capture_seconds = time.perf_counter() - capture_start
    # Capture and save every physical rollout before rendering any footage.
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint_hashes": checkpoint_hashes,
        "seed": seed,
        "physics_backend": physics_backend,
        "cases": records,
        "capture_save_or_replay_seconds": capture_seconds,
        "replay_from": str(replay_from) if replay else None,
        "playback_speed": 1,
        "captured_trial": 0,
        "selection": "Five motion presets and trial 0 declared before candidate evaluation",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    fps, count = 25, 0
    started = time.perf_counter()
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1920, 1080),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for motion in VIDEO_MOTIONS:
            chapter = [runs[f"{motion}_{v}"] for v in range(3)]
            renderers, datas = [], []
            for _, _, model in chapter:
                configure(model)
                model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
                renderers.append(mujoco.Renderer(model, height=680, width=640))
                datas.append(mujoco.MjData(model))
            try:
                for frame_no in range(10 * fps):
                    canvas = Image.new("RGB", (1920, 1080), "#091219")
                    draw = ImageDraw.Draw(canvas)
                    draw.text(
                        (24, 20),
                        "BALANCE ON A MOVING WORLD",
                        font=font(36),
                        fill="#91d8d0",
                    )
                    draw.text(
                        (24, 70),
                        f"{motion.upper()}  |  matched physical motion commands  |  1x real time",
                        font=font(27),
                        fill="white",
                    )
                    for i, ((result, frames, model), renderer, data) in enumerate(
                        zip(chapter, renderers, datas, strict=True)
                    ):
                        state = frames[
                            min(round(frame_no / fps / CONTROL_DT), len(frames) - 1)
                        ]
                        data.qpos[:], data.qvel[:], data.ctrl[:] = (
                            state["qpos"],
                            state["qvel"],
                            state["ctrl"],
                        )
                        data.time = float(state["time"])
                        mujoco.mj_forward(model, data)
                        cam = camera(model, data)
                        cam.lookat[:] = data.qpos[:3] - [0, 0, 0.12]
                        cam.distance = 1.9
                        renderer.update_scene(data, camera=cam, scene_option=option)
                        x = i * 640
                        canvas.paste(Image.fromarray(renderer.render()), (x, 190))
                        draw.text(
                            (x + 18, 133), variants[i][0], font=font(25), fill="white"
                        )
                        draw.text(
                            (x + 18, 166),
                            "Original weights" if i < 2 else "One shared policy",
                            font=font(20),
                            fill="#a9b9c1",
                        )
                        draw.text(
                            (x + 18, 888),
                            f"Drift on deck: {100 * state['relative_drift_m']:.1f} cm",
                            font=font(25),
                            fill="#91d8d0",
                        )
                        status = "UPRIGHT" if state["alive"] else "FAILED HOLD"
                        if not state["support_valid"]:
                            status += " / LINK CONTACT"
                        draw.text(
                            (x + 18, 930),
                            status,
                            font=font(22),
                            fill="#91d8d0" if state["support_valid"] else "#ffa378",
                        )
                        for leg, force in enumerate(state["tip_forces"]):
                            color = "#91d8d0" if force > 1 else "#45535d"
                            px = x + 20 + leg * 100
                            draw.ellipse((px, 978, px + 16, 994), fill=color)
                            draw.text(
                                (px + 25, 973),
                                ("FL", "FR", "RL", "RR")[leg],
                                font=font(22),
                                fill="white",
                            )
                    draw.text(
                        (24, 1032),
                        f"{(frame_no + 1) / fps:.2f} s  |  MuJoCo contacts + torque-limited joints  |  cameras are observer views",
                        font=font(22),
                        fill="#a9b9c1",
                    )
                    writer.send(np.asarray(canvas))
                    count += 1
            finally:
                for renderer in renderers:
                    renderer.close()
        # A repeated, explicitly labeled replay reveals the same final physical
        # trace in overview and onboard cameras, synchronized to one timestamp.
        result, frames, model = runs["combined_2"]
        data = mujoco.MjData(model)
        with (
            mujoco.Renderer(model, height=900, width=1280) as large,
            mujoco.Renderer(model, height=430, width=640) as small,
        ):
            for frame_no in range(10 * fps):
                state = frames[min(round(frame_no / fps / CONTROL_DT), len(frames) - 1)]
                data.qpos[:], data.qvel[:], data.ctrl[:] = (
                    state["qpos"],
                    state["qvel"],
                    state["ctrl"],
                )
                data.time = float(state["time"])
                mujoco.mj_forward(model, data)
                canvas = Image.new("RGB", (1920, 1080), "#091219")
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (24, 20), "ONE POLICY / THREE VIEWS", font=font(36), fill="#91d8d0"
                )
                draw.text(
                    (24, 72),
                    "Combined motion replay  |  synchronized cameras  |  1x",
                    font=font(25),
                    fill="white",
                )
                large.update_scene(
                    data, camera=camera(model, data), scene_option=option
                )
                canvas.paste(Image.fromarray(large.render()), (0, 130))
                for kind, y in (("overview", 130), ("head", 600)):
                    small.update_scene(
                        data, camera=camera(model, data, kind), scene_option=option
                    )
                    canvas.paste(Image.fromarray(small.render()), (1280, y))
                    draw.text((1295, y + 12), kind.upper(), font=font(23), fill="white")
                draw.text(
                    (24, 1037),
                    f"{(frame_no + 1) / fps:.2f} s  |  Drift on deck: {state['relative_drift_m'] * 100:.1f} cm",
                    font=font(24),
                    fill="#91d8d0",
                )
                writer.send(np.asarray(canvas))
                count += 1
        card = Image.new("RGB", (1920, 1080), "#091219")
        draw = ImageDraw.Draw(card)
        draw.text(
            (80, 100),
            "LEARNING TO BALANCE WITH THE SUPPORT",
            font=font(44),
            fill="#91d8d0",
        )
        draw.text(
            (80, 210),
            "Same network, original robot, bounded physical actuators.",
            font=font(32),
            fill="white",
        )
        for i, (label, _, _) in enumerate(variants):
            total = sum(r["passed"] for r in records if r["key"].endswith(f"_{i}"))
            alive = sum(r["survived"] for r in records if r["key"].endswith(f"_{i}"))
            draw.text(
                (80, 340 + i * 95),
                f"{label}: {alive}/20 upright; {total}/20 strict passes",
                font=font(31),
                fill="white",
            )
        draw.text(
            (80, 750),
            "Four trials per motion. Trial 0 shown. All failures retained.",
            font=font(28),
            fill="#a9b9c1",
        )
        draw.text(
            (80, 820),
            "Ideal current platform-state sensing. No future motion input.",
            font=font(28),
            fill="#a9b9c1",
        )
        draw.text(
            (80, 900),
            "A controlled simulation experiment; no real-robot validation.",
            font=font(28),
            fill="#a9b9c1",
        )
        for _ in range(4 * fps):
            writer.send(np.asarray(card))
            count += 1
    finally:
        writer.close()
    report.update(
        render_encode_seconds=time.perf_counter() - started,
        frames=count,
        fps=fps,
        duration_s=count / fps,
        dimensions=[1920, 1080],
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
