"""Live balance viewer and synchronized replay of measured physical rollouts."""

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
from .presentation import configure, damage_markers
from .record import font, model_hash
from .standing_env import StandingEnv
from .standing_evaluate import BODY_MAP, run_case
from .train import load_checkpoint

# Declared before inspecting any trained candidate; failures remain visible.
VIDEO_GROUPS = (
    ("WALK / BALANCE / WALK", (("healthy", "flat", True),)),
    (
        "LOCAL SUPPORT / ONE SHARED POLICY",
        tuple(
            ("healthy", s, False)
            for s in ("pads", "slope_x", "slope_y", "gap_fl", "gap_fr", "gap_rl")
        ),
    ),
    (
        "AGGRESSIVE TERRAIN / SAME POLICY",
        tuple(
            ("healthy", s, False)
            for s in (
                "pads_high",
                "pads_extreme",
                "slope_x_18",
                "slope_y_24",
                "steps_20",
                "steps_28",
            )
        ),
    ),
    (
        "DAMAGED BODIES / SAME POLICY",
        tuple(
            (b, "flat", False) for b in ("lower_fl", "lower_fr", "whole_rl", "whole_rr")
        ),
    ),
)


def ground_marks(scene, surface):
    """Observer-only paint gives walking/head cameras a fixed distance reference."""
    if surface != "flat":
        return
    for x in np.arange(-2, 12.01, 0.5):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("No room for observer ground marks")
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([0.003, 2, 0.0001]),
            np.array([x, 0, 0.0003]),
            np.eye(3).ravel(),
            np.array([0.25, 0.36, 0.40, 1], np.float32),
        )
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        scene.ngeom += 1


def camera(model, data, surface, kind="follow"):
    if kind == "head":
        return "head"
    cam = mujoco.MjvCamera()
    cam.lookat[:] = data.qpos[:3]
    cam.distance = 1.65 if kind == "follow" else 2.3
    cam.azimuth = -50 if surface.endswith("fl") else 50
    cam.elevation = -23 if kind == "follow" else -65
    if kind == "overview" and surface != "flat":
        cam.lookat[:] = [0, 0, 0.13]
    return cam


def record(checkpoint, output, physics_backend="mjbatch", fps=25, seed=9311):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    net, saved = load_checkpoint(checkpoint)
    trace_dir = ROOT / "outputs/locomotion/standing/recordings" / output.stem
    trace_dir.mkdir(parents=True, exist_ok=True)
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
    count, results = 0, []
    capture_seconds = 0.0
    started = time.perf_counter()
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    try:
        for heading, cases in VIDEO_GROUPS:
            runs = []
            for body, surface, transition in cases:
                capture_start = time.perf_counter()
                result, frames, model = run_case(
                    net,
                    body,
                    surface,
                    trials=1,
                    seconds=12 if transition else 10,
                    seed=seed,
                    physics_backend=physics_backend,
                    transition=transition,
                    capture=True,
                )
                result["model_mjb_sha256"] = model_hash(model)
                path = (
                    trace_dir
                    / f"{body}_{surface}_{'transition' if transition else 'stand'}.npz"
                )
                np.savez_compressed(
                    path, **{k: np.asarray([f[k] for f in frames]) for k in frames[0]}
                )
                result["trajectory_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                result["trajectory"] = str(path.relative_to(ROOT))
                results.append(result)
                capture_seconds += time.perf_counter() - capture_start
                configure(model)
                model.vis.global_.offwidth = 1920
                model.vis.global_.offheight = 1080
                runs.append((result, frames, model, mujoco.MjData(model)))
                print(
                    json.dumps(
                        {"captured": f"{body}/{surface}", "passed": result["passed"]}
                    ),
                    flush=True,
                )
            transition = len(runs) == 1
            width, height = (
                (1280, 760)
                if transition
                else (640, 430)
                if len(runs) == 6
                else (960, 430)
            )
            renderers = [
                mujoco.Renderer(r[2], height=height, width=width) for r in runs
            ]
            inset = (
                mujoco.Renderer(runs[0][2], height=380, width=640)
                if transition
                else None
            )
            try:
                duration = runs[0][0]["seconds"]
                for frame_no in range(round(duration * fps)):
                    canvas = Image.new("RGB", (1920, 1080), "#091219")
                    draw = ImageDraw.Draw(canvas)
                    draw.text((24, 16), heading, font=font(34), fill="#91d8d0")
                    for i, ((result, frames, model, data), renderer) in enumerate(
                        zip(runs, renderers, strict=True)
                    ):
                        state = frames[
                            min(round(frame_no / fps / CONTROL_DT), len(frames) - 1)
                        ]
                        data.qpos[:], data.qvel[:], data.ctrl[:] = (
                            state["qpos"],
                            state["qvel"],
                            state["ctrl"],
                        )
                        data.time = state["time"]
                        mujoco.mj_forward(model, data)
                        renderer.update_scene(
                            data,
                            camera=camera(model, data, result["surface"]),
                            scene_option=option,
                        )
                        ground_marks(renderer.scene, result["surface"])
                        damage_markers(
                            renderer.scene, model, data, BODY_MAP[result["body"]]
                        )
                        cols = 3 if len(runs) == 6 else 2
                        x, y = (
                            (0, 110)
                            if transition
                            else ((i % cols) * width, (i // cols) * 485 + 100)
                        )
                        canvas.paste(Image.fromarray(renderer.render()), (x, y))
                        title = result["surface"].upper().replace("_", " ")
                        if result["body"] != "healthy":
                            title = result["body"].upper().replace("_", " ") + " / FLAT"
                        draw.text((x + 16, y - 35), title, font=font(24), fill="white")
                        live_drift = np.linalg.norm(
                            data.qpos[:2] - state["hold_anchor"]
                        )
                        status = (
                            "WALK"
                            if np.linalg.norm(state["commands"]) > 0.05
                            else "BALANCE"
                        )
                        if not state["alive"]:
                            status = "FAILED / NO RESET"
                        elif not state["support_valid"]:
                            status = "UNINTENDED SUPPORT"
                        elif status == "BALANCE" and state["time"] >= (
                            5 if transition else 2
                        ):
                            tilt = np.rad2deg(
                                np.arccos(
                                    np.clip(
                                        data.xmat[model.body("base").id].reshape(3, 3)[
                                            2, 2
                                        ],
                                        -1,
                                        1,
                                    )
                                )
                            )
                            if live_drift > 0.15:
                                status = "OUTSIDE HOLD REGION"
                            elif tilt > 20:
                                status = "EXCESSIVE TILT"
                        draw.text(
                            (x + 16, y + height - 34),
                            f"{status}  |  drift {live_drift:.2f} m",
                            font=font(21),
                            fill="#91d8d0"
                            if status in ("WALK", "BALANCE")
                            else "#ff9b77",
                        )
                        for j, leg in enumerate(("FL", "FR", "RL", "RR")):
                            bx = x + width - 260 + j * 62
                            loaded = state["tip_forces"][j] > 1
                            draw.ellipse(
                                (bx, y + height - 29, bx + 10, y + height - 19),
                                fill="#91d8d0" if loaded else "#46535b",
                            )
                            draw.text(
                                (bx + 13, y + height - 33),
                                leg,
                                font=font(17),
                                fill="white",
                            )
                        if transition:
                            for j, kind in enumerate(("head", "overview")):
                                inset.update_scene(
                                    data,
                                    camera=camera(model, data, "flat", kind),
                                    scene_option=option,
                                )
                                ground_marks(inset.scene, "flat")
                                canvas.paste(
                                    Image.fromarray(inset.render()),
                                    (1280, 110 + j * 380),
                                )
                                draw.text(
                                    (1296, 122 + j * 380),
                                    kind.upper() + " / OBSERVER",
                                    font=font(22),
                                    fill="white",
                                )
                            draw.text(
                                (24, 920),
                                f"Command: {state['commands'][0]:.2f} m/s   |   3–8 s: hold position",
                                font=font(30),
                                fill="white",
                            )
                    draw.text(
                        (24, 1010),
                        f"t = {state['time']:.2f} s  |  1× real time  |  Learned joint control / scripted commands",
                        font=font(25),
                        fill="white",
                    )
                    draw.text(
                        (24, 1045),
                        f"Physics: {physics_backend}  |  Contact dots = measured load  |  RGB cameras are observer output",
                        font=font(21),
                        fill="#b6c8ce",
                    )
                    writer.send(np.asarray(canvas))
                    count += 1
            finally:
                for renderer in renderers:
                    renderer.close()
                if inset is not None:
                    inset.close()
        card = Image.new("RGB", (1920, 1080), "#091219")
        draw = ImageDraw.Draw(card)
        passed = sum(r["passed"] for r in results)
        draw.text(
            (40, 28),
            f"RECORDED TRIALS / {passed} OF {len(results)} PASS ALL BALANCE GATES",
            font=font(38),
            fill="#91d8d0",
        )
        for i, result in enumerate(results):
            col, row = divmod(i, 9)
            label = (
                result["body"].replace("_", " ")
                + " / "
                + result["surface"].replace("_", " ")
            )
            if result["transition"]:
                label += " / walk–stand–walk"
            draw.text(
                (40 + col * 960, 135 + row * 80),
                label.upper(),
                font=font(26),
                fill="white",
            )
            draw.text(
                (820 + col * 960, 135 + row * 80),
                "PASS" if result["passed"] else "FAIL",
                font=font(26),
                fill="#91d8d0" if result["passed"] else "#ff9b77",
            )
        draw.text(
            (40, 925),
            "Pass: upright, allowed supports, low drift and speed, tilt and physical limits respected.",
            font=font(27),
            fill="white",
        )
        draw.text(
            (40, 980),
            "One shared policy. Failures retained. See the full metrics for each trial.",
            font=font(27),
            fill="white",
        )
        for _ in range(3 * fps):
            writer.send(np.asarray(card))
            count += 1
    finally:
        writer.close()
    elapsed = time.perf_counter() - started
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "mode": saved["mode"],
        "seed": seed,
        "frames": count,
        "fps": fps,
        "duration_s": count / fps,
        "dimensions": [1920, 1080],
        "physics_backend": physics_backend,
        "cases": results,
        "capture_and_save_seconds": capture_seconds,
        "render_and_encode_seconds": elapsed - capture_seconds,
        "record_and_render_seconds": elapsed,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def view(checkpoint, surface="gap_fr", body="healthy", seconds=0, transition=False):
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
                "adaptive_locomotion.standing_cli",
                *sys.argv[1:],
            ],
            env,
        )
    import mujoco.viewer

    torch.set_num_threads(1)
    net, _ = load_checkpoint(checkpoint)
    net.eval()
    env = StandingEnv(
        1, 9311, cases=[(BODY_MAP[body], surface)], schedule=False, randomize=True
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
                env.set_commands(
                    [0.55 if transition and (t < 3 or t >= 8) else 0, 0, 0]
                )
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
                viewer.cam.lookat[:] = data.qpos[:3]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = (
                    1.9,
                    50,
                    -25,
                )
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    ground_marks(viewer.user_scn, surface)
                    damage_markers(viewer.user_scn, g.model, data, BODY_MAP[body])
                viewer.sync()
                time.sleep(max(0, CONTROL_DT - (time.perf_counter() - tick)))
    finally:
        env.close()
