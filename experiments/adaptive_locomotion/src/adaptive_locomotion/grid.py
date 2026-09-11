"""Render all trained body families together using one frozen policy checkpoint."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .bodies import CONTROL_DT, ROOT
from .evaluate import CASES, make_case, rollout
from .paired import training_pairs
from .presentation import camera_azimuth, configure, damage_markers
from .record import font, model_hash
from .train import load_checkpoint

GRID_CASES = (
    ("healthy", "HEALTHY", "Four intact legs", 0, 0),
    ("weak_fr_thigh_60", "MOTOR FAULT", "FR thigh: 60% torque after 3 s", 0, 1),
    *(
        (f"short_{leg.lower()}", f"SINGLE / {leg}", "Calf: 70% remaining", 1, i)
        for i, leg in enumerate(("FL", "FR", "RL", "RR"))
    ),
    *(
        (body.name, f"MILD PAIR / {label}", "Calves: 90% / 85% remaining", 2, i)
        for i, (body, label) in enumerate(
            zip(
                training_pairs("mild"),
                ("FL + RR", "FR + RL", "FRONT", "REAR"),
                strict=True,
            )
        )
    ),
    *(
        (body.name, f"STRONG PAIR / {label}", "Calves: 75% / 65% remaining", 3, i)
        for i, (body, label) in enumerate(
            zip(
                training_pairs("hard"),
                ("FL + RR", "FR + RL", "FRONT", "REAR"),
                strict=True,
            )
        )
    ),
)


def grid(
    checkpoint,
    output,
    seconds=12,
    fps=25,
    seed=9137,
    family="partial",
    presentation="original",
    replay_from=None,
    label=None,
    timestep=0.002,
):
    """Capture physical runs, then replay a common timestamp in every panel."""
    if family not in ("partial", "limb_loss"):
        raise ValueError(family)
    limb = family == "limb_loss"
    cases = GRID_CASES
    if limb:
        cases = (
            ("healthy", "HEALTHY", "Four intact legs", 0, 0),
            *(
                (f"{kind}_{leg.lower()}", f"{label} / {leg}", detail, row, i)
                for kind, label, detail, row in (
                    (
                        "lower",
                        "LOWER LEG REMOVED",
                        "Entire calf + foot absent / 11 actuators",
                        1,
                    ),
                    (
                        "whole",
                        "ENTIRE LEG REMOVED",
                        "Hip + thigh + calf absent / 9 actuators",
                        2,
                    ),
                )
                for i, leg in enumerate(("FL", "FR", "RL", "RR"))
            ),
        )
    start = time.perf_counter()
    checkpoint, output = Path(checkpoint), Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    net, saved = load_checkpoint(checkpoint)
    directory = ROOT / "outputs/locomotion/recordings" / output.stem
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "capture.json"
    source_directory = Path(replay_from) if replay_from else directory
    source_manifest = source_directory / "capture.json"
    if replay_from and not source_manifest.exists():
        raise ValueError("Replay source must contain capture.json")
    cache = (
        json.loads(source_manifest.read_text()) if source_manifest.exists() else None
    )
    if presentation not in ("original", "damage"):
        raise ValueError(presentation)
    if cache and (
        cache["checkpoint_sha256"] != digest
        or cache["seed"] != seed
        or cache["seconds"] != seconds
        or cache.get("family", "partial") != family
        or cache.get("physics_timestep_s", 0.002) != timestep
    ):
        raise ValueError(
            "Existing grid capture uses different weights, seed or duration; use another output name"
        )
    prior_path = ROOT / "previews/locomotion/paired_damage.json"
    prior = json.loads(prior_path.read_text()) if prior_path.exists() else None
    entries, runs = [], []
    for case, title, subtitle, row, col in cases:
        result, states, source = None, None, None
        if cache:
            entry = next((c for c in cache["cases"] if c["case"] == case), None)
            if entry:
                source = source_directory / f"{case}.npz"
                if (
                    hashlib.sha256(source.read_bytes()).hexdigest()
                    != entry["trajectory_sha256"]
                ):
                    raise ValueError("Cached grid trajectory hash changed")
                result = entry["outcome"]
        if (
            result is None
            and prior
            and prior["checkpoint_sha256"] == digest
            and seed == 9137
        ):
            result = next(
                (
                    r
                    for r in prior["cases"]
                    if r["case"] == case and r["seconds"] == seconds
                ),
                None,
            )
            if result:
                source = (
                    ROOT / "outputs/locomotion/recordings/paired_damage" / f"{case}.npz"
                )
                if not source.exists():
                    result = None
        if result is not None:
            env = make_case(case, trials=1, seed=seed, timestep=timestep)
            model = env.groups[0].model
            env.close()
            if model_hash(model) != result["model_mjb_sha256"]:
                raise ValueError(
                    "Cached trajectory model differs from the current model"
                )
            with np.load(source) as capture:
                states = {k: capture[k] for k in capture.files}
            reuse = True
        else:
            if replay_from:
                raise ValueError(
                    f"Replay source is missing {case}; refusing a new rollout"
                )
            result, frames, model = rollout(
                net,
                case,
                trials=1,
                seed=seed,
                seconds=seconds,
                capture=True,
                timestep=timestep,
            )
            result["model_mjb_sha256"] = model_hash(model)
            states = {key: np.asarray([f[key] for f in frames]) for key in frames[0]}
            reuse = False
        if len(states["time"]) != round(seconds / CONTROL_DT):
            raise ValueError("Trajectory length differs from requested duration")
        np.testing.assert_allclose(
            states["time"],
            np.arange(1, len(states["time"]) + 1) * CONTROL_DT,
            atol=1e-10,
        )
        dest = directory / f"{case}.npz"
        if source != dest:
            np.savez_compressed(dest, **states)
        entry = {
            "case": case,
            "title": title,
            "subtitle": subtitle,
            "row": row,
            "column": col,
            "reused_saved_trajectory": reuse,
            "trajectory_sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "outcome": result,
        }
        entries.append(entry)
        runs.append((entry, states, model, mujoco.MjData(model)))
        print(
            json.dumps(
                {
                    "captured": case,
                    "valid": result["completed_with_allowed_support"],
                    "reused": reuse,
                }
            ),
            flush=True,
        )
    capture_seconds = time.perf_counter() - start
    manifest = {
        "family": family,
        "physics_timestep_s": timestep,
        "checkpoint_sha256": digest,
        "seed": seed,
        "seconds": seconds,
        "cases": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    render_models = {}
    if presentation == "damage":
        for entry, _, model, _ in runs:
            configure(model)
            render_models[entry["case"]] = model_hash(model)

    width, height = 3840, 2160
    tile_w, tile_h, gap, margin, top = 936, 472, 16, 24, 148
    if limb:
        tile_h = 632
    image_h = tile_h - 120
    renderers = [mujoco.Renderer(run[2], height=image_h, width=tile_w) for run in runs]
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    heading, body, small = font(32), font(26), font(24)
    camera = mujoco.MjvCamera()
    camera.distance, camera.azimuth, camera.elevation = 1.3, 50, -18
    temp = output.with_name(output.stem + ".rendering.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(temp),
        (width, height),
        fps=fps,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    render_start = time.perf_counter()
    try:
        for frame_no in range(round(seconds * fps)):
            k = min(round(frame_no / fps / CONTROL_DT), len(runs[0][1]["time"]) - 1)
            timestamp = float(runs[0][1]["time"][k])
            canvas = Image.new("RGB", (width, height), (9, 18, 25))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (28, 24),
                label
                or (
                    "ONE POLICY / COMPLETE LIMB LOSS"
                    if limb
                    else "ONE POLICY / FOURTEEN CONDITIONS"
                ),
                font=font(52),
                fill="white",
            )
            draw.text(
                (29, 89),
                "Healthy + four lower-leg removals + four whole-leg removals   |   All outcomes shown"
                if limb
                else "Synchronized MuJoCo rollouts   |   Every trained body variant   |   Representative motor fault",
                font=body,
                fill=(157, 180, 190),
            )
            for renderer, (entry, states, model, data) in zip(
                renderers, runs, strict=True
            ):
                assert abs(float(states["time"][k]) - timestamp) < 1e-10
                x, y = (
                    margin + entry["column"] * (tile_w + gap),
                    top + entry["row"] * (tile_h + gap),
                )
                data.qpos[:], data.qvel[:], data.time = (
                    states["qpos"][k],
                    states["qvel"][k],
                    timestamp,
                )
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3]
                camera.azimuth = (
                    camera_azimuth(CASES[entry["case"]][0])
                    if presentation == "damage"
                    else 50
                )
                renderer.update_scene(data, camera=camera, scene_option=option)
                if presentation == "damage":
                    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
                    damage_markers(renderer.scene, model, data, CASES[entry["case"]][0])
                draw.rectangle(
                    (x, y, x + tile_w - 1, y + tile_h - 1), fill=(17, 29, 38)
                )
                canvas.paste(Image.fromarray(renderer.render()), (x, y + 76))
                draw.text(
                    (x + 16, y + 8), entry["title"], font=heading, fill=(109, 224, 204)
                )
                subtitle = entry["subtitle"]
                if entry["case"] == "weak_fr_thigh_60":
                    subtitle = f"FR thigh: {states['strength'][k, 4] * 100:.0f}% torque  |  fault at 3 s"
                draw.text((x + 16, y + 46), subtitle, font=small, fill="white")
                status = "WALKING"
                color = (180, 198, 208)
                if not states["alive"][k]:
                    status, color = "FAILED / NO RESET", (255, 136, 116)
                elif not states["support_valid"][k]:
                    status, color = "INVALID SUPPORT", (255, 136, 116)
                elif states["completed"][k]:
                    status, color = "5 m COMPLETE", (109, 224, 204)
                elif limb and frame_no >= round(seconds * fps) - fps:
                    status, color = "INCOMPLETE", (255, 136, 116)
                draw.text(
                    (x + 16, y + tile_h - 36),
                    f"{data.qpos[0]:.2f} m  |  {status}",
                    font=small,
                    fill=color,
                )
                allowed = entry["outcome"]["allowed_support_geom_names"]
                names = entry["outcome"]["support_geom_names"]
                for i, leg in enumerate(("FL", "FR", "RL", "RR")):
                    name = next((n for n in allowed if n.startswith(leg)), None)
                    loaded = (
                        name is not None
                        and states["support_peak_forces_n"][k, names.index(name)] > 1
                    )
                    dx = x + 568 + i * 90
                    draw.ellipse(
                        (dx, y + tile_h - 31, dx + 14, y + tile_h - 17),
                        fill=(109, 224, 204) if loaded else (61, 76, 85),
                    )
                    draw.text((dx + 23, y + tile_h - 37), leg, font=small, fill="white")
                    if name is None:
                        draw.line(
                            (dx, y + tile_h - 31, dx + 14, y + tile_h - 17),
                            fill=(230, 140, 80),
                            width=2,
                        )
            # Two explanatory cards keep the four single/mild/strong columns aligned.
            for col in (2, 3):
                x = margin + col * (tile_w + gap)
                draw.rectangle(
                    (x, top, x + tile_w - 1, top + tile_h - 1), fill=(17, 29, 38)
                )
            x = margin + 2 * (tile_w + gap)
            draw.text(
                (x + 32, top + 26),
                "SHARED WEIGHTS",
                font=font(38),
                fill=(109, 224, 204),
            )
            draw.text(
                (x + 32, top + 96), "1 learned controller", font=font(52), fill="white"
            )
            draw.text(
                (x + 32, top + 165),
                "9 physical conditions" if limb else "14 physical conditions",
                font=font(44),
                fill="white",
            )
            draw.text(
                (x + 32, top + 241),
                f"{saved['cumulative_training_seconds'] / 60:.2f} min cumulative training",
                font=heading,
                fill=(157, 180, 190),
            )
            draw.text(
                (x + 32, top + 302),
                "Same checkpoint in every panel",
                font=heading,
                fill="white",
            )
            draw.text(
                (x + 32, top + 371),
                "Independent rollouts / shared simulation time",
                font=body,
                fill=(157, 180, 190),
            )
            x = margin + 3 * (tile_w + gap)
            draw.text(
                (x + 32, top + 26),
                "READING THE GRID",
                font=font(38),
                fill=(109, 224, 204),
            )
            lines = [
                "Orange = shortened calf",
                "FL / FR = front left / right",
                "RL / RR = rear left / right",
                "Percentages = remaining calf length",
                "Contact dots = measured foot support",
                "Following cameras / flat ground",
            ]
            if limb:
                lines = [
                    "Lower = everything below knee removed",
                    "Whole = entire leg chain removed",
                    "Orange sphere = damage location"
                    if presentation == "damage"
                    else "Orange = exposed upper-leg stump",
                    "Crossed dot = no supporting limb",
                    "FL / FR = front left / right",
                    "RL / RR = rear left / right",
                ]
                lx = margin + tile_w + gap
                draw.rectangle(
                    (lx, top, lx + tile_w - 1, top + tile_h - 1), fill=(17, 29, 38)
                )
                for j, line in enumerate(
                    [
                        "COMPLETE REMOVALS",
                        "One fixed checkpoint in all nine panels",
                        "No policy switching or online training",
                        "Physical contact / original torque caps",
                        "Failed attempts remain visible",
                        "Damage-side cameras / contact shadows"
                        if presentation == "damage"
                        else "Flat ground / scripted velocity commands",
                    ]
                ):
                    draw.text(
                        (lx + 32, top + 26 + j * 65), line, font=heading, fill="white"
                    )
            for j, line in enumerate(lines):
                draw.text(
                    (x + 32, top + 100 + j * 55), line, font=heading, fill="white"
                )
            draw.text(
                (28, 2120),
                f"t = {timestamp:05.2f} s    |    1x playback    |    Learned joint control; scripted lane commands    |    Cameras are observers",
                font=body,
                fill="white",
            )
            writer.send(np.asarray(canvas))
            if frame_no in (0, round(seconds * fps) // 2, round(seconds * fps) - 1):
                canvas.save(directory / f"frame_{frame_no:04d}.png")
            if frame_no % 50 == 0:
                print(json.dumps({"rendered_frame": frame_no}), flush=True)
    finally:
        writer.close()
        for renderer in renderers:
            renderer.close()
    temp.replace(output)
    report = {
        **manifest,
        "checkpoint": str(checkpoint.resolve().relative_to(ROOT)),
        "policy_training_seconds": saved["cumulative_training_seconds"],
        "new_training_seconds": 0,
        "resolution": [width, height],
        "fps": fps,
        "frames": round(seconds * fps),
        "playback_speed": 1,
        "capture_wall_seconds": capture_seconds,
        "render_wall_seconds": time.perf_counter() - render_start,
        "single_shared_policy": True,
        "camera_pixels_used_by_policy": False,
        "contact_indicator": "Allowed terminal force >1 N anywhere in the current 20 ms control window",
        "scope": "Healthy plus all eight single complete-removal conditions; fixed demonstrations, no claim of arbitrary limb-loss recovery"
        if limb
        else "All 13 geometry variants across mild/strong training, plus one representative 60% motor fault; not every randomized fault, command or initial condition",
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "render_source_commit": source_commit,
        "presentation": presentation,
        "label": label,
        "replayed_existing_trajectories": all(
            e["reused_saved_trajectory"] for e in entries
        ),
        "render_model_sha256": render_models,
        "damage_marker": "Observer-only sphere: 28 mm radius at the calf cut, 40 mm at the original hip attachment; no added collision or mass"
        if presentation == "damage"
        else None,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report
