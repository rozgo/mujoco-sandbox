"""Matched physical demonstrations of the three final training continuations."""

import argparse
import hashlib
import json
import time

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import CONTROL_DT, ROOT
from adaptive_locomotion.evaluate import CASES, make_case, rollout
from adaptive_locomotion.presentation import camera_azimuth, configure, damage_markers
from adaptive_locomotion.record import font, model_hash
from adaptive_locomotion.train import load_checkpoint

LABELS = ("mac_mps", "desktop_cpu", "desktop_cuda")
TITLES = (
    "M3 MAX / MPS TRAINING",
    "RYZEN 5950X / CPU TRAINING",
    "RTX 4090 / CUDA TRAINING",
)
CASES_SHOWN = ("healthy", "lower_fr", "whole_fr")


def main(replay=False):
    output = ROOT / "previews/locomotion/learner_backend_comparison.mp4"
    directory = ROOT / "outputs/locomotion/recordings/learner_backend_comparison"
    directory.mkdir(parents=True, exist_ok=True)
    prior = json.loads(output.with_suffix(".json").read_text()) if replay else None
    runs, entries = [], []
    start = time.perf_counter()
    for col, label in enumerate(LABELS):
        checkpoint = (
            ROOT / f"assets/locomotion/checkpoints/learner_{label}_90s_seed2.pt"
        )
        net, _ = load_checkpoint(checkpoint)
        for row, case in enumerate(CASES_SHOWN):
            path = directory / f"{label}_{case}.npz"
            if replay:
                entry = next(
                    e
                    for e in prior["cases"]
                    if e["label"] == label and e["case"] == case
                )
                assert (
                    entry["checkpoint_sha256"]
                    == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                )
                assert (
                    entry["trajectory_sha256"]
                    == hashlib.sha256(path.read_bytes()).hexdigest()
                )
                env = make_case(case, trials=1, seed=9143, timestep=0.0005)
                model = env.groups[0].model
                env.close()
                assert model_hash(model) == entry["model_mjb_sha256"]
                outcome = entry["outcome"]
                with np.load(path) as source:
                    states = {key: source[key] for key in source.files}
            else:
                outcome, frames, model = rollout(
                    net, case, trials=1, seed=9143, capture=True, timestep=0.0005
                )
                states = {
                    key: np.asarray([f[key] for f in frames]) for key in frames[0]
                }
                np.savez_compressed(path, **states)
            assert len(states["time"]) == 600
            np.testing.assert_allclose(
                states["time"], np.arange(1, 601) * CONTROL_DT, atol=1e-10
            )
            data = mujoco.MjData(model)
            penetration = 0.0
            for q, v in zip(states["qpos"], states["qvel"], strict=True):
                assert np.isfinite(q).all() and np.isfinite(v).all()
                data.qpos[:] = q
                data.qvel[:] = v
                mujoco.mj_forward(model, data)
                if data.ncon:
                    penetration = max(penetration, -float(data.contact.dist.min()))
            entry = {
                "label": label,
                "case": case,
                "checkpoint": str(checkpoint.relative_to(ROOT)),
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
                "trajectory_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "model_mjb_sha256": model_hash(model),
                "maximum_sampled_penetration_m": penetration,
                "penetration_below_original_8mm_target": penetration < 0.008,
                "outcome": outcome,
            }
            configure(model)
            runs.append((row, col, case, model, data, states, entry))
            entries.append(entry)
            print(
                "CAPTURE",
                label,
                case,
                outcome["completed_with_allowed_support"],
                flush=True,
            )
    for row in range(3):
        assert (
            len(
                {
                    entry["model_mjb_sha256"]
                    for entry in entries
                    if entry["case"] == CASES_SHOWN[row]
                }
            )
            == 1
        )
    capture_seconds = time.perf_counter() - start
    renderers = [mujoco.Renderer(run[3], height=330, width=608) for run in runs]
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (1920, 1440),
        fps=25,
        codec="libx264",
        quality=8,
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
    start = time.perf_counter()
    try:
        for frame in range(300):
            canvas = Image.new("RGB", (1920, 1440), (9, 18, 25))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (16, 10), "MATCHED 90-SECOND TRAINING RUNS", font=font(38), fill="white"
            )
            for col, title in enumerate(TITLES):
                draw.text((16 + col * 640, 70), title, font=font(25), fill="#91d8d0")
            k = min(round(frame / 25 / CONTROL_DT), 599)
            for renderer, (row, col, case, model, data, states, entry) in zip(
                renderers, runs, strict=True
            ):
                data.qpos[:] = states["qpos"][k]
                data.qvel[:] = states["qvel"][k]
                data.time = states["time"][k]
                mujoco.mj_forward(model, data)
                body = CASES[case][0]
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.qpos[:3]
                camera.lookat[2] = 0.23
                camera.distance = 1.35
                camera.azimuth = camera_azimuth(body)
                camera.elevation = -18
                renderer.update_scene(data, camera=camera, scene_option=option)
                damage_markers(renderer.scene, model, data, body)
                x, y = 16 + col * 640, 120 + row * 422
                draw.rectangle((x, y, x + 607, y + 405), fill=(17, 29, 38))
                title = {
                    "healthy": "HEALTHY",
                    "lower_fr": "LOWER LEG REMOVED / FR",
                    "whole_fr": "ENTIRE LEG REMOVED / FR",
                }[case]
                draw.text((x + 12, y + 8), title, font=font(25), fill="#91d8d0")
                canvas.paste(Image.fromarray(renderer.render()), (x, y + 44))
                status = "5 m COMPLETE" if states["completed"][k] else "WALKING"
                if not states["alive"][k]:
                    status = "FAILED / NO RESET"
                elif not states["support_valid"][k]:
                    status = "INVALID SUPPORT"
                distance = states["qpos"][k, 0] - states["qpos"][0, 0]
                caption = f"{distance:.2f} m | {status}"
                if (
                    frame >= 275
                    and not entry["outcome"]["completed_with_allowed_support"]
                ):
                    caption = "TRIAL RESULT: GOAL NOT MET"
                draw.text(
                    (x + 12, y + 379),
                    caption,
                    font=font(22),
                    fill="white",
                )
            draw.text(
                (16, 1390),
                f"t = {runs[0][5]['time'][k]:05.2f} s | Same parent and recipe | Different final weights per column | 1x playback",
                font=font(23),
                fill="white",
            )
            draw.text(
                (16, 1418),
                "Physics remained CPU MuJoCo in every training run. Demonstrations use CPU inference and scripted lane commands.",
                font=font(19),
                fill="#9db8c1",
            )
            writer.send(np.asarray(canvas))
    finally:
        writer.close()
        for renderer in renderers:
            renderer.close()
    report = {
        "seed": 9143,
        "cases": entries,
        "physics_timestep_s": 0.0005,
        "control_timestep_s": CONTROL_DT,
        "resolution": [1920, 1440],
        "fps": 25,
        "frames": 300,
        "seconds": 12,
        "playback_speed": 1,
        "capture_seconds": capture_seconds,
        "replayed_existing_states": replay,
        "original_capture_seconds": prior.get(
            "original_capture_seconds", prior["capture_seconds"]
        )
        if replay
        else capture_seconds,
        "previous_render_seconds": prior["render_seconds"] if replay else None,
        "render_seconds": time.perf_counter() - start,
        "new_training_seconds_for_video": 0,
        "same_initial_conditions_and_physics_per_row": True,
        "runtime_inference": "CPU for every panel; only the training backend differed",
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "scope": "Predetermined three-case illustration; full nine-body evaluation is separate. All failures retained. No promotion of these benchmark policies.",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    main(**vars(parser.parse_args()))
