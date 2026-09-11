"""Replay all four rear-removal cases, before/after, facing the intact rear leg."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import CONTROL_DT, ROOT
from adaptive_locomotion.evaluate import make_case
from adaptive_locomotion.presentation import configure, damage_markers
from adaptive_locomotion.record import font, model_hash


def compare(before, after, output):
    manifests = [json.loads(Path(p).read_text()) for p in (before, after)]
    assert manifests[0]["seed"] == manifests[1]["seed"] == 9143
    assert all(m["seconds"] == 12 and m["playback_speed"] == 1 for m in manifests)
    cases = ("lower_rl", "lower_rr", "whole_rl", "whole_rr")
    runs = []
    for row, manifest in enumerate(manifests):
        assert (
            hashlib.sha256((ROOT / manifest["checkpoint"]).read_bytes()).hexdigest()
            == manifest["checkpoint_sha256"]
        )
        directory = (
            ROOT / "outputs/locomotion/recordings" / Path((before, after)[row]).stem
        )
        for col, case in enumerate(cases):
            entry = next(c for c in manifest["cases"] if c["case"] == case)
            path = directory / f"{case}.npz"
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest()
                == entry["trajectory_sha256"]
            )
            states = dict(np.load(path))
            env = make_case(case, trials=1, seed=manifest["seed"])
            g = env.groups[0]
            model = g.model
            body = g.body
            assert model_hash(model) == entry["outcome"]["model_mjb_sha256"]
            env.close()
            configure(model)
            runs.append(
                (row, col, case, body, model, mujoco.MjData(model), states, entry)
            )
    for run in runs[1:]:
        np.testing.assert_array_equal(run[6]["time"], runs[0][6]["time"])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    renderers = [mujoco.Renderer(r[4], height=510, width=620) for r in runs]
    writer = imageio_ffmpeg.write_frames(
        str(output),
        (2560, 1440),
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
            canvas = Image.new("RGB", (2560, 1440), (9, 18, 25))
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (24, 16),
                "INTACT REAR FOOT / BEFORE AND AFTER",
                font=font(38),
                fill="white",
            )
            draw.text(
                (24, 66),
                "Same start and physics | Same timestamp | 1x playback | Camera faces the healthy rear leg",
                font=font(24),
                fill="#9db8c1",
            )
            k = min(round(frame / 25 / CONTROL_DT), 599)
            for renderer, (row, col, case, body, model, data, states, entry) in zip(
                renderers, runs, strict=True
            ):
                data.qpos[:] = states["qpos"][k]
                data.qvel[:] = states["qvel"][k]
                data.time = states["time"][k]
                mujoco.mj_forward(model, data)
                camera = mujoco.MjvCamera()
                camera.lookat[:] = data.qpos[:3]
                camera.lookat[2] = 0.22
                camera.distance = 1.18
                camera.azimuth = 90 if case.endswith("rl") else -90
                camera.elevation = -12
                renderer.update_scene(data, camera=camera, scene_option=option)
                damage_markers(renderer.scene, model, data, body)
                x, y = 16 + col * 636, 112 + row * 644
                draw.rectangle((x, y, x + 619, y + 615), fill=(17, 29, 38))
                draw.text(
                    (x + 16, y + 12),
                    ("BEFORE" if row == 0 else "AFTER")
                    + " / "
                    + case.replace("_", " ").upper(),
                    font=font(29),
                    fill="#92d9d0",
                )
                canvas.paste(Image.fromarray(renderer.render()), (x, y + 58))
                leg = "RR" if case.endswith("rl") else "RL"
                geom = model.geom(f"{leg}_terminal").id
                height = data.geom_xpos[geom, 2] - model.geom_size[geom, 0]
                status = "5 m COMPLETE" if states["completed"][k] else "WALKING"
                if not states["alive"][k]:
                    status = "FAILED / NO RESET"
                elif not states["support_valid"][k]:
                    status = "INVALID SUPPORT"
                draw.text(
                    (x + 16, y + 573),
                    f"{leg} foot: {max(height, 0) * 1000:.1f} mm clearance | {status}",
                    font=font(22),
                    fill="white",
                )
            draw.text(
                (24, 1400),
                f"t = {runs[0][6]['time'][k]:05.2f} s | Learned joint control; scripted lane commands | Orange = removed limb",
                font=font(23),
                fill="white",
            )
            writer.send(np.asarray(canvas))
    finally:
        writer.close()
        for r in renderers:
            r.close()
    report = {
        "before": str(Path(before).resolve().relative_to(ROOT)),
        "after": str(Path(after).resolve().relative_to(ROOT)),
        "before_checkpoint_sha256": manifests[0]["checkpoint_sha256"],
        "after_checkpoint_sha256": manifests[1]["checkpoint_sha256"],
        "seed": 9143,
        "cases": list(cases),
        "seconds": 12,
        "fps": 25,
        "frames": 300,
        "resolution": [2560, 1440],
        "playback_speed": 1,
        "replayed_existing_states": True,
        "new_training_seconds": 0,
        "render_seconds": time.perf_counter() - start,
        "video_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "description": "All four rear removals, camera facing the intact rear foot; same captured timestamps and physical trajectories as source grids. Foot clearance is the actual sphere bottom above flat ground.",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(compare(**vars(parser.parse_args())), indent=2))
