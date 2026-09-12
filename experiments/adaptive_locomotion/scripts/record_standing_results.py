"""Extend the results film with the eight terrain trials accepted after review."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import CONTROL_DT, ROOT
from adaptive_locomotion.presentation import configure
from adaptive_locomotion.record import font

BASE = ROOT / "previews/locomotion/standing/adaptive_standing_v2.mp4"
REVIEW = ROOT / "previews/locomotion/standing/failure_review.json"
OUTPUT = ROOT / "previews/locomotion/standing/adaptive_standing_v3.mp4"
SURFACES = (
    "gap_rl",
    "gap_rr",
    "pads_extreme",
    "slope_x_18",
    "slope_x_24",
    "slope_y_24",
    "steps_20",
    "steps_28",
)
FLAGS = (
    "Foot finds adjacent support",
    "Alternate support / tilt beyond target",
    "Hold drift beyond target",
    "Hold drift beyond target",
    "Alternate support / drift beyond target",
    "Hold drift beyond target",
    "Alternate support / tilt beyond target",
    "Alternate support / tilt beyond target",
)
TEAL, WHITE = "#91d8d0", "#eef4f5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Panel:
    def __init__(self, case):
        self.case = case
        model_path, trace_path = ROOT / case["model"], ROOT / case["trajectory"]
        assert sha(model_path) == case["model_sha256"]
        assert sha(trace_path) == case["trajectory_sha256"]
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        with np.load(trace_path, allow_pickle=False) as trace:
            self.trace = {k: trace[k] for k in trace.files}
        self.data = mujoco.MjData(self.model)
        configure(self.model)
        self.model.vis.global_.offwidth = 960
        self.model.vis.global_.offheight = 360
        self.renderer = mujoco.Renderer(self.model, width=960, height=360)
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False

    def frame(self, index):
        k = min(round(index / 25 / CONTROL_DT), len(self.trace["time"]) - 1)
        data, model = self.data, self.model
        data.qpos[:], data.qvel[:], data.ctrl[:] = (
            self.trace["qpos"][k],
            self.trace["qvel"][k],
            self.trace["ctrl"][k],
        )
        data.time = self.trace["time"][k]
        mujoco.mj_forward(model, data)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = data.qpos[:3]
        camera.distance, camera.elevation = 1.55, -24
        camera.azimuth = -50 if self.case["surface"].endswith("rl") else 50
        self.renderer.update_scene(data, camera=camera, scene_option=self.option)
        return Image.fromarray(self.renderer.render())

    def close(self):
        self.renderer.close()


def terrain_frame(panels, start, index):
    canvas = Image.new("RGB", (1920, 1080), "#091219")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (24, 16),
        "BALANCING ON UNEVEN SUPPORTS / ONE SHARED POLICY",
        font=font(34),
        fill=TEAL,
    )
    draw.text(
        (24, 62),
        "Terrain behavior accepted after visual review  |  Original measured flags retained",
        font=font(24),
        fill=WHITE,
    )
    for i, panel in enumerate(panels):
        x, y = (i % 2) * 960, 100 + (i // 2) * 440
        draw.text(
            (x + 18, y + 5),
            panel.case["surface"].upper().replace("_", " "),
            font=font(26),
            fill=WHITE,
        )
        canvas.paste(panel.frame(index), (x, y + 40))
        draw.text((x + 18, y + 408), FLAGS[start + i], font=font(22), fill="#ffbd94")
    t = panels[0].data.time
    draw.text(
        (24, 1008),
        f"t = {t:.2f} s  |  1x real time  |  All four remain upright  |  Learned joint control",
        font=font(25),
        fill=WHITE,
    )
    draw.text(
        (24, 1046),
        "CPU MuJoCo / mjbatch  |  Same weights and original friction  |  Observer cameras  |  Seed 9307",
        font=font(21),
        fill="#b6c8ce",
    )
    return canvas


def card():
    canvas = Image.new("RGB", (1920, 1080), "#091219")
    draw = ImageDraw.Draw(canvas)
    lines = (
        ("ONE POLICY / WALK, STOP AND BALANCE", 48, TEAL),
        ("25 / 25 recorded trials remain upright", 39, WHITE),
        ("8 additional terrain trials accepted after visual review", 34, TEAL),
        (
            "Natural corrections, sliding and alternate support remain visible",
            30,
            WHITE,
        ),
        ("Original strict balance checks: 10 / 25 recorded trials pass", 30, "#ffbd94"),
        ("Visual acceptance and measured flags are recorded separately", 28, WHITE),
        ("RTX 4090 training: 8 min 57 s of standing fine-tuning", 29, WHITE),
        (
            "Inherited walker: 32 min 03 s  |  No new training for this edit",
            26,
            "#b6c8ce",
        ),
        (
            "One trained policy across different bodies and terrain. No retraining between scenes.",
            26,
            "#b6c8ce",
        ),
        (
            "Selected demonstration trials; full holdout results remain in the run report",
            25,
            "#b6c8ce",
        ),
    )
    for (text, size, color), y in zip(
        lines, (60, 185, 295, 355, 465, 525, 655, 715, 835, 915), strict=True
    ):
        draw.text((48, y), text, font=font(size), fill=color)
    return canvas


def main(args):
    started = time.perf_counter()
    base = json.loads(BASE.with_suffix(".json").read_text())
    review = json.loads(REVIEW.read_text())
    assert sha(BASE) == base["video_sha256"]
    assert base["checkpoint_sha256"] == review["checkpoint_sha256"]
    assert base["fps"] == 25 and base["frames"] == 1125
    assert len(base["cases"]) == 17
    assert sum(c["passed"] for c in base["cases"]) == 10
    assert sum(c["survived"] for c in base["cases"]) == 17
    selected = review["cases"][:8]
    assert tuple(c["surface"] for c in selected) == SURFACES
    assert all(
        c["body"] == "healthy" and c["repeated_row"]["survived"] for c in selected
    )
    assert all(not c["repeated_row"]["passed"] for c in selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.preview_only:
        directory = ROOT / "outputs/locomotion/standing/results_v3"
        directory.mkdir(parents=True, exist_ok=True)
        for start in (0, 4):
            panels = [Panel(c) for c in selected[start : start + 4]]
            try:
                terrain_frame(panels, start, 200).save(
                    directory / f"preview_{start}.png"
                )
            finally:
                for panel in panels:
                    panel.close()
        card().save(directory / "card.png")
        return
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        (1920, 1080),
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    count, decoded = 0, 0
    frames = imageio_ffmpeg.read_frames(str(BASE), pix_fmt="rgb24")
    metadata = next(frames)
    assert metadata["size"] == (1920, 1080) and metadata["fps"] == 25
    try:
        for i, raw in enumerate(frames):
            decoded += 1
            if i == 800:
                for start in (0, 4):
                    panels = [Panel(c) for c in selected[start : start + 4]]
                    try:
                        for k in range(250):
                            writer.send(np.asarray(terrain_frame(panels, start, k)))
                            count += 1
                    finally:
                        for panel in panels:
                            panel.close()
                    print(
                        json.dumps({"accepted_terrain_group_rendered": start // 4 + 1}),
                        flush=True,
                    )
            if i < 1050:
                writer.send(raw)
                count += 1
        assert decoded == base["frames"]
        for _ in range(125):
            writer.send(np.asarray(card()))
            count += 1
    finally:
        frames.close()
        writer.close()
    cases = list(base["cases"])
    for case in selected:
        result = dict(case["repeated_result"])
        result.update(
            trials=1,
            passed=int(case["repeated_row"]["passed"]),
            survived=1,
            rows=[case["repeated_row"]],
            selected_trial=case["selected_trial"],
            visual_acceptance=True,
        )
        cases.append(result)
    chapters = [
        (0, 12, "Walk / balance / walk", "existing Warp footage"),
        (12, 10, "Local support", "existing Warp footage"),
        (22, 10, "Aggressive terrain", "existing Warp footage"),
        (32, 10, "Reviewed terrain / rear gaps, pads, slope", "CPU recorded states"),
        (42, 10, "Reviewed terrain / steep slopes and steps", "CPU recorded states"),
        (52, 10, "Damaged bodies", "existing Warp footage"),
        (62, 5, "Results", "labeled static results card"),
    ]
    report = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "checkpoint_sha256": base["checkpoint_sha256"],
        "video_sha256": sha(args.output),
        "source_video": str(BASE.relative_to(ROOT)),
        "source_video_sha256": base["video_sha256"],
        "terrain_capture_report": str(REVIEW.relative_to(ROOT)),
        "terrain_capture_report_sha256": sha(REVIEW),
        "accepted_terrain_cases": selected,
        "selection_note": "User accepted the eight terrain conditions before the damaged section (0:00–1:28) of the failure review. These are selected post-review presentation cases, with original measured failures retained.",
        "edit_note": "Preserves all 42 seconds of v2 physical footage; inserts two ten-second four-panel terrain groups before damaged bodies; replaces the final card. Physical states, friction and weights unchanged. Source MP4 sections decoded and re-encoded once.",
        "cases": cases,
        "chapters": [
            {"start_s": a, "duration_s": b, "label": c, "source": d}
            for a, b, c, d in chapters
        ],
        "frames": count,
        "fps": 25,
        "duration_s": count / 25,
        "dimensions": [1920, 1080],
        "thumbnail_frame": 1000,
        "new_training_seconds": 0,
        "new_simulation_seconds": 0,
        "render_and_encode_seconds": time.perf_counter() - started,
    }
    assert count == 1675
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "duration_s": count / 25,
                "render_seconds": report["render_and_encode_seconds"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preview-only", action="store_true")
    main(parser.parse_args())
