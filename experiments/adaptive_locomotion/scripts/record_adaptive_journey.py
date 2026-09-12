"""Render the accepted adaptive-dog progression from verified saved physical states.

Parts render on their original capture hosts. Git/LFS carries the finished chapter
movies; assembly needs only those movies and their provenance reports, no GPU.
"""

import argparse
import hashlib
import json
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import ROOT, build_model
from adaptive_locomotion.evaluate import make_case
from adaptive_locomotion.graphite import (
    BG,
    LINE,
    MUTED,
    ORANGE,
    PANEL,
    TEAL,
    WHITE,
    configure,
    decorate,
    layout,
)
from adaptive_locomotion.moving_record import camera as moving_camera
from adaptive_locomotion.presentation import camera_azimuth
from adaptive_locomotion.record import font, model_hash
from adaptive_locomotion.standing_evaluate import BODY_MAP
from adaptive_locomotion.standing_record import VIDEO_GROUPS, ground_marks

DEST = ROOT / "previews/locomotion/graphite"
MANIFESTS = {
    "walking": "previews/locomotion/adaptive_dog_v1.json",
    "standing": "previews/locomotion/standing/adaptive_standing_v2.json",
    "terrain": "previews/locomotion/standing/adaptive_standing_v3.json",
    "moving": "previews/locomotion/moving/moving_supports_v1.json",
}
LEGS = ("FL", "FR", "RL", "RR")
SURFACES = {
    "pads": "INDIVIDUAL FOOT PADS",
    "pads_high": "HIGH PADS",
    "pads_extreme": "EXTREME PADS",
    "slope_x": "PITCH SLOPE / 12°",
    "slope_y": "ROLL SLOPE / 12°",
    "slope_x_18": "PITCH SLOPE / 18°",
    "slope_x_24": "PITCH SLOPE / 24°",
    "slope_y_24": "ROLL SLOPE / 24°",
    "steps_20": "STEPS / 20 cm",
    "steps_28": "STEPS / 28 cm",
    **{f"gap_{leg.lower()}": f"MISSING SUPPORT / {leg}" for leg in LEGS},
}
MOTIONS = {
    "translate": "TRANSLATION",
    "yaw": "ROTATION / YAW",
    "heave": "VERTICAL MOTION",
    "rock": "ROCKING / PITCH + ROLL",
    "combined": "COMBINED MOTION",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def body_label(body):
    if body == "healthy":
        return "HEALTHY / FOUR INTACT LEGS"
    kind, leg = body.split("_")
    return f"{'LOWER LEG' if kind == 'lower' else 'ENTIRE LEG'} REMOVED / {leg.upper()}"


def chapters(part):
    if part == "walking":
        return [
            (
                "LEARNED WALKING",
                "One shared actor / nine physical bodies",
                ["healthy"],
                12,
            ),
            (
                "ADAPTING TO LIMB LOSS",
                "Entire calf and foot removed / same walking weights",
                [f"lower_{s.lower()}" for s in LEGS],
                12,
            ),
            (
                "WALKING WITH THREE LEGS",
                "Hip, thigh and calf removed / same walking weights",
                [f"whole_{s.lower()}" for s in LEGS],
                12,
            ),
        ]
    if part == "standing":
        titles = (
            "WALK. STOP. BALANCE. CONTINUE.",
            "FINDING SUPPORT",
            "HOLDING ON DIFFICULT TERRAIN",
            "BALANCING AFTER LIMB LOSS",
        )
        subtitles = (
            "Standing added / one actor changes behavior with the command",
            "Individual pads, slopes and missing foot support",
            "Higher pads, steeper slopes and larger steps",
            "Same standing weights across physical body variants",
        )
        return [
            (
                title,
                subtitle,
                [f"{b}:{s}:{int(t)}" for b, s, t in cases],
                12 if i == 0 else 10,
            )
            for i, ((_, cases), title, subtitle) in enumerate(
                zip(VIDEO_GROUPS, titles, subtitles, strict=True)
            )
        ]
    if part == "terrain":
        cases = read(ROOT / MANIFESTS[part])["accepted_terrain_cases"]
        return [
            (
                "NATURAL CORRECTIONS",
                "Reviewed terrain trials / original measured misses remain visible",
                [c["surface"] for c in cases[i : i + 4]],
                10,
            )
            for i in (0, 4)
        ]
    return [
        (
            "THE WORLD STARTS MOVING",
            "Moving supports added / one actor across all platform motions",
            [m + "_2" for m in ("translate", "yaw", "heave", "rock")],
            10,
        ),
        (
            "BALANCING ON A MOVING WORLD",
            "Healthy robot / combined platform motion / synchronized cameras",
            ["combined_2"],
            10,
        ),
    ]


class Run:
    def __init__(self, part, key, trace_root):
        manifest = read(ROOT / MANIFESTS[part])
        self.part, self.key = part, key
        self.surface = "flat"
        if part == "walking":
            case = next(c for c in manifest["cases"] if c["case"] == key)
            outcome = case["outcome"]
            self.body = BODY_MAP[key]
            self.title = body_label(key)
            path = (
                trace_root / f"outputs/locomotion/recordings/adaptive_dog_v1/{key}.npz"
            )
            env = make_case(key, trials=1, timestep=manifest["physics_timestep_s"])
            model = env.groups[0].model
            env.close()
            expected_model = outcome["model_mjb_sha256"]
            row = outcome["rows"][0]
            self.passed = bool(row["completed_with_allowed_support"])
        elif part == "standing":
            body, self.surface, transition = key.split(":")
            case = next(
                c
                for c in manifest["cases"]
                if (c["body"], c["surface"], c["transition"])
                == (body, self.surface, bool(int(transition)))
            )
            self.body = BODY_MAP[body]
            self.title = (
                body_label(body) if self.surface == "flat" else SURFACES[self.surface]
            )
            path = trace_root / case["trajectory"]
            model = build_model(
                self.body, "stand_" + self.surface, timestep=case["physics_timestep_s"]
            )
            expected_model = case["model_mjb_sha256"]
            row = case["rows"][0]
            self.passed = bool(row["passed"])
        elif part == "terrain":
            case = next(
                c for c in manifest["accepted_terrain_cases"] if c["surface"] == key
            )
            self.surface, self.body = key, BODY_MAP[case["body"]]
            self.title = SURFACES[key]
            path = trace_root / case["trajectory"]
            model_path = trace_root / case["model"]
            if sha(model_path) != case["model_sha256"]:
                raise ValueError(f"Source model file changed: {key}")
            model = mujoco.MjModel.from_binary_path(str(model_path))
            expected_model = case["model_sha256"]
            row = case["repeated_row"]
            self.passed = bool(row["passed"])
        else:
            case = next(c for c in manifest["cases"] if c["key"] == key)
            self.surface, self.body = "moving", BODY_MAP[case["body"]]
            self.title = MOTIONS[case["motion"]]
            path = trace_root / case["trajectory"]
            model = mujoco.MjModel.from_binary_path(
                str(trace_root / case["model_binary"])
            )
            expected_model = case["model_mjb_sha256"]
            row = case["rows"][0]
            self.passed = bool(row["passed"])
        if sha(path) != case["trajectory_sha256"]:
            raise ValueError(f"Source trace changed: {key}")
        if model_hash(model) != expected_model:
            raise ValueError(f"Physical model differs from original capture: {key}")
        with np.load(path, allow_pickle=False) as trace:
            self.trace = {k: trace[k] for k in trace.files}
        times = self.trace["time"]
        if not np.allclose(times, np.arange(1, len(times) + 1) * 0.02):
            raise ValueError(f"Incomplete control timeline: {key}")
        for field in ("qpos", "qvel"):
            if not np.isfinite(self.trace[field]).all():
                raise ValueError(f"Nonfinite states: {key}")
        if part == "walking":
            names = outcome["support_geom_names"]
            allowed = outcome["allowed_support_geom_names"]
            contacts = []
            for leg in LEGS:
                name = next((n for n in allowed if n.startswith(leg)), None)
                contacts.append(
                    self.trace["support_peak_forces_n"][:, names.index(name)]
                    if name
                    else np.zeros(len(times))
                )
            self.trace["tip_forces"] = np.stack(contacts, axis=1)
        self.provenance = {
            "key": key,
            "source_manifest": MANIFESTS[part],
            "source_manifest_sha256": sha(ROOT / MANIFESTS[part]),
            "trajectory": str(path.relative_to(trace_root)),
            "trajectory_sha256": case["trajectory_sha256"],
            "physical_model_sha256": expected_model,
            "checkpoint_sha256": manifest["checkpoint_hashes"][1]
            if part == "moving"
            else manifest["checkpoint_sha256"],
            "seed": case.get("seed", manifest.get("seed")),
            "trial": case.get("selected_trial", 0),
            "original_trial_result": row,
            "physics_backend": "mjbatch"
            if part in ("walking", "terrain")
            else manifest["physics_backend"],
            "state_count": len(times),
            "state_start_s": float(times[0]),
            "state_end_s": float(times[-1]),
        }
        configure(model)
        model.vis.global_.offwidth, model.vis.global_.offheight = 1920, 1080
        self.model, self.data = model, mujoco.MjData(model)
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.renderers = {}
        self.index = 0

    def set_frame(self, frame):
        self.index = 4 * frame + 3
        if self.index >= len(self.trace["time"]):
            raise ValueError("Replay exceeds saved physical timeline")
        for field in ("qpos", "qvel", "ctrl"):
            if field in self.trace:
                getattr(self.data, field)[:] = self.trace[field][self.index]
        self.data.time = self.trace["time"][self.index]
        mujoco.mj_forward(self.model, self.data)

    def image(self, size, kind="follow"):
        if size not in self.renderers:
            self.renderers[size] = mujoco.Renderer(
                self.model, width=size[0], height=size[1]
            )
        renderer = self.renderers[size]
        if self.part == "moving":
            cam = moving_camera(self.model, self.data, kind)
            if kind == "follow" and size[1] == 350:
                cam.distance = 1.9
        elif kind == "head":
            cam = "head"
        else:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = self.data.qpos[:3] - [0, 0, 0.06]
            cam.distance = 1.55 if kind == "follow" else 2.3
            if kind == "follow" and size[1] == 350 and self.surface == "flat":
                cam.distance = 1.15
            cam.elevation = -23 if kind == "follow" else -65
            cam.azimuth = camera_azimuth(self.body)
            if self.surface.endswith(("fl", "rl")):
                cam.azimuth = -50
        renderer.update_scene(self.data, camera=cam, scene_option=self.option)
        decorate(renderer.scene, self.model, self.data, self.body)
        if self.part in ("standing", "terrain"):
            ground_marks(renderer.scene, self.surface)
        return Image.fromarray(renderer.render())

    def close(self):
        for renderer in self.renderers.values():
            renderer.close()


def footer(draw, part, seconds):
    stage = {
        "walking": "01 / WALKING",
        "standing": "02 / STANDING ADDED",
        "terrain": "02 / STANDING ADDED",
        "moving": "03 / MOVING SUPPORTS ADDED",
    }[part]
    draw.rectangle((0, 1010, 1920, 1080), fill=BG)
    draw.line((24, 1010, 1896, 1010), fill=LINE, width=1)
    draw.text((32, 1030), f"{seconds:05.2f} s / 2x playback", font=font(24), fill=WHITE)
    draw.text((425, 1032), stage, font=font(21), fill=TEAL)
    draw.text(
        (1130, 1032), "ONE SHARED POLICY WITHIN THIS STAGE", font=font(21), fill=MUTED
    )


def contacts(draw, run, x, y, spacing=69):
    for i, (label, load) in enumerate(
        zip(LEGS, run.trace["tip_forces"][run.index], strict=True)
    ):
        px = x + i * spacing
        draw.ellipse((px, y + 5, px + 10, y + 15), fill=TEAL if load > 1 else LINE)
        if label in run.body.absent_legs:
            draw.line((px, y + 5, px + 10, y + 15), fill=ORANGE, width=2)
        draw.text((px + 16, y), label, font=font(17), fill=MUTED)


def canvas_for(part, title, subtitle, runs, frame):
    for run in runs:
        run.set_frame(frame)
    if len(runs) == 1:
        run = runs[0]
        canvas = layout(
            run.image((1260, 840)),
            run.image((592, 404), "overview"),
            run.image((592, 404), "head"),
            title=title,
            subtitle=subtitle,
            seconds=run.data.time,
        )
        draw = ImageDraw.Draw(canvas)
        if part == "standing":
            command = run.trace["commands"][run.index]
            label = "BALANCE" if np.linalg.norm(command) < 0.01 else "WALK"
        else:
            label = "LEARNED WALK" if part == "walking" else "LEARNED BALANCE"
        draw.rounded_rectangle((45, 928, 465, 970), radius=4, fill=PANEL)
        draw.text((62, 938), label, font=font(21), fill=TEAL)
        contacts(draw, run, 935, 947)
    else:
        canvas = Image.new("RGB", (1920, 1080), BG)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((32, 30, 38, 90), fill=TEAL)
        draw.text((58, 23), "MUJOCO / ADAPTIVE LOCOMOTION", font=font(20), fill=MUTED)
        draw.text((56, 52), title, font=font(37), fill=WHITE)
        draw.text((56, 104), subtitle, font=font(21), fill=MUTED)
        columns = 3 if len(runs) == 6 else 2
        width = (1872 - (columns - 1) * 16) // columns
        image_height = 350
        for i, run in enumerate(runs):
            x, y = 24 + (width + 16) * (i % columns), 150 + 430 * (i // columns)
            draw.rectangle((x, y, x + width, y + 414), fill=PANEL)
            draw.text(
                (x + 14, y + 10),
                run.title,
                font=font(21 if columns == 3 else 24),
                fill=WHITE,
            )
            canvas.paste(run.image((width, image_height)), (x, y + 44))
            if part == "walking":
                status = f"{run.data.qpos[0] - run.trace['qpos'][0, 0]:.2f} m / WALKING"
            elif part == "moving":
                status = f"DECK DRIFT {100 * run.trace['relative_drift_m'][run.index]:.1f} cm"
            else:
                status = "UPRIGHT" if run.passed else "UPRIGHT / STRICT TARGET MISSED"
            draw.text(
                (x + 14, y + 390),
                status,
                font=font(16 if columns == 3 else 18),
                fill=MUTED if run.passed else ORANGE,
            )
            if columns == 2:
                contacts(draw, run, x + width - 294, y + 390)
    footer(ImageDraw.Draw(canvas), part, runs[0].data.time)
    return canvas


def writer(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = imageio_ffmpeg.write_frames(
        str(path),
        (1920, 1080),
        fps=25,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart"],
    )
    stream.send(None)
    return stream


def render(part, trace_root, output, preview_only):
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    started, count, rows, records = time.perf_counter(), 0, [], []
    stream = None if preview_only else writer(output)
    preview_dir = ROOT / "outputs/locomotion/graphite/journey_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)
    try:
        for chapter, (title, subtitle, keys, seconds) in enumerate(chapters(part)):
            with ExitStack() as stack:
                runs = []
                for key in keys:
                    run = Run(part, key, trace_root)
                    stack.callback(run.close)
                    runs.append(run)
                    records.append(run.provenance)
                start = count
                indices = (
                    (min(50, seconds * 25 // 2 - 1),)
                    if preview_only
                    else range(seconds * 25 // 2)
                )
                for frame in indices:
                    canvas = canvas_for(part, title, subtitle, runs, frame)
                    if preview_only:
                        canvas.save(preview_dir / f"{part}_{chapter}.png")
                    else:
                        stream.send(np.asarray(canvas))
                    count += 1
                rows.append(
                    {
                        "title": title,
                        "subtitle": subtitle,
                        "cases": keys,
                        "start_s": start / 25,
                        "end_s": count / 25,
                    }
                )
                print(
                    json.dumps({"part": part, "chapter": chapter, "frames": count}),
                    flush=True,
                )
    finally:
        if stream is not None:
            stream.close()
    if not preview_only:
        write(
            output.with_suffix(".json"),
            {
                "source_commit": source,
                "part": part,
                "theme": "graphite",
                "chapters": rows,
                "cases": records,
                "frames": count,
                "fps": 25,
                "duration_s": count / 25,
                "dimensions": [1920, 1080],
                "playback_speed": 2,
                "state_indices": "3,7,... at 50 Hz; 0.08 s to exact endpoint; no interpolation; 2x playback",
                "new_physics_steps": 0,
                "new_training_seconds": 0,
                "render_encode_seconds": time.perf_counter() - started,
                "video_sha256": sha(output),
                "visual_inspection": "pending",
            },
        )


def closing_card():
    canvas = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((62, 78, 69, 164), fill=TEAL)
    draw.text((95, 76), "ADAPTIVE LOCOMOTION", font=font(50), fill=WHITE)
    draw.text(
        (97, 143),
        "THREE TRAINING STAGES. A GROWING SET OF SKILLS.",
        font=font(25),
        fill=MUTED,
    )
    for y, number, title, text in (
        (290, "01", "WALK", "Healthy + eight physical limb-removal variants"),
        (475, "02", "STAND", "Walk, stop and balance on pads, gaps, slopes and steps"),
        (
            660,
            "03",
            "MOVE WITH THE WORLD",
            "Healthy robot balances on translating and rotating supports",
        ),
    ):
        draw.text((96, y), number, font=font(46), fill=TEAL)
        draw.text((210, y), title, font=font(36), fill=WHITE)
        draw.text((212, y + 61), text, font=font(27), fill=MUTED)
        draw.line((210, y + 122, 1800, y + 122), fill=LINE, width=1)
    draw.text(
        (96, 875),
        "Learned joint control / physical MuJoCo contact / native rendering",
        font=font(27),
        fill=WHITE,
    )
    draw.text(
        (96, 943),
        "Three successive checkpoints shown; one shared actor within each stage.",
        font=font(24),
        fill=MUTED,
    )
    draw.text(
        (96, 990),
        "Selected recorded trials. Terrain misses remain visible; damaged moving balance remains unresolved.",
        font=font(23),
        fill=MUTED,
    )
    return canvas


def assemble(output):
    started = time.perf_counter()
    sources, records, chapter_rows, count = [], [], [], 0
    stream = writer(output)
    try:
        for part in ("walking", "standing", "terrain", "moving"):
            path = DEST / "chapters" / f"{part}_v1.mp4"
            report = read(path.with_suffix(".json"))
            if sha(path) != report["video_sha256"]:
                raise ValueError(f"Chapter bytes differ: {part}")
            frames = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
            metadata = next(frames)
            if metadata["size"] != (1920, 1080) or metadata["fps"] != 25:
                raise ValueError("Chapter format differs")
            before = count
            for raw in frames:
                stream.send(raw)
                count += 1
            if count - before != report["frames"]:
                raise ValueError("Incomplete chapter decode")
            chapter_rows.extend(
                {
                    **c,
                    "stage": part,
                    "start_s": c["start_s"] + before / 25,
                    "end_s": c["end_s"] + before / 25,
                }
                for c in report["chapters"]
            )
            records.extend(report["cases"])
            sources.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "video_sha256": report["video_sha256"],
                    "manifest_sha256": sha(path.with_suffix(".json")),
                    "render_encode_seconds": report["render_encode_seconds"],
                }
            )
        card = closing_card()
        chapter_rows.append(
            {
                "title": "RESULTS / EXPLICIT SIX-SECOND CARD",
                "start_s": count / 25,
                "end_s": count / 25 + 6,
            }
        )
        for _ in range(150):
            stream.send(np.asarray(card))
            count += 1
    finally:
        stream.close()
    write(
        output.with_suffix(".json"),
        {
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "video_sha256": sha(output),
            "chapters": chapter_rows,
            "sources": sources,
            "cases": records,
            "frames": count,
            "fps": 25,
            "dimensions": [1920, 1080],
            "duration_s": count / 25,
            "playback_speed": 2,
            "new_physics_steps": 0,
            "new_training_seconds": 0,
            "assemble_encode_seconds": time.perf_counter() - started,
            "checkpoint_count": len({r["checkpoint_sha256"] for r in records}),
            "policy_provenance": "Three successive accepted checkpoints; one shared actor across cases within each stage. No claim that every scene uses the newest weights.",
            "camera_pixels_used_by_policy": False,
            "camera_views": ["observer", "overhead", "head"],
            "theme": "graphite",
            "generated_media": False,
            "limitations": "Original failed quantitative gates retained. Damaged moving transfer remains unresolved. Selected demonstration trials are not a holdout success-rate estimate.",
            "visual_inspection": "pending",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=(*MANIFESTS, "assemble"), required=True)
    parser.add_argument("--trace-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    output = args.output or (
        DEST / "adaptive_dog_complete_v1.mp4"
        if args.part == "assemble"
        else DEST / "chapters" / f"{args.part}_v1.mp4"
    )
    if args.part == "assemble":
        assemble(output)
    else:
        render(args.part, args.trace_root, output, args.preview_only)


if __name__ == "__main__":
    main()
