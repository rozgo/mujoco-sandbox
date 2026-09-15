"""Show the failed full updates and the best physically tested smaller updates."""

import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_fly.full_body_guidance_evaluate import summarize
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.record import font
from embodied_fly.velocity_exercise import rolling_velocity

BG, FG, MUTED = "#121516", "#e6e1db", "#a7adad"
COLORS = ("#ffc31f", "#a7adad", "#71aaa4", "#73ac87")
FPS, SIZE = 50, (2560, 1080)


class Capture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.report = json.loads((self.folder / "report.json").read_text())
        self.data = []
        self.follow = []
        self.speed = []
        for case in self.report["cases"]:
            file = self.folder / case["file"]
            if sha256(file) != case["sha256"]:
                raise ValueError("Changed physical capture")
            with np.load(file) as z:
                a = {k: z[k] for k in z.files}
            self.data.append(a)
            self.speed.append(
                np.linalg.norm(rolling_velocity(a["measured_velocity"] * 10), axis=1)
            )
            poses = a["qpos"][::10, :3]
            follow = poses.copy()
            rate = 1 - np.exp(-0.02 / np.array([0.2, 0.2, 0.3]))
            for i in range(1, len(follow)):
                follow[i] = follow[i - 1] + rate * (poses[i] - follow[i - 1])
            self.follow.append(follow)


class Film:
    def __init__(self, args):
        self.args = args
        self.initial = json.loads((args.initial_evaluation / "report.json").read_text())
        self.backtrack_reports = [
            json.loads((p / "report.json").read_text()) for p in args.backtracks
        ]
        self.candidates = [c for r in self.backtrack_reports for c in r["candidates"]]
        self.best = {}
        for method in ("analytical", "learned_residual"):
            complete = [
                c
                for c in self.candidates
                if c["method"] == method and c["summary"]["complete"] == 4
            ]
            if not complete:
                raise ValueError("No complete small-update flight to show")
            self.best[method] = min(complete, key=lambda c: c["summary"]["velocity_rms_mm_s"])
        self.parent = Capture(args.parent)
        self.pid = Capture(args.pid)
        full = {
            c["method"]: Capture(c["capture"])
            for c in self.initial["candidates"]
            if c["step"] == 100
        }
        self.groups = {
            "full": [self.pid, self.parent, full["analytical"], full["learned_residual"]],
            "small": [
                self.pid,
                self.parent,
                Capture(self.best["analytical"]["capture"]),
                Capture(self.best["learned_residual"]["capture"]),
            ],
        }
        contract = self.parent.report["physical_contract"]
        if any(
            c.report["physical_contract"] != contract
            for group in self.groups.values()
            for c in group
        ):
            raise ValueError("Video methods must share the same physical contract")
        self.model = mujoco.MjModel.from_binary_path(str(args.parent / "model.mjb"))
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 628)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 460)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=460, width=628)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.limits = {}
        for name, group in self.groups.items():
            height = max(float(a["qpos"][:, 2].max() * 10) for c in group for a in c.data)
            speed = max(float(s.max()) for c in group for s in c.speed)
            self.limits[name] = (
                max(25, 5 * np.ceil(height / 5)),
                max(20, 10 * np.ceil(speed / 10)),
            )

    def frame(self, name, world, frame):
        group = self.groups[name]
        seconds = frame / FPS
        step = frame * 10
        image = Image.new("RGB", SIZE, BG)
        d = ImageDraw.Draw(image)
        d.rectangle((0, 0, 2560, 6), fill=COLORS[0])
        d.text(
            (24, 22),
            "FLIGHT LAB / LEARNING THROUGH ONE FULL-BODY DECODER",
            font=font(36),
            fill=COLORS[0],
        )
        caption = (
            "Full updates: lower predicted cost, but startup fails in real flight."
            if name == "full"
            else "Smaller weight updates: same physical starts, live MaleCNS feedback, no controller assistance."
        )
        d.text((26, 77), caption, font=font(25), fill=FG)
        episode = group[0].report["cases"][world]["episode"]
        d.text(
            (26, 117),
            f"START {episode}  /  {seconds:.2f} s  /  1x REAL TIME",
            font=font(22),
            fill=MUTED,
        )
        labels = [
            "PID REFERENCE",
            "PRESERVED PARENT",
            "ANALYTICAL GUIDANCE",
            "LEARNED-MODEL GUIDANCE",
        ]
        for column, c in enumerate(group):
            x = 6 + 640 * column
            color = COLORS[column]
            a = c.data[world]
            case = c.report["cases"][world]
            d.rectangle((x, 157, x + 628, 234), fill="#23292a")
            d.text((x + 12, 167), labels[column], font=font(25), fill=color)
            detail = "One shared decoder / 78 outputs"
            if column >= 2:
                fraction = (
                    self.best["analytical" if column == 2 else "learned_residual"]["fraction"]
                    if name == "small"
                    else 1
                )
                detail = f"{100 * fraction:g}% of trained weight update"
            elif column == 0:
                detail = "Reference only / separate from learned actors"
            d.text((x + 12, 201), detail, font=font(19), fill=MUTED)
            failed = (
                case["first_failure_seconds"] is not None
                and seconds >= case["first_failure_seconds"]
            )
            if failed:
                d.rectangle((x, 240, x + 628, 700), fill="#1b2021")
                d.text((x + 70, 420), "FLIGHT FAILED", font=font(35), fill="#d77762")
                d.text(
                    (x + 70, 478),
                    f"at {case['first_failure_seconds']:.3f} seconds",
                    font=font(26),
                    fill=FG,
                )
            else:
                index = min(step, len(a["time"]) - 1)
                for key in ("qpos", "qvel", "act", "ctrl"):
                    getattr(self.data, key)[:] = a[key][index]
                self.data.time = seconds
                mujoco.mj_forward(self.model, self.data)
                camera = mujoco.MjvCamera()
                camera.azimuth = 135
                camera.elevation = -22
                camera.distance = 1.15
                smooth = c.follow[world][min(frame, len(c.follow[world]) - 1)]
                camera.lookat[:] = self.data.qpos[:3] + np.clip(
                    smooth - self.data.qpos[:3], -0.15, 0.15
                )
                self.renderer.update_scene(self.data, camera=camera, scene_option=self.option)
                image.paste(Image.fromarray(self.renderer.render()), (x, 240))
            stop = min(step + 1, len(a["time"]))
            if case["first_failure_seconds"] is not None:
                stop = min(stop, round(case["first_failure_seconds"] * 500))
            for y, values, label, maximum in (
                (718, a["qpos"][:, 2] * 10, "ALTITUDE / mm", self.limits[name][0]),
                (857, c.speed[world], "VELOCITY ERROR / mm/s", self.limits[name][1]),
            ):
                d.rectangle((x, y, x + 628, y + 121), fill="#23292a")
                d.text((x + 10, y + 5), f"{label}   0–{maximum:g}", font=font(18), fill=MUTED)
                d.line((x + 8, y + 106, x + 620, y + 106), fill="#62696a", width=1)
                points = [
                    (x + 8 + i / 5000 * 612, y + 105 - float(values[i]) / maximum * 71)
                    for i in range(0, stop, 10)
                ]
                if len(points) > 1:
                    d.line(points, fill=color, width=3)
                d.line(
                    (
                        x + 8 + min(seconds, 10) / 10 * 612,
                        y + 30,
                        x + 8 + min(seconds, 10) / 10 * 612,
                        y + 107,
                    ),
                    fill="#566061",
                    width=1,
                )
        d.text(
            (26, 1022),
            "1,000 Hz MuJoCo physics  /  500 Hz actor  /  Model is used for training only  /  Complete physical traces",
            font=font(25),
            fill=MUTED,
        )
        return image

    def card(self):
        im = Image.new("RGB", SIZE, BG)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 2560, 6), fill=COLORS[0])
        d.text((40, 35), "WHAT THIS TRIAL ACHIEVED", font=font(43), fill=COLORS[0])
        d.text(
            (40, 107),
            "One shared 78-output decoder. Small improvements transfer; the full updates fail.",
            font=font(31),
            fill=FG,
        )
        parent = summarize(self.parent.report)
        rows = [("Preserved parent", parent, COLORS[1])] + [
            (label, self.best[method]["summary"], COLORS[i])
            for label, method, i in [
                ("Analytical-guided candidate", "analytical", 2),
                ("Learned-model candidate", "learned_residual", 3),
            ]
        ]
        d.text((40, 223), "MEASURED IN MUJOCO", font=font(27), fill=MUTED)
        for x, text in (
            (960, "Complete flights"),
            (1440, "Velocity RMS"),
            (1970, "Net climb"),
        ):
            d.text((x, 223), text, font=font(27), fill=MUTED)
        for y, (label, m, color) in zip((298, 387, 476), rows, strict=True):
            d.text((40, y), label, font=font(33), fill=color)
            for x, text in (
                (960, f"{m['complete']}/4"),
                (1440, f"{m['velocity_rms_mm_s']:.2f} mm/s"),
                (1970, f"{m['net_climb_mm']:.1f} mm"),
            ):
                d.text((x, y), text, font=font(33), fill=color)
        eligible = [c for c in self.candidates if c["passed"]]
        status = (
            "A candidate passes the predefined promotion gates."
            if eligible
            else "No candidate passes every predefined promotion gate; the parent stays preferred."
        )
        for y, text, color in (
            (597, status, COLORS[0]),
            (
                665,
                "Both methods: 100 optimizer updates. Learned model 8.4 s; analytical 8.2 s.",
                FG,
            ),
            (
                721,
                "Extra physical line search adds evaluation time, not optimizer updates.",
                MUTED,
            ),
            (
                798,
                "Diagnosis: changing the decoder changes future neural activity and wing timing.",
                FG,
            ),
            (
                854,
                "Frozen recorded neural histories overstate the benefit of large updates.",
                MUTED,
            ),
            (
                944,
                "Next: refresh actual histories after each small, physically accepted update.",
                COLORS[0],
            ),
        ):
            d.text((40, y), text, font=font(29), fill=color)
        return im


def run(args):
    started = time.perf_counter()
    film = Film(args)
    try:
        if args.preview:
            args.output.mkdir(parents=True, exist_ok=False)
            film.frame("full", 2, 50).save(args.output / "failures.png")
            film.frame("small", 2, 200).save(args.output / "comparison.png")
            film.card().save(args.output / "results.png")
            return
        if args.output.exists():
            raise FileExistsError("Preserve previous video")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(args.output),
            SIZE,
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        frames = 0
        try:
            for i in range(125):
                writer.send(np.asarray(film.frame("full", 2, i)))
                frames += 1
            for world in range(4):
                for i in range(500):
                    writer.send(np.asarray(film.frame("small", world, i)))
                    frames += 1
            card = np.asarray(film.card())
            for _ in range(300):
                writer.send(card)
                frames += 1
        finally:
            writer.close()
        report = {
            "provenance": evidence(),
            "completed_utc": utc_now(),
            "render_seconds": time.perf_counter() - started,
            "video_sha256": sha256(args.output),
            "fps": FPS,
            "frames": frames,
            "duration_seconds": frames / FPS,
            "dimensions": SIZE,
            "playback_speed": 1,
            "selected_for_display": film.best,
            "selection_rule": "Lowest velocity RMS among complete physically evaluated candidates per method; full-update failures shown first",
            "chapters": [
                {"start": 0, "seconds": 2.5, "name": "Full-update startup failures / start 8"},
                {"start": 2.5, "seconds": 40, "name": "Small updates / all four starts"},
                {"start": 42.5, "seconds": 6, "name": "Measured result and next action"},
            ],
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("provenance", "selected_for_display")
                }
            ),
            flush=True,
        )
    finally:
        film.renderer.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("initial-evaluation", "parent", "pid", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--backtracks", type=Path, nargs="+", required=True)
    p.add_argument("--preview", action="store_true")
    run(p.parse_args())
