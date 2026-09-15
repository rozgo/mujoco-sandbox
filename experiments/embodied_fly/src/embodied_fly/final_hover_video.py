"""Show original physics, unchanged-weight transfer, and measured PPO outcome."""

import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_fly.full_body_guidance_evaluate import summarize
from embodied_fly.full_body_guidance_video import BG, COLORS, FG, FPS, MUTED, SIZE, Capture
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.record import font


class Film:
    def __init__(self, args):
        self.report = json.loads((args.training / "report.json").read_text())
        self.selected = self.report["selected"] or self.report["evaluations"][-1]
        self.captures = [
            Capture(p)
            for p in (
                args.pid,
                args.original,
                args.transferred,
                Path(self.selected["capture"]),
            )
        ]
        self.labels = [
            "PID / NEW PHYSICS",
            "RETAINED FLY / ORIGINAL",
            "SAME WEIGHTS / NEW",
            "PPO / NEW PHYSICS",
        ]
        self.details = [
            "Reference controller",
            "Original v5 baseline",
            "Physics change only / zero training",
            f"{self.selected['training_seconds'] / 60:.1f} minutes of full-body PPO",
        ]
        self.models, self.data, self.renderers = [], [], []
        for c in self.captures:
            model = mujoco.MjModel.from_binary_path(str(c.folder / "model.mjb"))
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, 628)
            model.vis.global_.offheight = max(model.vis.global_.offheight, 460)
            self.models.append(model)
            self.data.append(mujoco.MjData(model))
            self.renderers.append(mujoco.Renderer(model, height=460, width=628))
        new_contract = self.captures[0].report["physical_contract"]
        if any(self.captures[i].report["physical_contract"] != new_contract for i in (2, 3)):
            raise ValueError(
                "PID, transfer and trained actor must share the new physical plant"
            )
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.height_max = max(
            30,
            5
            * np.ceil(
                max(float(a["qpos"][:, 2].max() * 10) for c in self.captures for a in c.data)
                / 5
            ),
        )
        self.speed_max = max(
            20, 10 * np.ceil(max(float(s.max()) for c in self.captures for s in c.speed) / 10)
        )

    def frame(self, world, frame):
        seconds, step = frame / FPS, frame * 10
        im = Image.new("RGB", SIZE, BG)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 2560, 6), fill=COLORS[0])
        d.text((24, 22), "FLIGHT LAB / THE FINAL HOVER PUSH", font=font(38), fill=COLORS[0])
        d.text(
            (26, 78),
            "Separating a change in physics from learning through the full MaleCNS controller.",
            font=font(26),
            fill=FG,
        )
        episode = self.captures[0].report["cases"][world]["episode"]
        d.text(
            (26, 119),
            f"START {episode}  /  {seconds:.2f} s  /  1x REAL TIME  /  All learned actors use one 78-output decoder",
            font=font(23),
            fill=MUTED,
        )
        for col, capture in enumerate(self.captures):
            x, color = 6 + col * 640, COLORS[col]
            a, case = capture.data[world], capture.report["cases"][world]
            d.rectangle((x, 157, x + 628, 233), fill="#23292a")
            d.text((x + 12, 167), self.labels[col], font=font(24), fill=color)
            d.text((x + 12, 202), self.details[col], font=font(19), fill=MUTED)
            index = min(step, len(a["time"]) - 1)
            failed_at = case["first_failure_seconds"]
            after_failure = failed_at is not None and seconds >= failed_at
            if after_failure and seconds >= failed_at + 0.4:
                d.rectangle((x, 240, x + 628, 700), fill="#1b2021")
                d.text((x + 70, 425), "FLIGHT FAILED", font=font(35), fill="#d77762")
                d.text((x + 70, 481), f"at {failed_at:.3f} seconds", font=font(26), fill=FG)
            else:
                data, model = self.data[col], self.models[col]
                for key in ("qpos", "qvel", "act", "ctrl"):
                    getattr(data, key)[:] = a[key][index]
                data.time = seconds
                mujoco.mj_forward(model, data)
                camera = mujoco.MjvCamera()
                camera.azimuth, camera.elevation, camera.distance = 135, -22, 1.15
                smooth = capture.follow[world][min(frame, len(capture.follow[world]) - 1)]
                camera.lookat[:] = data.qpos[:3] + np.clip(smooth - data.qpos[:3], -0.15, 0.15)
                self.renderers[col].update_scene(data, camera=camera, scene_option=self.option)
                im.paste(Image.fromarray(self.renderers[col].render()), (x, 240))
                if after_failure:
                    d.text(
                        (x + 12, 655),
                        "FAILURE / recorded physical motion",
                        font=font(23),
                        fill="#e07862",
                    )
            stop = min(step + 1, len(a["time"]))
            if failed_at is not None:
                stop = min(stop, round(failed_at * 500))
            for y, values, label, maximum in (
                (718, a["qpos"][:, 2] * 10, "ALTITUDE / mm", self.height_max),
                (857, capture.speed[world], "VELOCITY ERROR / mm/s", self.speed_max),
            ):
                d.rectangle((x, y, x + 628, y + 121), fill="#23292a")
                d.text((x + 10, y + 5), f"{label}   0–{maximum:g}", font=font(18), fill=MUTED)
                points = [
                    (x + 8 + i / 5000 * 612, y + 105 - float(values[i]) / maximum * 71)
                    for i in range(0, stop, 10)
                ]
                if len(points) > 1:
                    d.line(points, fill=color, width=3)
                d.line((x + 8, y + 106, x + 620, y + 106), fill="#62696a")
                if label.startswith("ALTITUDE"):
                    h0 = float(a["qpos"][0, 2] * 10)
                    d.line(
                        (
                            x + 8,
                            y + 105 - h0 / maximum * 71,
                            x + 620,
                            y + 105 - h0 / maximum * 71,
                        ),
                        fill="#4b5253",
                    )
        d.text(
            (26, 1022),
            "1,000 Hz MuJoCo physics  /  500 Hz actor  /  Frozen connectome wiring  /  Forces from measured wing motion",
            font=font(25),
            fill=MUTED,
        )
        return im

    def card(self):
        im = Image.new("RGB", SIZE, BG)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 2560, 6), fill=COLORS[0])
        d.text((40, 40), "MEASURED OUTCOME", font=font(43), fill=COLORS[0])
        for x, text in (
            (40, "Controller"),
            (1080, "10-second flights"),
            (1530, "Velocity error"),
            (2030, "Net climb"),
        ):
            d.text((x, 178), text, font=font(28), fill=MUTED)
        for row, i in enumerate((1, 2, 3)):
            y, m, color = 253 + row * 92, summarize(self.captures[i].report), COLORS[i]
            for x, text in (
                (40, self.labels[i]),
                (1080, f"{m['complete']}/4"),
                (
                    1530,
                    f"{m['velocity_rms_mm_s']:.2f} mm/s"
                    if m["complete"] == 4
                    else "Incomplete",
                ),
                (2030, f"{m['net_climb_mm']:.1f} mm" if m["complete"] == 4 else "Incomplete"),
            ):
                d.text((x, y), text, font=font(32), fill=color)
        success = self.report["selected"] is not None
        lines = [
            "PPO passes the predefined joint improvement criteria."
            if success
            else "PPO did not meet the predefined joint improvement criteria.",
            f"Actual training: {self.report['training_seconds'] / 60:.2f} minutes / 32 parallel worlds / {self.report['counters']['transitions']:,} transitions.",
            "All 78 decoder outputs can learn; the encoder and MaleCNS core remain frozen.",
            "A physics-only improvement is not counted as learned improvement.",
            "This result is improved hover control, not proof of complete fly behavior."
            if success
            else "Retain the previous policy. Pause further training on this approach.",
        ]
        for y, text in zip((596, 672, 744, 816, 939), lines, strict=True):
            d.text((40, y), text, font=font(29), fill=COLORS[0] if y in (596, 939) else FG)
        return im

    def close(self):
        for renderer in self.renderers:
            renderer.close()


def run(args):
    start = time.perf_counter()
    film = Film(args)
    try:
        if args.preview:
            args.output.mkdir(parents=True, exist_ok=False)
            film.frame(2, 200).save(args.output / "comparison.png")
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
            for world in range(4):
                for i in range(500):
                    writer.send(np.asarray(film.frame(world, i)))
                    frames += 1
            card = np.asarray(film.card())
            for _ in range(350):
                writer.send(card)
                frames += 1
        finally:
            writer.close()
        report = {
            "provenance": evidence(),
            "completed_utc": utc_now(),
            "render_seconds": time.perf_counter() - start,
            "video_sha256": sha256(args.output),
            "fps": FPS,
            "frames": frames,
            "duration_seconds": frames / FPS,
            "size": SIZE,
            "selected": film.selected,
            "training_report_sha256": sha256(args.training / "report.json"),
            "capture_reports": [
                {
                    "path": c.folder.as_posix(),
                    "sha256": sha256(c.folder / "report.json"),
                    "physical_contract": c.report["physical_contract"],
                }
                for c in film.captures
            ],
            "different_physics_explicitly_labeled": True,
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps({"video": args.output.as_posix(), "seconds": report["render_seconds"]}),
            flush=True,
        )
    finally:
        film.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("training", "pid", "original", "transferred", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    run(p.parse_args())
