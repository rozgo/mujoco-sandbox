"""Current shared-decoder flight and learned dynamics, rendered from measured traces."""

import argparse
import json
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from embodied_fly.neural_view import colorize
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.record import font

BG, PANEL, FG, MUTED = "#121516", "#202526", "#E6E1DB", "#A7ADAD"
GOLD, GREEN, GRAY, RED = "#FFC31F", "#73AC87", "#8D9699", "#C76D55"
SIZE, FPS = (1920, 1080), 50


class Film:
    def __init__(self, args):
        self.args = args
        self.report = json.loads((args.capture / "report.json").read_text())
        self.benchmark = json.loads(args.benchmark.read_text())
        self.cases = self.report["flight"]["cases"]
        self.captures = []
        for case in self.cases:
            file = args.capture / "flight" / case["file"]
            if sha256(file) != case["sha256"]:
                raise ValueError("Flight capture checksum differs")
            with np.load(file) as z:
                self.captures.append({k: z[k] for k in z.files})
        for name, key in [
            ("neural_maps.npz", "neural_maps_sha256"),
            ("forecast_windows.npz", "forecast_windows_sha256"),
        ]:
            if sha256(args.capture / name) != self.report[key]:
                raise ValueError("Observer data checksum differs")
        with np.load(args.capture / "neural_maps.npz") as z:
            self.neural = {k: z[k] for k in z.files}
        with np.load(args.capture / "forecast_windows.npz") as z:
            self.forecast = {k: z[k] for k in z.files}
        self.model = mujoco.MjModel.from_binary_path(str(args.capture / "flight/model.mjb"))
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1920)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 1080)
        self.data = mujoco.MjData(self.model)
        self.option = mujoco.MjvOption()
        self.option.geomgroup[3:] = 0
        self.small = mujoco.Renderer(self.model, height=330, width=620)
        self.large = mujoco.Renderer(self.model, height=760, width=1160)
        # Precompute camera smoothing from actual poses, rather than smoothing
        # independently during preview seeks. Rendering order cannot change it.
        self.follow = []
        for c in self.captures:
            p = c["qpos"][::10, :3]
            smooth = np.empty_like(p)
            smooth[0] = p[0]
            rate = 1 - np.exp(-1 / FPS / np.array([0.2, 0.2, 0.3]))
            for i in range(1, len(p)):
                smooth[i] = smooth[i - 1] + rate * (p[i] - smooth[i - 1])
            self.follow.append(smooth)
        self.forecast_limits = []
        for axis in range(3):
            arrays = [
                (self.forecast[k][..., axis] - self.forecast["initial"][:, None, axis]) * 10
                for k in ("actual", "predicted", "analytical")
            ]
            lo = min(float(x.min()) for x in arrays)
            hi = max(float(x.max()) for x in arrays)
            margin = max(0.05, (hi - lo) * 0.12)
            self.forecast_limits.append((lo - margin, hi + margin))

    def close(self):
        self.small.close()
        self.large.close()

    def scene(self, index, frame, large=False):
        capture = self.captures[index]
        step = min(frame * 10, len(capture["time"]) - 1)
        for key in ("qpos", "qvel", "act", "ctrl"):
            getattr(self.data, key)[:] = capture[key][step]
        self.data.time = float(capture["time"][step])
        mujoco.mj_forward(self.model, self.data)
        camera = mujoco.MjvCamera()
        camera.azimuth = 135
        camera.elevation = -22
        camera.distance = 1.65 if not large else 1.55
        camera.lookat[:] = self.follow[index][min(frame, len(self.follow[index]) - 1)]
        renderer = self.large if large else self.small
        renderer.update_scene(self.data, camera=camera, scene_option=self.option)
        return Image.fromarray(renderer.render())

    def board(self, title, subtitle):
        image = Image.new("RGB", SIZE, BG)
        d = ImageDraw.Draw(image)
        d.rectangle((0, 0, 1920, 6), fill=GOLD)
        d.text((26, 25), title, font=font(36), fill=GOLD)
        d.text((28, 77), subtitle, font=font(23), fill=FG)
        d.line((24, 1011, 1896, 1011), fill="#43494A", width=2)
        d.text(
            (28, 1033),
            "1x REAL TIME   /   1,000 Hz physics   /   500 Hz brain control   /   Same MuJoCo fly",
            font=font(22),
            fill=MUTED,
        )
        return image, d

    def four_flies(self, frame):
        image, d = self.board(
            "FLY LAB / FULL-BODY CONTROL",
            "One MaleCNS actor. One shared decoder. All 78 actuator commands.",
        )
        for i, (case, capture) in enumerate(zip(self.cases, self.captures, strict=True)):
            x = 24 + (i % 2) * 640
            y = 126 + (i // 2) * 438
            d.rectangle((x, y, x + 620, y + 422), fill=PANEL)
            d.text(
                (x + 12, y + 10),
                f"START {case['episode']}  /  LEARNED HOVER",
                font=font(21),
                fill=GREEN if i == 2 else FG,
            )
            failed = (
                case["first_failure_seconds"] is not None
                and frame / FPS >= case["first_failure_seconds"]
            )
            if failed:
                d.text((x + 55, y + 175), "EPISODE ENDED", font=font(30), fill=RED)
            else:
                image.paste(self.scene(i, frame), (x, y + 43))
            step = min(frame * 10, len(capture["time"]) - 1)
            altitude = capture["qpos"][step, 2] * 10
            climb = (capture["qpos"][step, 2] - capture["qpos"][0, 2]) * 10
            d.text(
                (x + 12, y + 389),
                f"{frame / FPS:4.2f} s   /   ALT {altitude:5.1f} mm   /   CLIMB {climb:+5.1f} mm",
                font=font(19),
                fill=MUTED,
            )
        x = 1310
        d.rectangle((x, 126, 1896, 986), fill=PANEL)
        d.text((x + 18, 145), "SELECTED FLY / START 8", font=font(23), fill=GREEN)
        frame = min(frame, len(self.neural["activity"]) - 1)
        heat = colorize(
            self.neural["activity"][frame, 2].astype(np.float32), self.neural["occupancy"]
        )
        image.paste(
            Image.fromarray(heat).resize((340, 340), Image.Resampling.NEAREST), (x + 123, 186)
        )
        d.text((x + 28, 540), "SIMULATED NEURAL ACTIVITY", font=font(21), fill=FG)
        d.text(
            (x + 28, 571),
            "Measured cell locations / model latent state",
            font=font(17),
            fill=MUTED,
        )
        d.text((x + 28, 623), "78 ACTUATOR COMMANDS", font=font(23), fill=GOLD)
        values = self.captures[2]["action"][min(frame * 10, len(self.captures[2]["time"]) - 1)]
        for i, value in enumerate(values):
            row, col = divmod(i, 13)
            left = x + 22 + col * 43
            top = 669 + row * 37
            color = np.array([255, 195, 31]) if value >= 0 else np.array([74, 175, 166])
            rgb = tuple(
                np.rint(
                    np.array([32, 37, 38])
                    + np.clip(abs(value), 0, 1) * (color - np.array([32, 37, 38]))
                ).astype(int)
            )
            d.rounded_rectangle(
                (left, top, left + 37, top + 29), radius=3, fill=rgb, outline="#566064"
            )
            d.text(
                (left + 3, top + 6),
                str(i + 1),
                font=font(12),
                fill="#101416" if abs(value) > 0.6 else FG,
            )
        d.text(
            (x + 25, 916),
            "815 motor signals → 384 shared units → 78 outputs",
            font=font(18),
            fill=FG,
        )
        d.text(
            (x + 25, 949),
            "No wing-specific branch / hover still has drift",
            font=font(18),
            fill=MUTED,
        )
        return image

    def predictions(self, frame):
        image, d = self.board(
            "FLY LAB / PREDICTING THE CONSEQUENCES",
            "200 ms into the future, given the recorded actuator commands. Predictor is an observer here.",
        )
        image.paste(self.scene(2, frame, large=True), (24, 171))
        d.text(
            (40, 132),
            f"START 8 / SHARED FULL-BODY ACTOR / {frame / FPS:.2f} s",
            font=font(23),
            fill=GREEN,
        )
        actual = self.captures[2]
        step = frame * 10
        delta = (actual["qpos"][step, :3] - actual["qpos"][0, :3]) * 10
        d.text(
            (42, 954),
            f"DISPLACEMENT  X {delta[0]:+.1f}   Y {delta[1]:+.1f}   Z {delta[2]:+.1f} mm",
            font=font(23),
            fill=FG,
        )
        x = 1230
        for i, (label, color) in enumerate(
            (
                ("Actual future motion", GREEN),
                ("Physics + learned residual", GOLD),
                ("Analytical model", GRAY),
            )
        ):
            d.line((x + 5, 149 + i * 33, x + 29, 149 + i * 33), fill=color, width=4)
            d.text((x + 42, 135 + i * 33), label, font=font(22), fill=color)
        for axis, label in enumerate(("X / FORWARD", "Y / SIDEWAYS", "Z / VERTICAL")):
            y = 267 + axis * 230
            w = 650
            h = 206
            d.rectangle((x, y, x + w, y + h), fill=PANEL)
            d.text((x + 14, y + 10), f"{label} DISPLACEMENT / mm", font=font(20), fill=FG)
            lo, hi = self.forecast_limits[axis]
            left, right, top, bottom = x + 64, x + w - 20, y + 52, y + h - 35
            for tick in np.linspace(lo, hi, 4):
                py = bottom - (tick - lo) / (hi - lo) * (bottom - top)
                d.line((left, py, right, py), fill="#394142", width=1)
                d.text((x + 8, py - 8), f"{tick:.1f}", font=font(15), fill=MUTED)
            for key, color in (("analytical", GRAY), ("actual", GREEN), ("predicted", GOLD)):
                values = np.r_[
                    0,
                    (
                        self.forecast[key][frame, :, axis]
                        - self.forecast["initial"][frame, axis]
                    )
                    * 10,
                ]
                points = [
                    (
                        left + j / 100 * (right - left),
                        bottom - (float(value) - lo) / (hi - lo) * (bottom - top),
                    )
                    for j, value in enumerate(values)
                ]
                d.line(points, fill=color, width=3 if key != "analytical" else 2)
            d.text((left, y + h - 26), "now", font=font(15), fill=MUTED)
            d.text((right - 80, y + h - 26), "+200 ms", font=font(15), fill=MUTED)
        return image

    def outcome(self):
        image, d = self.board(
            "FLY LAB / CURRENT CHECKPOINT",
            "A shared motor decoder and a validated dynamics predictor. The next step is model-guided learning.",
        )
        for x in (24, 978):
            d.rectangle((x, 157, x + 918, 951), fill=PANEL)
        d.text((60, 189), "LEARNED MOTOR CONTROL", font=font(30), fill=GREEN)
        passed = sum(c["survived_ten_seconds"] for c in self.cases)
        speed = np.mean([c["velocity_rms_mm_s"] for c in self.cases])
        climb = np.mean([c["final_displacement_mm"][2] for c in self.cases])
        for y, line, size, color in [
            (272, "ONE POLICY / 78 OUTPUTS", 36, FG),
            (345, f"{passed}/4 complete ten-second flights", 29, FG),
            (418, f"Velocity RMS: {speed:.2f} mm/s", 28, MUTED),
            (472, f"Average net climb: {climb:.1f} mm", 28, MUTED),
            (568, "Hover remains imperfect; drift is visible.", 25, FG),
            (658, "One dense decoder for the whole body.", 25, FG),
            (710, "Leg, antenna and mouth skills come later.", 24, MUTED),
            (794, "Existing learned behavior preserved.", 24, MUTED),
            (838, "No new motor training in this video.", 24, MUTED),
        ]:
            d.text((60, y), line, font=font(size), fill=color)
        d.text((1012, 189), "LEARNED DYNAMICS RESIDUALS", font=font(30), fill=GOLD)
        for y, line, size, color in [
            (272, "120 SECONDS / RTX 4090", 36, FG),
            (345, "60% lower error after command changes", 29, GOLD),
            (391, "200 ms forecast / 504 held-out changes", 23, MUTED),
            (478, "1.6% lower average ordinary-flight error", 27, FG),
            (523, "Compared with the analytical model", 23, MUTED),
            (617, "100% correct significant change directions", 26, FG),
            (661, "Worst grouped effect-magnitude error: <5%", 23, MUTED),
            (756, "These are prediction results.", 26, GOLD),
            (803, "The predictor has not yet updated the actor.", 23, MUTED),
        ]:
            d.text((1012, y), line, font=font(size), fill=color)
        return image


def run(args):
    started = time.perf_counter()
    film = Film(args)
    try:
        if args.preview:
            args.output.mkdir(parents=True, exist_ok=False)
            film.four_flies(150).save(args.output / "flight.png")
            film.predictions(150).save(args.output / "forecast.png")
            film.outcome().save(args.output / "outcome.png")
            return
        if args.output.exists():
            raise FileExistsError("Preserve previous videos")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio_ffmpeg.write_frames(
            str(args.output),
            SIZE,
            fps=FPS,
            codec="libx264",
            quality=8,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )
        writer.send(None)
        frames = 0
        try:
            for i in range(500):
                writer.send(np.asarray(film.four_flies(i)))
                frames += 1
            for i in range(len(film.forecast["time"])):
                writer.send(np.asarray(film.predictions(i)))
                frames += 1
            card = np.asarray(film.outcome())
            for _ in range(200):
                writer.send(card)
                frames += 1
        finally:
            writer.close()
        report = {
            "provenance": evidence(),
            "completed_utc": utc_now(),
            "render_wall_seconds": time.perf_counter() - started,
            "video_sha256": sha256(args.output),
            "capture_report_sha256": sha256(args.capture / "report.json"),
            "benchmark_sha256": sha256(args.benchmark),
            "checkpoint_sha256": film.report["checkpoint_sha256"],
            "world_model_sha256": film.report["world_model_sha256"],
            "fps": FPS,
            "frames": frames,
            "dimensions": list(SIZE),
            "duration_seconds": frames / FPS,
            "playback_speed": 1,
            "sections": [
                {
                    "name": "shared actor, four starts",
                    "start_seconds": 0,
                    "duration_seconds": 10,
                },
                {
                    "name": "observer forecasts on start 8",
                    "start_seconds": 10,
                    "duration_seconds": len(film.forecast["time"]) / FPS,
                },
                {
                    "name": "measured results",
                    "start_seconds": 10 + len(film.forecast["time"]) / FPS,
                    "duration_seconds": 4,
                },
            ],
            "camera": "0.2 s horizontal, 0.3 s vertical damping; body-relative observer only",
            "neural_view": "Actual actor latent state, binned using measured MaleCNS cell positions; observer-only",
            "world_model_control_share": 0,
            "new_motor_training_updates": 0,
        }
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "provenance"}), flush=True)
    finally:
        film.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument(
        "--benchmark",
        type=Path,
        default=Path("docs/embodied_fly/world_model/residual_02/evaluation.json"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preview", action="store_true")
    run(p.parse_args())
