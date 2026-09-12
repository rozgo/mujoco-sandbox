"""Capture and render every demonstrated skill with one frozen final actor."""

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
from adaptive_locomotion import ember
from adaptive_locomotion.bodies import LEGS, ROOT
from adaptive_locomotion.evaluate import rollout
from adaptive_locomotion.limb_loss import LOSS_CASES
from adaptive_locomotion.moving_evaluate import run_case as move
from adaptive_locomotion.moving_record import camera as moving_camera
from adaptive_locomotion.presentation import camera_azimuth
from adaptive_locomotion.record import font
from adaptive_locomotion.standing_evaluate import BODY_MAP
from adaptive_locomotion.standing_evaluate import run_case as stand
from adaptive_locomotion.standing_surfaces import SURFACES, slope_degrees
from adaptive_locomotion.train import load_checkpoint
from PIL import Image, ImageDraw

CHECKPOINT = ROOT / "assets/locomotion/checkpoints/gpu_from_scratch/unified.pt"
EXPECTED = "076099ef8fc47bad53cb0cbeb9cc311e64ea47accacfa8ca4080488934b10241"
DEST = ROOT / "previews/locomotion/ember"
TRACE = ROOT / "outputs/locomotion/ember/unified_v1"
SEED = 9511
FPS = 25


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def body_label(body):
    if body == "healthy":
        return "HEALTHY / FOUR INTACT LEGS"
    kind, leg = body.split("_")
    return f"{'LOWER LEG' if kind == 'lower' else 'ENTIRE LEG'} REMOVED / {leg.upper()}"


def surface_label(surface):
    if surface.startswith("slope"):
        return f"{'PITCH' if surface.split('_')[1] == 'x' else 'ROLL'} SLOPE / {slope_degrees(surface):g}°"
    if surface.startswith("steps"):
        return f"STEPS / {surface.split('_')[1]} cm"
    if surface.startswith("gap"):
        return f"MISSING FOOT SUPPORT / {surface[-2:].upper()}"
    return {
        "flat": "FLAT GROUND",
        "pads": "INDIVIDUAL FOOT PADS",
        "pads_high": "HIGH PADS",
        "pads_extreme": "EXTREME PADS",
    }[surface]


def specifications():
    specs = {}
    for body in LOSS_CASES:
        key = f"walking_{body}"
        specs[key] = {
            "kind": "walking",
            "body": body,
            "surface": "flat",
            "transition": False,
            "seconds": 12,
        }
    for surface in SURFACES:
        specs[f"standing_healthy_{surface}"] = {
            "kind": "standing",
            "body": "healthy",
            "surface": surface,
            "transition": False,
            "seconds": 10,
        }
    for body in LOSS_CASES:
        specs[f"transition_{body}"] = {
            "kind": "standing",
            "body": body,
            "surface": "flat",
            "transition": True,
            "seconds": 12,
        }
        if body != "healthy":
            specs[f"standing_{body}_flat"] = {
                "kind": "standing",
                "body": body,
                "surface": "flat",
                "transition": False,
                "seconds": 10,
            }
    for surface in ("still", "translate", "yaw", "heave", "rock", "combined"):
        specs[f"moving_{surface}"] = {
            "kind": "moving",
            "body": "healthy",
            "surface": surface,
            "transition": False,
            "seconds": 10,
        }
    assert len(specs) == 51
    return specs


def chapters():
    plan = [
        (
            "LEARNED WALKING",
            "Healthy body / one final policy throughout this film",
            ["walking_healthy"],
            12,
        ),
        (
            "ADAPTING TO LIMB LOSS",
            "Entire calf and foot removed / four body configurations",
            [f"walking_lower_{s.lower()}" for s in LEGS],
            12,
        ),
        (
            "WALKING WITH THREE LEGS",
            "Hip, thigh and calf removed / same final policy",
            [f"walking_whole_{s.lower()}" for s in LEGS],
            12,
        ),
        (
            "WALK. HOLD. WALK.",
            "One actor follows changing velocity commands",
            ["transition_healthy"],
            12,
        ),
    ]
    groups = (
        ("FINDING SUPPORT", ["flat", "pads", "slope_x", "slope_y"]),
        ("ONE FOOT WITHOUT SUPPORT", [f"gap_{s.lower()}" for s in LEGS]),
        ("PADS AND STEPS", ["pads_high", "pads_extreme", "steps_12", "steps_20"]),
        (
            "BALANCING ON SLOPES",
            ["slope_x_12", "slope_y_12", "slope_x_18", "slope_y_18"],
        ),
        ("THE HARDEST STATIC SUPPORTS", ["slope_x_24", "slope_y_24", "steps_28"]),
    )
    for title, surfaces in groups:
        keys = [f"standing_healthy_{s}" for s in surfaces]
        if len(keys) == 3:
            keys.append("moving_still")
        plan.append(
            (
                title,
                "Holding position / contact and drift limits remain visible",
                keys,
                10,
            )
        )
    for kind, label in (("lower", "LOWER-LEG LOSS"), ("whole", "ENTIRE-LEG LOSS")):
        plan.append(
            (
                f"BALANCE AFTER {label}",
                "Same final policy / physical limb removals",
                [f"standing_{kind}_{s.lower()}_flat" for s in LEGS],
                10,
            )
        )
    for kind, label in (("lower", "LOWER-LEG LOSS"), ("whole", "ENTIRE-LEG LOSS")):
        plan.append(
            (
                f"WALK–HOLD–WALK / {label}",
                "The command changes / the policy stays the same",
                [f"transition_{kind}_{s.lower()}" for s in LEGS],
                12,
            )
        )
    plan.extend(
        [
            (
                "THE WORLD STARTS MOVING",
                "Translation, rotation, heave and rocking",
                [f"moving_{m}" for m in ("translate", "yaw", "heave", "rock")],
                10,
            ),
            (
                "BALANCING ON A MOVING WORLD",
                "Combined platform motion / synchronized observer cameras",
                ["moving_combined"],
                10,
            ),
        ]
    )
    assert {k for _, _, keys, _ in plan for k in keys} == set(specifications())
    return plan


def capture(directory):
    assert sha(CHECKPOINT) == EXPECTED
    net, _ = load_checkpoint(CHECKPOINT)
    records = []
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for key, spec in specifications().items():
        meta = directory / f"{key}.json"
        if meta.exists():
            record = read(meta)
            assert record["checkpoint_sha256"] == EXPECTED and record["seed"] == SEED
            assert record["spec"] == spec
            assert sha(directory / record["trace"]) == record["trace_sha256"]
            assert sha(directory / record["model"]) == record["model_sha256"]
            records.append(record)
            continue
        tick = time.perf_counter()
        if spec["kind"] == "walking":
            outcome, frames, model = rollout(
                net, spec["body"], trials=1, seed=SEED, capture=True, timestep=0.0005
            )
        else:
            fn = move if spec["kind"] == "moving" else stand
            outcome, frames, model = fn(
                net,
                body=spec["body"],
                surface=spec["surface"],
                trials=1,
                seconds=spec["seconds"],
                seed=SEED,
                transition=spec["transition"],
                capture=True,
                physics_backend="mjbatch",
            )
        trace = {k: np.asarray([f[k] for f in frames]) for k in frames[0]}
        if spec["kind"] == "walking":
            names, allowed = (
                outcome["support_geom_names"],
                outcome["allowed_support_geom_names"],
            )
            loads = []
            for leg in LEGS:
                name = next((n for n in allowed if n.startswith(leg)), None)
                loads.append(
                    trace["support_peak_forces_n"][:, names.index(name)]
                    if name
                    else np.zeros(len(frames))
                )
            trace["tip_forces"] = np.stack(loads, axis=1)
        assert np.allclose(trace["time"], np.arange(1, len(frames) + 1) * 0.02)
        assert len(frames) == spec["seconds"] * 50
        assert all(
            np.isfinite(trace[k]).all()
            for k in ("qpos", "qvel", "action", "torque", "tip_forces")
        )
        path, binary = directory / f"{key}.npz", directory / f"{key}.mjb"
        np.savez_compressed(path, **trace)
        mujoco.mj_saveModel(model, str(binary), None)
        record = {
            "key": key,
            "spec": spec,
            "checkpoint_sha256": EXPECTED,
            "seed": SEED,
            "trial": 0,
            "physics_backend": "mjbatch",
            "physics_timestep_s": float(model.opt.timestep),
            "trace": path.name,
            "trace_sha256": sha(path),
            "model": binary.name,
            "model_sha256": sha(binary),
            "states": len(frames),
            "outcome": outcome,
            "capture_save_seconds": time.perf_counter() - tick,
        }
        write(meta, record)
        records.append(record)
        row = outcome["rows"][0]
        print(
            json.dumps(
                {
                    "captured": key,
                    "upright": row["survived"],
                    "passed": row.get(
                        "passed", row.get("completed_with_allowed_support")
                    ),
                }
            ),
            flush=True,
        )
    assert sha(CHECKPOINT) == EXPECTED
    report = {
        "checkpoint_sha256": EXPECTED,
        "seed": SEED,
        "cases": records,
        "new_training_seconds": 0,
        "capture_save_seconds": sum(r["capture_save_seconds"] for r in records),
        "invocation_seconds": time.perf_counter() - started,
        "simulated_world_seconds": sum(r["spec"]["seconds"] for r in records),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    write(directory / "capture.json", report)
    return report


def outcome_label(record):
    row = record["outcome"]["rows"][0]
    if not row["survived"]:
        return "FELL / TRIAL RETAINED", ember.RED
    if row.get("passed", row.get("completed_with_allowed_support")):
        return "TARGET MET", ember.IVORY
    if row.get("peak_unintended_support_n", 0) > 1:
        return "EXTRA LINK CONTACT", ember.RED
    if row.get("max_idle_drift_m", 0) > 0.15:
        return "DRIFT ABOVE TARGET", ember.RED
    return "STRICT TARGET MISSED", ember.RED


class Run:
    def __init__(self, record, directory):
        self.record = record
        self.spec = record["spec"]
        self.body = BODY_MAP[self.spec["body"]]
        assert sha(directory / record["trace"]) == record["trace_sha256"]
        assert sha(directory / record["model"]) == record["model_sha256"]
        with np.load(directory / record["trace"], allow_pickle=False) as trace:
            self.trace = {k: trace[k] for k in trace.files}
        self.model = mujoco.MjModel.from_binary_path(str(directory / record["model"]))
        ember.configure(self.model)
        self.model.vis.global_.offwidth, self.model.vis.global_.offheight = 1920, 1080
        self.data = mujoco.MjData(self.model)
        self.renderers = {}
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.index = 0

    @property
    def title(self):
        if self.spec["kind"] == "moving":
            return {
                "still": "STATIONARY DECK / CONTROL",
                "translate": "TRANSLATION",
                "yaw": "ROTATION / YAW",
                "heave": "VERTICAL MOTION",
                "rock": "ROCKING",
                "combined": "COMBINED PLATFORM MOTION",
            }[self.spec["surface"]]
        if self.spec["surface"] == "flat":
            return body_label(self.spec["body"])
        return surface_label(self.spec["surface"])

    def set_frame(self, frame):
        self.index = 2 * frame + 1
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
        if self.spec["kind"] == "moving":
            camera = moving_camera(self.model, self.data, kind)
            if kind == "follow" and size[1] == 340:
                camera.distance = 2.0
        elif kind == "head":
            camera = "head"
        else:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self.data.qpos[:3] - [0, 0, 0.06]
            camera.distance = 1.35 if kind == "follow" else 2.0
            if size[1] == 340 and self.spec["surface"] == "flat":
                camera.distance = 1.1
            camera.elevation = -23 if kind == "follow" else -65
            camera.azimuth = camera_azimuth(self.body)
            if self.spec["surface"].endswith(("fl", "rl")):
                camera.azimuth = -50
        renderer.update_scene(self.data, camera=camera, scene_option=self.option)
        ember.decorate(renderer.scene, self.model, self.data, self.body)
        return Image.fromarray(renderer.render())

    def close(self):
        for r in self.renderers.values():
            r.close()


def contact_lights(draw, run, x, y):
    for i, (leg, load) in enumerate(
        zip(LEGS, run.trace["tip_forces"][run.index], strict=True)
    ):
        px = x + i * 68
        draw.ellipse(
            (px, y + 4, px + 11, y + 15), fill=ember.GREEN if load > 1 else ember.STEEL
        )
        if leg in run.body.absent_legs:
            draw.line((px, y + 4, px + 11, y + 15), fill=ember.RED, width=2)
        draw.text((px + 17, y), leg, font=font(17), fill=ember.IVORY)


def canvas_for(title, subtitle, runs, frame, chapter):
    for run in runs:
        run.set_frame(frame)
    if len(runs) == 1:
        run = runs[0]
        canvas = ember.layout(
            run.image((1260, 840)),
            run.image((592, 404), "overview"),
            run.image((592, 404), "head"),
            title=title,
            subtitle=subtitle,
            seconds=run.data.time,
        )
        draw = ImageDraw.Draw(canvas)
        command = "WALKING" if run.spec["kind"] == "walking" else "HOLDING POSITION"
        if run.spec["transition"]:
            command = (
                "WALKING"
                if np.linalg.norm(run.trace["commands"][run.index]) > 0.01
                else "HOLDING POSITION"
            )
        draw.rectangle((44, 929, 466, 978), fill=ember.CHARCOAL)
        draw.text((60, 941), command, font=font(22), fill=ember.YELLOW)
        contact_lights(draw, run, 940, 949)
    else:
        canvas = Image.new("RGB", (1920, 1080), ember.CHARCOAL)
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((24, 26, 35, 118), fill=ember.YELLOW)
        draw.text((56, 24), "FIELD ROBOTICS / EMBER", font=font(20), fill=ember.ORANGE)
        draw.text((54, 51), title, font=font(37), fill=ember.IVORY)
        draw.text((56, 104), subtitle, font=font(21), fill=ember.IVORY)
        for i, run in enumerate(runs):
            x, y = 24 + 944 * (i % 2), 150 + 430 * (i // 2)
            draw.text((x + 10, y + 7), run.title, font=font(23), fill=ember.YELLOW)
            canvas.paste(run.image((928, 340)), (x, y + 43))
            if run.spec["kind"] == "walking":
                status = f"{run.data.qpos[0] - run.trace['qpos'][0, 0]:.2f} m / WALKING"
                color = ember.IVORY
            elif run.spec["transition"]:
                status = (
                    "WALKING"
                    if np.linalg.norm(run.trace["commands"][run.index]) > 0.01
                    else "HOLDING POSITION"
                )
                color = ember.IVORY
            elif run.spec["kind"] == "moving":
                status = f"DECK DRIFT {run.trace['relative_drift_m'][run.index] * 100:.1f} cm"
                color = ember.IVORY
            else:
                status, color = outcome_label(run.record)
                status = "TRIAL RESULT / " + status
            draw.text((x + 10, y + 390), status, font=font(18), fill=color)
            contact_lights(draw, run, x + 640, y + 390)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 1010, 1920, 1080), fill=ember.CHARCOAL)
    draw.line((24, 1010, 1896, 1010), fill=ember.STEEL, width=2)
    draw.text(
        (32, 1030), f"{runs[0].data.time:05.2f} s / 1x", font=font(24), fill=ember.IVORY
    )
    draw.text(
        (390, 1032),
        f"{chapter + 1:02d} / ADAPTIVE LOCOMOTION",
        font=font(21),
        fill=ember.YELLOW,
    )
    draw.text(
        (1230, 1032), "SAME FINAL POLICY / EVERY SCENE", font=font(21), fill=ember.IVORY
    )
    return canvas


def stats_card(outcomes=False):
    report = read(ROOT / "docs/locomotion/GPU_REPORT.json")
    assert report["final_checkpoint_sha256"] == EXPECTED
    canvas = Image.new("RGB", (1920, 1080), ember.CHARCOAL)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((64, 68, 76, 166), fill=ember.YELLOW)
    draw.text((106, 68), "ONE POLICY / EVERY SCENE", font=font(48), fill=ember.IVORY)
    draw.text(
        (108, 137),
        "INDEPENDENT FINAL EVALUATION"
        if outcomes
        else "TRAINED FROM RANDOM WEIGHTS / RTX 4090",
        font=font(24),
        fill=ember.ORANGE,
    )
    total = report["total"]
    if outcomes:
        e = report["final_evaluation"]
        cards = [
            (
                "WALKING",
                f"{e['walking_completed']}/{e['walking_trials']}",
                "Nine physical body configurations",
            ),
            (
                "STATIC BALANCE",
                f"{e['standing_healthy']['passed'] + e['standing_damaged']['passed']}/144",
                "Strict holds and command transitions",
            ),
            (
                "MOVING SUPPORTS",
                f"{e['moving']['passed']}/{e['moving']['trials']}",
                "Healthy robot / six platform conditions",
            ),
            ("REMAINED UPRIGHT", "204/204", "Eight static target misses retained"),
        ]
    else:
        seconds = round(total["training_seconds"])
        cards = [
            (
                "GPU TRAINING",
                f"{seconds // 60}m {seconds % 60:02d}s",
                "Includes physics and learning updates",
            ),
            (
                "PARALLEL WORLDS",
                f"{total['parallel_worlds']:,}",
                "Consistent through the entire curriculum",
            ),
            (
                "NEW EXPERIENCES",
                f"{total['transitions'] / 1e6:.2f}M",
                "One experience = one world acting for 20 ms",
            ),
            (
                "SIMULATED EXPERIENCE",
                f"{total['aggregate_simulated_hours']:.1f} h",
                "Aggregate across all worlds and episodes",
            ),
        ]
    for i, (label, value, note) in enumerate(cards):
        x, y = 104 + 900 * (i % 2), 260 + 290 * (i // 2)
        draw.text((x, y), label, font=font(24), fill=ember.YELLOW)
        draw.text((x, y + 50), value, font=font(72), fill=ember.IVORY)
        draw.text((x, y + 148), note, font=font(23), fill=ember.IVORY)
        draw.line((x, y + 206, x + 805, y + 206), fill=ember.STEEL, width=2)
    if outcomes:
        lines = [
            "All original healthy gait checks pass. One frozen checkpoint for every command.",
            "204 held-out test trials; this film shows 51 separate predetermined demonstration trials.",
            "Native MuJoCo evaluation / no real-hardware or unseen-terrain claim.",
        ]
    else:
        lines = [
            f"{total['transitions_per_training_second']:,.0f} new experiences/s   /   29,196 actor parameters",
            "MuJoCo Warp / 500 Hz physics / 50 Hz policy / PPO",
            "Setup, standalone evaluation and video production are timed separately.",
        ]
    for y, line in zip((870, 927, 980), lines, strict=True):
        draw.text((104, y), line, font=font(25 if y == 870 else 22), fill=ember.IVORY)
    return canvas


def render(directory, output, preview_only=False):
    capture_report = read(directory / "capture.json")
    assert capture_report["checkpoint_sha256"] == EXPECTED
    records = {r["key"]: r for r in capture_report["cases"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    previews = directory / "render_preview"
    previews.mkdir(exist_ok=True)
    stream = None
    if not preview_only:
        if output.exists():
            raise ValueError("Preserve existing videos; choose a new output name")
        stream = imageio_ffmpeg.write_frames(
            str(output),
            (1920, 1080),
            fps=FPS,
            codec="libx264",
            quality=8,
            macro_block_size=1,
            pix_fmt_out="yuv420p",
            output_params=["-movflags", "+faststart"],
        )
        stream.send(None)
    started, count, ledger = time.perf_counter(), 0, []
    try:
        for chapter, (title, subtitle, keys, seconds) in enumerate(chapters()):
            with ExitStack() as stack:
                runs = [Run(records[k], directory) for k in keys]
                for run in runs:
                    stack.callback(run.close)
                first = count
                for frame in (
                    [min(100, seconds * FPS - 1)]
                    if preview_only
                    else range(seconds * FPS)
                ):
                    canvas = canvas_for(title, subtitle, runs, frame, chapter)
                    if preview_only:
                        canvas.save(previews / f"chapter_{chapter:02d}.png")
                    else:
                        stream.send(np.asarray(canvas))
                        count += 1
                ledger.append(
                    {
                        "title": title,
                        "cases": keys,
                        "start_s": first / FPS,
                        "end_s": count / FPS,
                    }
                )
                print(
                    json.dumps({"rendered_chapter": chapter, "frames": count}),
                    flush=True,
                )
        for title, is_outcome in (
            ("GPU TRAINING / MEASURED SCALE", False),
            ("FINAL CHECKPOINT / HELD-OUT RESULTS", True),
        ):
            canvas = stats_card(is_outcome)
            canvas.save(
                DEST / ("results_v1.png" if is_outcome else "training_stats_v1.png")
            )
            first = count
            if stream:
                for _ in range(6 * FPS):
                    stream.send(np.asarray(canvas))
                    count += 1
            ledger.append(
                {
                    "title": title,
                    "cases": [],
                    "start_s": first / FPS,
                    "end_s": count / FPS,
                }
            )
    finally:
        if stream:
            stream.close()
    if not preview_only:
        write(
            output.with_suffix(".json"),
            {
                "checkpoint_sha256": EXPECTED,
                "checkpoint_count": 1,
                "theme": "ember",
                "playback_speed": 1,
                "fps": FPS,
                "frames": count,
                "dimensions": [1920, 1080],
                "duration_s": count / FPS,
                "chapters": ledger,
                "cases": capture_report["cases"],
                "capture_manifest": str((directory / "capture.json").relative_to(ROOT)),
                "capture_manifest_sha256": sha(directory / "capture.json"),
                "capture_save_seconds": capture_report["capture_save_seconds"],
                "render_encode_seconds": time.perf_counter() - started,
                "new_training_seconds": 0,
                "training_report_sha256": sha(ROOT / "docs/locomotion/GPU_REPORT.json"),
                "physics_backend": "mjbatch",
                "state_indices": "1,3,... at 50 Hz, 0.04 s increments to each exact endpoint",
                "source_commit": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "video_sha256": sha(output),
                "visual_inspection": "pending",
            },
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--part", choices=("capture", "preview", "render", "all"), default="all"
    )
    parser.add_argument("--trace-root", type=Path, default=TRACE)
    parser.add_argument(
        "--output", type=Path, default=DEST / "adaptive_dog_complete_v1.mp4"
    )
    args = parser.parse_args()
    if args.part in ("capture", "all"):
        capture(args.trace_root)
    if args.part in ("render", "all", "preview"):
        render(args.trace_root, args.output, args.part == "preview")


if __name__ == "__main__":
    main()
