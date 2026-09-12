"""Close views of one predetermined failed CPU holdout trial per condition."""

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from adaptive_locomotion.bodies import CONTROL_DT, LEGS, ROOT, allowed_support_names
from adaptive_locomotion.presentation import configure, damage_markers
from adaptive_locomotion.record import font
from adaptive_locomotion.standing_evaluate import BODY_MAP, run_case
from adaptive_locomotion.standing_record import ground_marks
from adaptive_locomotion.standing_surfaces import surface_height
from adaptive_locomotion.train import load_checkpoint
from PIL import Image, ImageDraw

CHECKPOINT = ROOT / "assets/locomotion/checkpoints/standing/consolidate_120s_seed12.pt"
OUTPUT = ROOT / "previews/locomotion/standing/failure_review.mp4"
FPS = 25
ORANGE, TEAL, WHITE = "#ffad82", "#91d8d0", "#eef4f5"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selections():
    selected = []
    for name in ("healthy_cpu", "damaged_cpu"):
        path = ROOT / f"docs/locomotion/standing/final/{name}.json"
        report = json.loads(path.read_text())
        assert report["checkpoint_sha256"] == sha(CHECKPOINT)
        for case in report["cases"]:
            failed = [row for row in case["rows"] if not row["passed"]]
            if failed:
                selected.append(
                    {
                        "body": case["body"],
                        "surface": case["surface"],
                        "transition": case["transition"],
                        "seconds": case["seconds"],
                        "seed": case["seed"],
                        "trials": case["trials"],
                        "selected_trial": failed[0]["trial"],
                        "original_row": failed[0],
                        "reference": str(path.relative_to(ROOT)),
                        "reference_sha256": sha(path),
                    }
                )
    return selected


def capture(directory):
    manifest = directory / "capture.json"
    selected = selections()
    if manifest.exists():
        saved = json.loads(manifest.read_text())
        assert saved["selection"] == selected
        assert saved["checkpoint_sha256"] == sha(CHECKPOINT)
        return saved
    directory.mkdir(parents=True, exist_ok=True)
    net, _ = load_checkpoint(CHECKPOINT)
    cases = []
    start = time.perf_counter()
    for index, case in enumerate(selected):
        result, frames, model = run_case(
            net,
            case["body"],
            case["surface"],
            trials=case["trials"],
            seconds=case["seconds"],
            seed=case["seed"],
            physics_backend="mjbatch",
            transition=case["transition"],
            capture=True,
            capture_trial=case["selected_trial"],
        )
        path = directory / f"case_{index:02d}.npz"
        np.savez_compressed(
            path, **{k: np.asarray([f[k] for f in frames]) for k in frames[0]}
        )
        model_path = directory / f"case_{index:02d}.mjb"
        buffer = np.empty(mujoco.mj_sizeModel(model), np.uint8)
        mujoco.mj_saveModel(model, buffer=buffer)
        model_path.write_bytes(buffer.tobytes())
        row = result["rows"][case["selected_trial"]]
        cases.append(
            dict(
                **case,
                repeated_result=result,
                repeated_row=row,
                trajectory=str(path.relative_to(ROOT)),
                trajectory_sha256=sha(path),
                model=str(model_path.relative_to(ROOT)),
                model_sha256=sha(model_path),
            )
        )
        print(
            json.dumps(
                {
                    "captured": index + 1,
                    "body": case["body"],
                    "surface": case["surface"],
                    "transition": case["transition"],
                    "trial": case["selected_trial"],
                    "repeat_passed": row["passed"],
                }
            ),
            flush=True,
        )
    saved = {
        "checkpoint_sha256": sha(CHECKPOINT),
        "selection": selected,
        "cases": cases,
        "capture_seconds": time.perf_counter() - start,
        "selection_rule": "First failing trial in each failing CPU holdout condition; no retuning or retries.",
        "capture_note": "Repeated the original four-world CPU evaluation; recorded the selected world. These are new diagnostic captures, not the original video trajectories.",
    }
    manifest.write_text(json.dumps(saved, indent=2) + "\n")
    return saved


class Review:
    def __init__(self, case):
        self.case = case
        self.body = BODY_MAP[case["body"]]
        path, model_path = ROOT / case["trajectory"], ROOT / case["model"]
        assert sha(path) == case["trajectory_sha256"]
        assert sha(model_path) == case["model_sha256"]
        with np.load(path, allow_pickle=False) as values:
            self.trace = {k: values[k] for k in values.files}
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        names = [
            self.model.sensor(i).name.removeprefix("support_")
            for i in range(self.model.nsensor)
            if self.model.sensor(i).name.startswith("support_")
        ]
        bad = np.array([n not in allowed_support_names(self.body) for n in names])
        forces = self.trace["support_peak_forces_n"][:, bad]
        k, j = np.unravel_index(np.argmax(forces), forces.shape)
        self.bad_force = forces.max(1)
        self.bad_name = np.asarray(names)[bad][j]
        self.bad_geom = self.model.geom(self.bad_name).id
        self.peak_time = float(self.trace["time"][k])
        self.keyframe = k if forces.max() > 1 else len(forces) - 1
        self.focus_geom = self.bad_geom if forces.max() > 1 else None
        if forces.max() <= 1 and case["repeated_row"]["max_penetration_m"] > 0.008:
            self.keyframe = int(np.argmax(self.trace["sampled_penetration_m"]))
            self.data.qpos[:] = self.trace["qpos"][self.keyframe]
            mujoco.mj_forward(self.model, self.data)
            contact = self.data.contact[np.argmin(self.data.contact.dist)]
            self.focus_geom = next(
                int(g)
                for g in (contact.geom1, contact.geom2)
                if self.model.geom_bodyid[g] != 0
            )
        configure(self.model)
        self.model.vis.global_.offwidth = 1920
        self.model.vis.global_.offheight = 1080
        self.option = mujoco.MjvOption()
        self.option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = False
        self.main = mujoco.Renderer(self.model, height=690, width=1280)
        self.inset = mujoco.Renderer(self.model, height=345, width=640)

    def close(self):
        self.main.close()
        self.inset.close()

    def camera(self, kind):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.data.qpos[:3]
        target = (
            self.bad_name[:2].lower()
            if self.bad_force.max() > 1
            else self.case["surface"][-2:]
            if self.case["surface"].startswith("gap_")
            else self.case["body"][-2:]
        )
        cam.azimuth = -50 if target in ("fl", "rl") else 50
        cam.distance, cam.elevation = 1.5, -22
        if kind == "overhead":
            cam.distance, cam.elevation = 1.6, -75
        elif kind == "detail":
            cam.distance, cam.elevation = 0.80, -15
            if self.focus_geom is not None:
                cam.lookat[:] = self.data.geom_xpos[self.focus_geom]
            elif self.case["surface"].startswith("gap_"):
                cam.lookat[:] = self.trace["tip_positions"][self.k][
                    LEGS.index(target.upper())
                ]
            else:
                cam.distance = 1.25
                cam.elevation = -10
        return cam

    def decorate(self, scene):
        ground_marks(scene, self.case["surface"])
        damage_markers(scene, self.model, self.data, self.body)
        if self.focus_geom is not None:
            # Observer marker identifies a link, not an invented contact point.
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.full(3, 0.025),
                self.data.geom_xpos[self.focus_geom],
                np.eye(3).ravel(),
                np.array([1, 0.12, 0.10, 0.65], np.float32),
            )
            scene.ngeom += 1
        if np.linalg.norm(self.trace["commands"][self.k]) <= 0.05:
            anchor = self.trace["hold_anchor"][self.k]
            points = []
            for angle in np.linspace(0, 2 * np.pi, 33):
                xy = anchor + 0.15 * np.array([np.cos(angle), np.sin(angle)])
                z = float(surface_height(self.case["surface"], *xy)) + 0.004
                points.append(np.r_[xy, z])
            for a, b in itertools.pairwise(points):
                if abs(a[2] - b[2]) > 0.05:
                    continue
                geom = scene.geoms[scene.ngeom]
                mujoco.mjv_initGeom(
                    geom,
                    mujoco.mjtGeom.mjGEOM_LINE,
                    np.ones(3),
                    a,
                    np.eye(3).ravel(),
                    np.array([0.3, 0.9, 0.9, 1], np.float32),
                )
                mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_LINE, 2, a, b)
                scene.ngeom += 1

    def reasons(self):
        row = self.case["repeated_row"]
        lines = []
        if row["peak_unintended_support_n"] > 1:
            lines.append(
                f"Wrong-link load: {row['peak_unintended_support_n']:.1f} N > 1 N  |  "
                f"{self.bad_name}  |  peak interval ends {self.peak_time:.2f} s"
            )
        if row["max_idle_drift_m"] > 0.15:
            lines.append(f"Hold drift: {row['max_idle_drift_m'] * 100:.1f} cm > 15 cm")
        if row["max_idle_tilt_deg"] > 20:
            lines.append(
                f"Body tilt: {row['max_idle_tilt_deg']:.1f} degrees > 20 degrees"
            )
        air = row["unsupported_foot_fraction"]
        if air is not None and air < 0.95:
            lines.append(
                f"Designated foot off support: {100 * air:.1f}% of hold < 95% required"
            )
        if row["max_penetration_m"] > 0.008:
            peak = int(np.argmax(self.trace["sampled_penetration_m"]))
            lines.append(
                f"Sampled penetration: {row['max_penetration_m'] * 1000:.2f} mm > 8 mm"
                f"  |  peak at {self.trace['time'][peak]:.2f} s"
            )
        if row["mean_idle_speed_mps"] > 0.06:
            lines.append(
                f"Mean hold speed: {row['mean_idle_speed_mps']:.3f} m/s > 0.06 m/s"
            )
        if not lines:
            lines.append("Original trial was flagged; this repeated capture passed.")
        return lines

    def frame(self, k, index, total, paused=False):
        self.k = k
        trace, data, model = self.trace, self.data, self.model
        data.qpos[:], data.qvel[:], data.ctrl[:] = (
            trace["qpos"][k],
            trace["qvel"][k],
            trace["ctrl"][k],
        )
        data.time = trace["time"][k]
        mujoco.mj_forward(model, data)
        canvas = Image.new("RGB", (1920, 1080), "#091219")
        draw = ImageDraw.Draw(canvas)
        case = self.case
        title = f"{case['body']} / {case['surface']}".upper().replace("_", " ")
        if case["transition"]:
            title += " / WALK-STOP-WALK"
        draw.text(
            (24, 14),
            f"FAILURE REVIEW  {index + 1:02d}/{total}   |   {title}",
            font=font(31),
            fill=TEAL,
        )
        playback = "PAUSED / PEAK INSPECTION" if paused else "Full trial at 1x speed"
        draw.text(
            (24, 57),
            f"Same frozen policy  |  CPU holdout seed {case['seed']}, trial {case['selected_trial']}  |  {playback}",
            font=font(22),
            fill=WHITE,
        )
        for kind, renderer, xy in (
            ("follow", self.main, (0, 110)),
            ("overhead", self.inset, (1280, 110)),
            ("detail", self.inset, (1280, 455)),
        ):
            renderer.update_scene(
                data, camera=self.camera(kind), scene_option=self.option
            )
            self.decorate(renderer.scene)
            canvas.paste(Image.fromarray(renderer.render()), xy)
            draw.text((xy[0] + 16, xy[1] + 10), kind.upper(), font=font(20), fill=WHITE)
        draw.text(
            (24, 811), "FLAGGED METRICS / COMPLETE TRIAL", font=font(23), fill=ORANGE
        )
        for i, line in enumerate(self.reasons()):
            draw.text((24, 846 + i * 33), line, font=font(23), fill=WHITE)
        drift = np.linalg.norm(data.qpos[:2] - trace["hold_anchor"][k])
        tilt = np.rad2deg(
            np.arccos(np.clip(data.xmat[model.body("base").id][8], -1, 1))
        )
        draw.text(
            (1380, 812),
            f"t = {data.time:.2f} s  |  {'PAUSED' if paused else '1x'}",
            font=font(26),
            fill=WHITE,
        )
        draw.text(
            (1380, 853),
            "UPRIGHT" if trace["alive"][k] else "FALL / NO RESET",
            font=font(25),
            fill=TEAL,
        )
        draw.text(
            (1380, 893),
            f"Now: drift {drift * 100:.1f} cm / tilt {tilt:.1f} deg",
            font=font(21),
            fill=WHITE,
        )
        draw.text(
            (1380, 927),
            f"Command: {trace['commands'][k, 0]:.2f} m/s",
            font=font(21),
            fill=WHITE,
        )
        for j, leg in enumerate(LEGS):
            x = 1380 + j * 122
            loaded = trace["tip_forces"][k, j] > 1
            draw.ellipse((x, 970, x + 12, 982), fill=TEAL if loaded else "#46535b")
            draw.text((x + 19, 962), leg, font=font(21), fill=WHITE)
        draw.text(
            (24, 1009),
            "Cyan ring: 15 cm hold region  |  Red marker: worst flagged link  |  Orange: limb removal  |  Dots: terminal load",
            font=font(21),
            fill=TEAL,
        )
        draw.text(
            (24, 1043),
            "2 ms physics / 20 ms control  |  Support checked each physics step; penetration sampled at 50 Hz  |  Observer cameras",
            font=font(20),
            fill="#b6c8ce",
        )
        return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--preview-only", action="store_true")
    args = parser.parse_args()
    directory = ROOT / "outputs/locomotion/standing" / args.output.stem
    saved = capture(directory)
    preview = Image.new(
        "RGB", (1920, 360 * ((len(saved["cases"]) + 2) // 3)), "#091219"
    )
    if args.preview_only:
        for index, case in enumerate(saved["cases"]):
            review = Review(case)
            try:
                frame = review.frame(review.keyframe, index, len(saved["cases"]))
                frame.save(directory / f"preview_{index:02d}.png")
                preview.paste(
                    frame.resize((640, 360)), ((index % 3) * 640, (index // 3) * 360)
                )
            finally:
                review.close()
        preview.save(directory / "preview.png")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        (1920, 1080),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    writer.send(None)
    chapters, count = [], 0
    started = time.perf_counter()
    try:
        for index, case in enumerate(saved["cases"]):
            review = Review(case)
            pause = 2 if review.focus_geom is not None else 0
            chapters.append(
                {
                    "start_s": count / FPS,
                    "duration_s": case["seconds"] + pause,
                    "physical_trial_s": case["seconds"],
                    "inspection_pause_s": pause,
                    "body": case["body"],
                    "surface": case["surface"],
                    "transition": case["transition"],
                    "reasons": review.reasons(),
                }
            )
            try:
                for i in range(round(case["seconds"] * FPS)):
                    k = min(round(i / FPS / CONTROL_DT), len(review.trace["time"]) - 1)
                    writer.send(np.asarray(review.frame(k, index, len(saved["cases"]))))
                    count += 1
                if pause:
                    still = np.asarray(
                        review.frame(
                            review.keyframe, index, len(saved["cases"]), paused=True
                        )
                    )
                    for _ in range(pause * FPS):
                        writer.send(still)
                        count += 1
            finally:
                review.close()
            print(
                json.dumps({"rendered": index + 1, "total": len(saved["cases"])}),
                flush=True,
            )
    finally:
        writer.close()
    report = dict(
        **saved,
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        video_sha256=sha(args.output),
        chapters=chapters,
        frames=count,
        fps=FPS,
        duration_s=count / FPS,
        dimensions=[1920, 1080],
        render_seconds=time.perf_counter() - started,
        new_training_seconds=0,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
