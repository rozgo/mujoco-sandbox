"""Verify the one-policy film, saved physical traces and encoded chapter boundaries."""

import argparse
import time
from collections import Counter
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from adaptive_locomotion.bodies import ROOT
from PIL import Image, ImageDraw
from record_unified_showcase import (
    CHECKPOINT,
    EXPECTED,
    FPS,
    TRACE,
    chapters,
    read,
    sha,
    specifications,
    write,
)


def main(video, trace_root):
    started = time.perf_counter()
    report = read(video.with_suffix(".json"))
    specs = specifications()
    records = {r["key"]: r for r in report["cases"]}
    duration = sum(c[3] for c in chapters()) + 12
    checks = {
        "video_hash": sha(video) == report["video_sha256"],
        "one_final_checkpoint": report["checkpoint_count"] == 1
        and report["checkpoint_sha256"] == EXPECTED == sha(CHECKPOINT)
        and all(r["checkpoint_sha256"] == EXPECTED for r in records.values()),
        "all_51_conditions": set(records) == set(specs) and len(report["cases"]) == 51,
        "each_condition_shown_once": Counter(
            k for c in report["chapters"] for k in c["cases"]
        )
        == Counter(specs.keys()),
        "capture_manifest_hash": sha(trace_root / "capture.json")
        == report["capture_manifest_sha256"],
        "training_report_hash": sha(ROOT / "docs/locomotion/GPU_REPORT.json")
        == report["training_report_sha256"],
        "one_x_native_ember": report["playback_speed"] == 1
        and report["theme"] == "ember"
        and report["physics_backend"] == "mjbatch",
        "no_new_training": report["new_training_seconds"] == 0,
    }
    if report.get("environment") == "industrial":
        previous = read(video.parent / "adaptive_dog_complete_v1.json")
        checks["same_physical_runs_as_previous_film"] = (
            report["cases"] == previous["cases"]
            and report["capture_manifest_sha256"] == previous["capture_manifest_sha256"]
            and report["chapters"] == previous["chapters"]
        )
        checks["previous_film_preserved"] = (
            sha(video.parent / "adaptive_dog_complete_v1.mp4")
            == previous["video_sha256"]
        )
        checks["approved_environment_preview"] = (
            sha(video.parent / "environment_v2.png")
            == report["approved_environment_preview_sha256"]
        )
        checks["no_new_render_physics"] = report["new_physics_steps_during_render"] == 0
    state_count, misses, upright = 0, [], 0
    for key, record in records.items():
        assert record["spec"] == specs[key]
        assert sha(trace_root / record["model"]) == record["model_sha256"]
        assert sha(trace_root / record["trace"]) == record["trace_sha256"]
        with np.load(trace_root / record["trace"], allow_pickle=False) as trace:
            n = record["spec"]["seconds"] * 50
            assert n == record["states"] == len(trace["time"])
            assert np.allclose(trace["time"], np.arange(1, n + 1) * 0.02)
            assert all(np.isfinite(trace[k]).all() for k in trace.files)
            assert trace["qpos"].shape[0] == trace["qvel"].shape[0] == n
            state_count += n
        row = record["outcome"]["rows"][0]
        upright += int(row["survived"])
        if not row.get("passed", row.get("completed_with_allowed_support")):
            misses.append({"key": key, "result": row})
    checks["physical_trace_hashes_and_timelines"] = state_count == 27300
    plan = report["chapters"]
    cursor = 0
    for c in plan:
        assert c["start_s"] == cursor
        cursor = c["end_s"]
    checks["contiguous_chapters"] = cursor == duration
    directory = trace_root / "encoded_qa" / video.stem
    directory.mkdir(parents=True, exist_ok=True)
    samples = set()
    for c in plan:
        start, end = (round(c[k] * FPS) for k in ("start_s", "end_s"))
        samples.update((start, (start + end - 1) // 2, end - 1))
        if c["cases"] and c["cases"][0].startswith("transition"):
            samples.update(start + round(t * FPS) for t in (3.96, 4.04, 7.96, 8.04))
    samples = sorted(samples)
    slots = {frame: slot for slot, frame in enumerate(samples)}
    sheets = [
        Image.new("RGB", (1920, 1140), "#1F1F1F")
        for _ in range((len(samples) + 8) // 9)
    ]
    hero = plan[-3]
    thumbnail = round((hero["start_s"] + hero["end_s"]) * FPS / 2) - 1
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(frames)
    count = 0
    for index, raw in enumerate(frames):
        count += 1
        assert len(raw) == 1920 * 1080 * 3
        if index in slots or index == thumbnail:
            frame = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3))
            if index == thumbnail:
                frame.save(video.with_suffix(".png"))
            if index in slots:
                frame.save(directory / f"frame_{index:04d}.png")
                slot = slots[index]
                sheet = sheets[slot // 9]
                x, y = (slot % 3) * 640, ((slot % 9) // 3) * 380
                sheet.paste(frame.resize((640, 360)), (x, y + 20))
                ImageDraw.Draw(sheet).text(
                    (x + 8, y + 4),
                    f"Frame {index} / video {index / FPS:.2f} s",
                    fill="white",
                )
    for i, sheet in enumerate(sheets):
        sheet.save(directory / f"sheet_{i}.png")
    checks.update(
        complete_decode=count == report["frames"] == round(duration * FPS),
        dimensions=metadata["size"] == (1920, 1080),
        fps=metadata["fps"] == FPS,
        duration=abs(metadata["duration"] - duration) < 0.05,
    )
    qa = {
        "video_sha256": report["video_sha256"],
        "checks": checks,
        "decoded_frames": count,
        "duration_s": duration,
        "sampled_frames": samples,
        "captured_control_states": state_count,
        "demonstration_trials": len(records),
        "demonstration_upright": upright,
        "demonstration_strict_passes": len(records) - len(misses),
        "retained_demonstration_misses": misses,
        "held_out_evaluation_trials_separate_from_film": 204,
        "visual_inspection": "pending",
        "qa_seconds": time.perf_counter() - started,
    }
    write(video.with_suffix(".qa.json"), qa)
    print(qa)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--trace-root", type=Path, default=TRACE)
    args = parser.parse_args()
    raise SystemExit(main(args.video, args.trace_root))
