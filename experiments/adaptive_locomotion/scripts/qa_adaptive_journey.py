"""Decode the complete journey film and export chapter/transition QA images."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import ROOT


def main(video):
    report = json.loads(video.with_suffix(".json").read_text())
    assert hashlib.sha256(video.read_bytes()).hexdigest() == report["video_sha256"]
    directory = ROOT / "outputs/locomotion/graphite/journey_qa" / video.stem
    speed = report["playback_speed"]
    assert speed in (1, 2)
    expected_duration = 118 / speed + 6
    hero = report["chapters"][-2]
    thumbnail = round((hero["start_s"] + hero["end_s"]) * 25 / 2) - 1
    directory.mkdir(parents=True, exist_ok=True)
    samples = set()
    for chapter in report["chapters"]:
        start, end = (round(chapter[k] * 25) for k in ("start_s", "end_s"))
        samples.update((start, (start + end - 1) // 2, end - 1))
    samples = sorted(samples)
    sheets = [
        Image.new("RGB", (1920, 1140), "#08090b")
        for _ in range((len(samples) + 8) // 9)
    ]
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(frames)
    count = 0
    for index, raw in enumerate(frames):
        count += 1
        assert len(raw) == 1920 * 1080 * 3
        if index in samples:
            frame = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3))
            frame.save(directory / f"frame_{index:04d}.png")
            slot = samples.index(index)
            sheet = sheets[slot // 9]
            x, y = (slot % 3) * 640, ((slot % 9) // 3) * 380
            sheet.paste(frame.resize((640, 360)), (x, y + 20))
            ImageDraw.Draw(sheet).text(
                (x + 8, y + 4),
                f"Frame {index} / video {index / 25:.2f} s",
                fill="white",
            )
        if index == thumbnail:
            Image.fromarray(np.frombuffer(raw, np.uint8).reshape(1080, 1920, 3)).save(
                video.with_suffix(".png")
            )
    for i, sheet in enumerate(sheets):
        sheet.save(directory / f"sheet_{i}.png")
    checks = {
        "complete_decode": count == report["frames"] == round(expected_duration * 25),
        "dimensions": metadata["size"] == (1920, 1080),
        "fps": metadata["fps"] == 25,
        "duration": abs(metadata["duration"] - expected_duration) < 0.05,
        "playback_speed": speed in (1, 2),
        "three_training_stages": report["checkpoint_count"] == 3,
        "all_accepted_runs": len(report["cases"]) == 39,
        "nine_walking_bodies": sum(
            c["source_manifest"].endswith("adaptive_dog_v1.json")
            for c in report["cases"]
        )
        == 9,
        "all_original_standing_cases": sum(
            c["source_manifest"].endswith("adaptive_standing_v2.json")
            for c in report["cases"]
        )
        == 17,
        "eight_reviewed_terrain_cases": sum(
            c["source_manifest"].endswith("adaptive_standing_v3.json")
            for c in report["cases"]
        )
        == 8,
        "five_trained_platform_motions": sum(
            c["source_manifest"].endswith("moving_supports_v1.json")
            for c in report["cases"]
        )
        == 5,
        "no_new_training_or_physics": report["new_training_seconds"]
        == report["new_physics_steps"]
        == 0,
    }
    qa = {
        "video_sha256": report["video_sha256"],
        "checks": checks,
        "decoded_frames": count,
        "sampled_frames": samples,
        "visual_inspection": "pending",
        "final_card_seconds": 6,
        "physical_footage_seconds": 118 / speed,
        "source_timeline_seconds": 118,
    }
    video.with_suffix(".qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    raise SystemExit(main(parser.parse_args().video))
