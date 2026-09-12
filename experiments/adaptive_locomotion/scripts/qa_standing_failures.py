"""Decode every failure-review frame and extract chapter/peak inspection images."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from adaptive_locomotion.bodies import ROOT
from PIL import Image, ImageDraw


def main(video):
    video = Path(video)
    report = json.loads(video.with_suffix(".json").read_text())
    assert hashlib.sha256(video.read_bytes()).hexdigest() == report["video_sha256"]
    fps = report["fps"]
    samples, featured = set(), {}
    for i, chapter in enumerate(report["chapters"]):
        start = round(chapter["start_s"] * fps)
        end = round((chapter["start_s"] + chapter["duration_s"]) * fps) - 1
        inspect = end - fps
        samples.update((start, start + fps, inspect, end))
        featured[inspect] = i
    directory = ROOT / "outputs/locomotion/standing/failure_review/qa"
    directory.mkdir(parents=True, exist_ok=True)
    board = Image.new("RGB", (1920, 388 * ((len(featured) + 2) // 3)), "#091219")
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(frames)
    width, height = metadata["size"]
    count = 0
    for index, raw in enumerate(frames):
        count += 1
        assert len(raw) == width * height * 3
        if index not in samples:
            continue
        frame = Image.fromarray(np.frombuffer(raw, np.uint8).reshape(height, width, 3))
        frame.save(directory / f"frame_{index:04d}.png")
        if index in featured:
            i = featured[index]
            x, y = (i % 3) * 640, (i // 3) * 388
            board.paste(frame.resize((640, 360)), (x, y + 28))
            ImageDraw.Draw(board).text(
                (x + 10, y + 6),
                f"Case {i + 1} | video {index / fps:.2f} s",
                fill="white",
            )
            if i == 8:
                frame.save(video.with_suffix(".png"))
    board.save(directory / "contact_sheet.png")
    checks = {
        "every_frame_decoded": count == report["frames"],
        "dimensions": (width, height) == tuple(report["dimensions"]),
        "fps": metadata["fps"] == fps,
        "duration": abs(metadata["duration"] - report["duration_s"]) < 0.05,
    }
    cases = report["cases"]
    qa = {
        "video_sha256": report["video_sha256"],
        "encoding_checks": checks,
        "decoded_frames": count,
        "sampled_frames": sorted(samples),
        "visual_inspection": "Pending inspection of decoded samples",
        "selected_conditions": len(cases),
        "reproduced_failure_count": sum(not c["repeated_row"]["passed"] for c in cases),
        "upright_count": sum(c["repeated_row"]["survived"] for c in cases),
        "every_selected_row_matches_original_exactly": all(
            c["repeated_row"] == c["original_row"] for c in cases
        ),
        "all_torque_limits": all(
            c["repeated_row"]["peak_torque_limit_ratio"] <= 1.0001 for c in cases
        ),
        "max_sampled_penetration_m": max(
            c["repeated_row"]["max_penetration_m"] for c in cases
        ),
        "new_training_seconds": 0,
        "physics_backend": "CPU MuJoCo via mjbatch",
    }
    video.with_suffix(".qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    raise SystemExit(main(parser.parse_args().video))
