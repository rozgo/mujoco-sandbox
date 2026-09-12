"""Decode every balance video frame and save chapter/transition inspection images."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import ROOT


def main(video):
    video = Path(video)
    provenance = json.loads(video.with_suffix(".json").read_text())
    assert hashlib.sha256(video.read_bytes()).hexdigest() == provenance["video_sha256"]
    samples = (0, 75, 125, 200, 299, 300, 425, 549, 550, 675, 799, 800, 925, 1049, 1050)
    directory = ROOT / "outputs/locomotion/standing/video_qa" / video.stem
    directory.mkdir(parents=True, exist_ok=True)
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(frames)
    width, height = metadata["size"]
    count = 0
    board = Image.new("RGB", (1920, 1940), "#091219")
    for index, raw in enumerate(frames):
        count += 1
        assert len(raw) == width * height * 3
        if index in samples:
            frame = Image.fromarray(
                np.frombuffer(raw, np.uint8).reshape(height, width, 3)
            )
            frame.save(directory / f"frame_{index:04d}.png")
            k = samples.index(index)
            board.paste(frame.resize((640, 360)), ((k % 3) * 640, (k // 3) * 388 + 28))
            ImageDraw.Draw(board).text(
                ((k % 3) * 640 + 12, (k // 3) * 388 + 6), f"Frame {index}", fill="white"
            )
            if index == 425:
                frame.save(video.with_suffix(".png"))
    board.save(directory / "contact_sheet.png")
    checks = {
        "every_frame_decoded": count == provenance["frames"],
        "dimensions": (width, height) == tuple(provenance["dimensions"]),
        "fps": metadata["fps"] == provenance["fps"],
        "duration": abs(metadata["duration"] - provenance["duration_s"]) < 0.05,
    }
    report = {
        "video_sha256": provenance["video_sha256"],
        "encoding_checks": checks,
        "decoded_frames": count,
        "visual_inspection": "Pending inspection of decoded samples",
        "cases_passed": sum(c["passed"] for c in provenance["cases"]),
        "cases_total": len(provenance["cases"]),
        "all_torque_limits": all(
            r["peak_torque_limit_ratio"] <= 1.0001
            for c in provenance["cases"]
            for r in c["rows"]
        ),
        "max_sampled_penetration_m": max(
            r["max_penetration_m"] for c in provenance["cases"] for r in c["rows"]
        ),
    }
    video.with_suffix(".qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    raise SystemExit(main(parser.parse_args().video))
