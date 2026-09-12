"""Decode every comparison frame and retain inspectable chapter samples."""

import argparse
import hashlib
import json

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from adaptive_locomotion.bodies import ROOT


def main(prefix="scaled"):
    video = ROOT / f"previews/locomotion/warp_{prefix}_comparison.mp4"
    provenance = json.loads(video.with_suffix(".json").read_text())
    assert hashlib.sha256(video.read_bytes()).hexdigest() == provenance["video_sha256"]
    samples = (0, 150, 299, 300, 450, 599, 600, 750, 899)
    output = ROOT / f"outputs/locomotion/warp_{prefix}_video_qa"
    output.mkdir(parents=True, exist_ok=True)
    frames = imageio_ffmpeg.read_frames(str(video), pix_fmt="rgb24")
    metadata = next(frames)
    width, height = metadata["size"]
    count = 0
    sheet = Image.new("RGB", (1440, 1152), "#091219")
    for index, frame in enumerate(frames):
        count += 1
        assert len(frame) == width * height * 3
        if index in samples:
            image = Image.fromarray(
                np.frombuffer(frame, np.uint8).reshape(height, width, 3)
            )
            image.save(output / f"frame_{index:04d}.png")
            tile = image.resize((480, 360))
            n = samples.index(index)
            sheet.paste(tile, ((n % 3) * 480, (n // 3) * 384 + 24))
            ImageDraw.Draw(sheet).text(
                ((n % 3) * 480 + 8, (n // 3) * 384 + 5), f"Frame {index}", fill="white"
            )
            if index == 150:
                image.save(video.with_suffix(".png"))
    sheet.save(output / "contact_sheet.png")
    checks = {
        "all_900_frames_decoded": count == 900,
        "resolution_1920x1440": (width, height) == (1920, 1440),
        "fps_25": metadata["fps"] == 25,
        "duration_36_seconds": abs(metadata["duration"] - 36) < 0.05,
        "original_torque_limits": all(
            row["peak_torque_limit_ratio"] <= 1 + 1e-6
            for case in provenance["cases"]
            for row in case["outcome"]["rows"]
        ),
        "all_upright_with_valid_support": all(
            row["survived"] and row["bad_support_control_windows"] == 0
            for case in provenance["cases"]
            for row in case["outcome"]["rows"]
        ),
        "original_8mm_sampled_penetration_target": all(
            case["penetration_below_original_8mm_target"]
            for case in provenance["cases"]
        ),
    }
    report = {
        "video_sha256": provenance["video_sha256"],
        "decoded_frames": count,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "sampled_frame_indices": list(samples),
        "visual_inspection": "Pending human/agent inspection of the decoded sample images",
        "maximum_sampled_penetration_m": max(
            c["maximum_sampled_penetration_m"] for c in provenance["cases"]
        ),
        "scope": "Encoding checks and recorded physical limits, separate from training acceptance. Penetration is reconstructed every 20 ms, not a continuous bound.",
    }
    video.with_suffix(".qa.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", choices=("matched", "scaled"), default="scaled")
    main(**vars(parser.parse_args()))
