"""Decode the complete moving-support film and extract declared review frames."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", type=Path)
    p.add_argument("--output", type=Path, default=Path("build/moving_video_qa"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads(args.video.with_suffix(".json").read_text())
    chosen = [
        0,
        124,
        249,
        250,
        374,
        499,
        624,
        749,
        874,
        999,
        1124,
        1249,
        1250,
        1374,
        1499,
        1500,
        1599,
    ]
    reader = imageio_ffmpeg.read_frames(str(args.video), pix_fmt="rgb24")
    metadata = next(reader)
    assert metadata["size"] == (1920, 1080)
    assert metadata["fps"] == report["fps"] == 25
    assert abs(metadata["duration"] - report["duration_s"]) < 0.05
    sheet = Image.new("RGB", (1600, 5 * 245), "#091219")
    count = 0
    entries = []
    for i, frame in enumerate(reader):
        count += 1
        if i in chosen:
            img = Image.frombytes("RGB", metadata["size"], frame)
            path = args.output / f"frame_{i:04d}.png"
            img.save(path)
            panel = img.resize((400, 225))
            j = len(entries)
            sheet.paste(panel, ((j % 4) * 400, (j // 4) * 245))
            ImageDraw.Draw(sheet).text(
                ((j % 4) * 400 + 8, (j // 4) * 245 + 228),
                f"{i / 25:.2f} s / frame {i}",
                fill="white",
            )
            entries.append(str(path))
    assert count == report["frames"] == 1600
    sheet.save(args.output / "contact_sheet.png")
    # Verify physical timestamps used by the 50->25 Hz selection include the
    # complete endpoint; this is presentation timing, not a simulation change.
    indices = np.rint((np.arange(250) + 1) / 25 / 0.02).astype(int) - 1
    assert indices[0] == 1 and indices[-1] == 499
    qa = {
        "video_sha256": hashlib.sha256(args.video.read_bytes()).hexdigest(),
        "decoded_frames": count,
        "dimensions": metadata["size"],
        "fps": metadata["fps"],
        "duration_s": metadata["duration"],
        "inspection_frames": entries,
        "decode_passed": True,
        "visual_inspection": "pending",
    }
    args.video.with_suffix(".qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(json.dumps(qa, indent=2))


if __name__ == "__main__":
    main()
