"""Prepare a source clip, submit once, and resume a saved WaveSpeed task."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import imageio_ffmpeg
import wavespeed
from dotenv import load_dotenv
from PIL import Image

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "previews/locomotion/moving/moving_supports_v1.mp4"
SOURCE_HASH = "aa4290c5691e2938cecf18d47037e7ce7aa2d59567340293b729d5574d1f1c15"
MODEL = "bytedance/seedance-2.5/video-edit"
API = "https://api.wavespeed.ai/api/v3"
MODELS = (MODEL, "kwaivgi/kling-video-o3-pro/video-edit", "alibaba/wan-2.7/video-edit")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def ffmpeg(args):
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", *args],
        check=True,
    )


def inspect_video(path, frames_dir=None):
    """Decode every frame, optionally saving 0/1/2/3/4 s and the final frame."""
    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    meta = next(reader)
    count, selected, previous, last = 0, [], None, None
    changes = []
    try:
        for index, raw in enumerate(reader):
            count += 1
            img = Image.frombytes("RGB", meta["size"], raw)
            # Small per-frame change diagnostic; no motion-fidelity claim.
            small = img.resize((32, 18)).convert("L").tobytes()
            if previous is not None:
                changes.append(sum(abs(a - b) for a, b in zip(small, previous)) / 576)
            previous, last = small, img
            if frames_dir and index in {round(t * meta["fps"]) for t in range(5)}:
                frames_dir.mkdir(parents=True, exist_ok=True)
                p = frames_dir / f"frame_{index:04d}.png"
                img.save(p)
                selected.append(str(p.relative_to(ROOT)))
        if frames_dir and last:
            p = frames_dir / f"frame_{count - 1:04d}.png"
            last.save(p)
            if str(p.relative_to(ROOT)) not in selected:
                selected.append(str(p.relative_to(ROOT)))
    finally:
        reader.close()
    if not count:
        raise ValueError("Video has no decodable frames")
    return {
        "sha256": sha(path),
        "dimensions": list(meta["size"]),
        "fps": meta["fps"],
        "frames": count,
        "duration_s": count / meta["fps"],
        "container_duration_s": meta["duration"],
        "mean_small_frame_change": sum(changes) / len(changes) if changes else 0,
        "inspection_frames": selected,
        "complete_decode": True,
    }


def prepare(output):
    if output.exists():
        raise FileExistsError("Source clip already exists; use a new path")
    if sha(SOURCE) != SOURCE_HASH:
        raise ValueError("Accepted source movie hash changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        [
            "-i",
            str(SOURCE),
            "-ss",
            "52",
            "-t",
            "5",
            "-vf",
            "crop=1280:900:0:130",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "16",
            "-preset",
            "fast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    info = inspect_video(output, ROOT / "build/video_styling/source")
    assert info["frames"] == 125 and info["duration_s"] == 5
    info.update(
        original_video=str(SOURCE.relative_to(ROOT)),
        original_sha256=SOURCE_HASH,
        source_interval_s=[52, 57],
        crop_xywh=[0, 130, 1280, 900],
        playback_speed=1,
        source_camera="third person",
        source_physics="MuJoCo Warp recorded states",
        checkpoint_sha256="5c349861f504bdb43e29e3a4351998cc22ed50bacfde86d3772f3684bc8b7879",
    )
    save(output.with_suffix(".json"), info)
    return info


def client():
    load_dotenv(ROOT / ".env", override=False)
    key = os.environ.get("WAVESPEED_API_KEY")
    if not key or key.startswith("replace-"):
        raise ValueError("Set WAVESPEED_API_KEY in the ignored repository-root .env")
    return wavespeed.Client(api_key=key, max_retries=0, max_connection_retries=3)


def submit(video, prompt_path, job_dir, model=MODEL):
    """Reserve a durable job record before the single billable submission."""
    if model not in MODELS:
        raise ValueError("Model not in the inspected video-edit shortlist")
    metadata = inspect_video(video)
    if not 4 <= metadata["duration_s"] <= 10:
        raise ValueError("This trial workflow accepts only 4–10 second clips")
    if not prompt_path.read_text().strip():
        raise ValueError("An editing prompt is required")
    job_dir.mkdir(parents=True, exist_ok=True)
    journal = job_dir / "job.json"
    # Exclusive create also protects against concurrent duplicate submissions.
    with journal.open("x") as handle:
        json.dump({"status": "preparing", "created_at": now()}, handle)
    record = {
        "created_at": now(),
        "status": "preparing",
        "model": model,
        "source": str(video.resolve().relative_to(ROOT)),
        "source_sha256": sha(video),
        "source_metadata": metadata,
        "prompt": prompt_path.read_text().strip(),
        "sdk_version": importlib.metadata.version("wavespeed"),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    }
    c = client()
    started = time.perf_counter()
    source_url = c.upload(str(video))
    payload = {"video": source_url, "prompt": record["prompt"]}
    if model == MODEL:
        payload.update(resolution="720p", generate_audio=False)
        record["estimated_cost_usd"] = round(math.ceil(metadata["duration_s"]) * 0.44, 2)
    elif model == "kwaivgi/kling-video-o3-pro/video-edit":
        payload.update(keep_original_sound=False, shot_type="customize")
        record["estimated_cost_usd"] = round(math.ceil(metadata["duration_s"]) * 0.168, 3)
    elif model == "alibaba/wan-2.7/video-edit":
        payload.update(resolution="720p", duration=0, audio_setting="origin", seed=9411)
        record["estimated_cost_usd"] = round(math.ceil(metadata["duration_s"]) * 0.20, 2)
    record.update(status="submission_started", upload_seconds=time.perf_counter() - started)
    record["parameters"] = {k: v for k, v in payload.items() if k != "video"}
    save(journal, record)
    # SDK 2.0.2 run() returns outputs only. Submit through the documented REST
    # endpoint to persist the task ID immediately; SDK handles uploads and polling.
    # Never retry this POST: a disconnected request may already be billed.
    response = httpx.post(
        f"{API}/{model}",
        json=payload,
        headers={"Authorization": f"Bearer {c.api_key}"},
        timeout=60,
    )
    response.raise_for_status()
    body = response.json()
    data = body.get("data", body)
    if not data.get("id"):
        raise RuntimeError(
            "Submission returned no ID; inspect provider history, do not retry"
        )
    record.update(
        task_id=data["id"], status=data.get("status", "created"), submitted_at=now()
    )
    save(journal, record)
    print(
        json.dumps({"status": record["status"], "task_id": record["task_id"]}), flush=True
    )
    return record


def collect(job_dir, output, wait_seconds=0):
    journal = job_dir / "job.json"
    record = json.loads(journal.read_text())
    if not record.get("task_id"):
        raise ValueError(
            "No task ID saved; inspect provider history before submitting again"
        )
    c, started = client(), time.monotonic()
    while True:
        body = c.get_result(record["task_id"], timeout=30)
        data = body.get("data", body)
        status = data.get("status")
        # Full provider responses/CDN URLs remain in ignored local job storage.
        save(job_dir / "provider_result.json", body)
        record.update(status=status, checked_at=now())
        save(journal, record)
        print(json.dumps({"status": status, "task_id": record["task_id"]}), flush=True)
        if status == "completed":
            urls = data.get("outputs", [])
            if len(urls) != 1:
                raise ValueError("Expected exactly one generated video")
            if output.exists():
                raise FileExistsError("Generated output exists; use its saved manifest")
            output.parent.mkdir(parents=True, exist_ok=True)
            temp = output.with_suffix(".download")
            with httpx.stream("GET", urls[0], follow_redirects=True, timeout=60) as r:
                r.raise_for_status()
                with temp.open("wb") as f:
                    for block in r.iter_bytes():
                        f.write(block)
            temp.replace(output)
            info = inspect_video(output, ROOT / "build/video_styling/generated")
            info.update(record)
            info.update(
                generated_media=True,
                physics_evidence=False,
                visual_inspection="pending",
                actual_cost_usd=None,
                cost_note="Published estimate; actual charge not returned by this endpoint",
            )
            save(output.with_suffix(".json"), info)
            return info
        if status in {"failed", "cancelled", "timeout", "deleted"}:
            raise RuntimeError(
                f"Provider task {status}; local response saved. No replacement created"
            )
        if time.monotonic() - started >= wait_seconds:
            return record
        time.sleep(min(10, max(0, wait_seconds - (time.monotonic() - started))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("submit")
    a.add_argument("--video", type=Path, required=True)
    a.add_argument(
        "--prompt", type=Path, default=ROOT / "docs/video_styling/brushed_metal.txt"
    )
    a.add_argument("--job-dir", type=Path, required=True)
    a.add_argument("--model", default=MODEL)
    a = sub.add_parser("collect")
    a.add_argument("--job-dir", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--wait-seconds", type=float, default=0)
    args = vars(p.parse_args())
    command = args.pop("command")
    if command == "prepare":
        print(json.dumps(prepare(**args), indent=2))
    elif command == "submit":
        args["prompt_path"] = args.pop("prompt")
        submit(**args)
    else:
        collect(**args)


if __name__ == "__main__":
    main()
