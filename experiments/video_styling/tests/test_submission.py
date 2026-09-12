"""Paid API calls must never be silently duplicated by a retry."""

import json

import pytest

from video_styling import cli


def test_existing_job_cannot_submit_again(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "inspect_video", lambda _: {"duration_s": 5})
    monkeypatch.setattr(cli, "client", lambda: pytest.fail("Unexpected API access"))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Brushed metal")
    job = tmp_path / "job"
    job.mkdir()
    journal = job / "job.json"
    journal.write_text(json.dumps({"status": "submission_started"}))
    with pytest.raises(FileExistsError):
        cli.submit(tmp_path / "source.mp4", prompt, job)
    assert json.loads(journal.read_text())["status"] == "submission_started"


def test_unknown_model_rejected_before_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "client", lambda: pytest.fail("Unexpected API access"))
    with pytest.raises(ValueError, match="shortlist"):
        cli.submit(tmp_path / "source.mp4", tmp_path / "prompt.txt", tmp_path, "unknown")


def test_long_clip_rejected_before_upload(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "inspect_video", lambda _: {"duration_s": 60})
    monkeypatch.setattr(cli, "client", lambda: pytest.fail("Unexpected API access"))
    with pytest.raises(ValueError, match="second clips"):
        cli.submit(tmp_path / "source.mp4", tmp_path / "prompt.txt", tmp_path)


def test_resume_without_task_id_never_submits(tmp_path, monkeypatch):
    (tmp_path / "job.json").write_text(json.dumps({"status": "submission_started"}))
    monkeypatch.setattr(cli, "client", lambda: pytest.fail("Unexpected API access"))
    with pytest.raises(ValueError, match="No task ID"):
        cli.collect(tmp_path, tmp_path / "output.mp4")
