"""Portable run evidence without credentials, host names or private local paths."""

import hashlib
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def utc_now():
    return datetime.now(UTC).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence():
    project = Path(__file__).resolve().parents[2]
    repository = project.parents[1]
    files = [
        project / "pyproject.toml",
        project / "uv.lock",
        *sorted((project / "src").rglob("*.py")),
    ]
    file_hashes = {str(path.relative_to(project)): sha256(path) for path in files}
    packages = {}
    for package in (
        "torch",
        "numpy",
        "scipy",
        "mujoco",
        "mujoco-warp",
        "flybody",
        "dm-control",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "experiments/embodied_fly"],
        cwd=repository,
        text=True,
    )
    return {
        "schema": "embodied-fly-evidence-v1",
        "captured_utc": utc_now(),
        "source_commit": commit,
        "package_has_uncommitted_changes": bool(status.strip()),
        "source_file_sha256": file_hashes,
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "architecture": platform.machine(),
            "packages": packages,
        },
    }
