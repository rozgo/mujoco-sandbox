"""Promote selected local runs to portable, public-safe checkpoint artifacts."""

import hashlib
import json
from pathlib import Path

import torch

from .bodies import ROOT


def curate(names):
    target = ROOT / "assets/locomotion/checkpoints"
    reports = ROOT / "docs/locomotion/runs"
    target.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    for name in names:
        source = ROOT / "outputs/locomotion" / name
        original = source / "policy.pt"
        saved = torch.load(original, map_location="cpu", weights_only=False)
        if saved["parent"]:
            parent_path = Path(saved["parent"])
            parent_name = (
                parent_path.parent.name
                if parent_path.name == "policy.pt"
                else parent_path.stem
            )
            saved["parent"] = f"assets/locomotion/checkpoints/{parent_name}.pt"
        checkpoint = target / f"{name}.pt"
        torch.save(saved, checkpoint)
        report = json.loads((source / "training.json").read_text())
        report["original_checkpoint_sha256"] = report["checkpoint_sha256"]
        report["checkpoint_sha256"] = hashlib.sha256(
            checkpoint.read_bytes()
        ).hexdigest()
        report["checkpoint"] = str(checkpoint.relative_to(ROOT))
        report["parent"] = saved["parent"]
        report["curation_note"] = (
            "Only checkpoint parent paths were made repository-relative; learned tensors unchanged."
        )
        if "contact_profile" not in report:
            report["contact_profile"] = (
                "legacy_soft; final evaluation separately uses corrected firm contacts"
            )
        (reports / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        if (source / "evaluation.json").exists():
            evaluation = json.loads((source / "evaluation.json").read_text())
            evaluation["checkpoint"] = str(checkpoint.relative_to(ROOT))
            (reports / f"{name}_development.json").write_text(
                json.dumps(evaluation, indent=2) + "\n"
            )
        print(report["checkpoint"], flush=True)
