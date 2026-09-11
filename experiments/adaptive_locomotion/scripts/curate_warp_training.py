"""Preserve every finished CPU/Warp verification run with public-safe provenance."""

import hashlib
import json

import torch
from curate_learner import state_digest

from adaptive_locomotion.bodies import ROOT


def main():
    source = ROOT / "outputs/locomotion/warp_training"
    docs = ROOT / "docs/locomotion/warp_training"
    assets = ROOT / "assets/locomotion/checkpoints/warp_training"
    assets.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.glob("*/training.json")):
        run = path.parent
        target = assets / f"{run.name}.pt"
        if target.exists():
            continue
        report = json.loads(path.read_text())
        assert not report["source_dirty"]
        raw = run / "policy.pt"
        saved = torch.load(raw, weights_only=False, map_location="cpu")
        initial = torch.load(run / "initial.pt", weights_only=False, map_location="cpu")
        parent = (
            ROOT / "assets/locomotion/checkpoints/limb_rear_overlap_iter100_seed2.pt"
        )
        reference = torch.load(parent, weights_only=False, map_location="cpu")
        assert state_digest(initial["state"]) == state_digest(reference["state"])
        saved["parent"] = str(parent.relative_to(ROOT))
        saved["source_run"] = str(run.relative_to(ROOT))
        torch.save(saved, target)
        restored = torch.load(target, weights_only=False, map_location="cpu")
        assert state_digest(restored["state"]) == state_digest(saved["state"])
        directory = docs / run.name
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("training", "benchmark", "backend_probe"):
            data = json.loads((run / f"{name}.json").read_text())
            (directory / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
        provenance = {
            "checkpoint": str(target.relative_to(ROOT)),
            "checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "raw_checkpoint_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "initial_tensor_sha256": state_digest(initial["state"]),
            "final_tensor_sha256": state_digest(saved["state"]),
            "curation": "Parent/source path metadata normalized; all tensors verified unchanged",
        }
        (directory / "checkpoint.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        print(json.dumps({"curated": run.name, **provenance}), flush=True)


if __name__ == "__main__":
    main()
