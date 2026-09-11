"""Preserve final benchmark checkpoints and public-safe provenance."""

import argparse
import hashlib
import json

import torch

from adaptive_locomotion.bodies import ROOT

LABELS = ("mac_mps", "desktop_cpu", "desktop_cuda", "warp_bvh_4096")


def state_digest(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def curate(label):
    run = ROOT / "outputs/locomotion/learner_comparison" / label
    raw = run / "policy.pt"
    saved = torch.load(raw, weights_only=False, map_location="cpu")
    parent_path = (
        ROOT / "assets/locomotion/checkpoints/limb_rear_overlap_iter100_seed2.pt"
    )
    parent = torch.load(parent_path, weights_only=False, map_location="cpu")
    initial = torch.load(run / "initial.pt", weights_only=False, map_location="cpu")
    assert state_digest(initial["state"]) == state_digest(parent["state"])
    saved["parent"] = str(parent_path.relative_to(ROOT))
    saved["source_run"] = str(run.relative_to(ROOT))
    target = ROOT / f"assets/locomotion/checkpoints/learner_{label}_90s_seed2.pt"
    torch.save(saved, target)
    restored = torch.load(target, weights_only=False, map_location="cpu")
    assert state_digest(saved["state"]) == state_digest(restored["state"])
    docs = ROOT / "docs/locomotion/gpu_comparison"
    docs.mkdir(exist_ok=True)
    for name in ("training", "benchmark", "backend_probe"):
        report = json.loads((run / f"{name}.json").read_text())
        (docs / f"{label}_{name}.json").write_text(json.dumps(report, indent=2) + "\n")
    report = {
        "label": label,
        "checkpoint": str(target.relative_to(ROOT)),
        "checkpoint_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "raw_checkpoint_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "initial_tensor_sha256": state_digest(initial["state"]),
        "parent_tensor_sha256": state_digest(parent["state"]),
        "final_tensor_sha256": state_digest(saved["state"]),
        "curation": "Only parent path/source-run metadata normalized; all tensors verified identical",
        "training_seconds": saved["training_seconds"],
        "cumulative_training_seconds": saved["cumulative_training_seconds"],
        "transitions": saved["transitions"],
        "iterations": saved["iterations"],
    }
    (docs / f"{label}_checkpoint.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", nargs="+", choices=LABELS)
    for label in parser.parse_args().labels:
        curate(label)
