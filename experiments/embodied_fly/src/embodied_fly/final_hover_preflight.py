"""One explicit physical transfer, followed by PID and unchanged-weight flight."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from embodied_fly.evaluate import load_actor
from embodied_fly.full_body_guidance_evaluate import summarize
from embodied_fly.physical_contract import ARRAYS, physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import evaluate_hover


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device(args.device)
    actor, parent = load_actor(args.checkpoint, args.graph, device)
    old, new = environment(1, 1), environment(1, 1, reduced_coupling=True)
    if physical_contract(old.model) != parent["physical_contract"]:
        raise ValueError("Parent must use the retained agile-v5 plant")
    for name in ARRAYS:
        np.testing.assert_array_equal(getattr(old.model, name), getattr(new.model, name))
    migrated = dict(parent)
    migrated["physical_contract"] = physical_contract(new.model)
    migrated["physical_transfer"] = {
        "parent_sha256": sha256(args.checkpoint),
        "old_contract": parent["physical_contract"],
        "weights_changed": False,
        "optimizer_updates": 0,
        "new_force_law": "wing_motion_reduced_coupling_v6",
    }
    checkpoint = args.output / "transferred_parent.pt"
    torch.save(migrated, checkpoint)
    reports = {}
    for label, teacher in (("pid", True), ("transferred_parent", False)):
        reports[label] = evaluate_hover(
            actor, migrated, args.dataset, args.output / label, device, teacher=teacher
        )
        print(json.dumps({label: summarize(reports[label])}), flush=True)
    pid, transferred = (summarize(reports[k]) for k in ("pid", "transferred_parent"))
    pid_ok = pid["complete"] == 4 and pid["velocity_rms_mm_s"] < 1
    parent_ok = transferred["complete"] == 4
    report = {
        "provenance": evidence(),
        "completed_utc": utc_now(),
        "parent_sha256": sha256(args.checkpoint),
        "transferred_checkpoint_sha256": sha256(checkpoint),
        "physical_arrays_unchanged": ARRAYS,
        "old_physical_contract": parent["physical_contract"],
        "physical_contract": migrated["physical_contract"],
        "pid": pid,
        "transferred_parent": transferred,
        "pid_passed": pid_ok,
        "parent_passed": parent_ok,
        "proceed_on_v6": pid_ok and parent_ok,
        "fallback": "If either fails, retain the failure and train on unchanged v5",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "graph", "dataset", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
