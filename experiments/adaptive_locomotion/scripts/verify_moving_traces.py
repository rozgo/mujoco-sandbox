"""Verify recorded physics independently of the video, on the capture host."""

import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.record import model_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    report = json.loads(args.report.read_text())
    counts = []
    for case in report["cases"]:
        path = ROOT / case["trajectory"]
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == case["trajectory_sha256"]
        )
        model = mujoco.MjModel.from_binary_path(str(ROOT / case["model_binary"]))
        assert model_hash(model) == case["model_mjb_sha256"]
        with np.load(path, allow_pickle=False) as trace:
            for key in trace.files:
                assert np.isfinite(trace[key]).all(), (case["key"], key)
            assert trace["qpos"].shape == (500, model.nq)
            assert trace["qvel"].shape == (500, model.nv)
            assert trace["ctrl"].shape == (500, model.nu)
            np.testing.assert_allclose(
                trace["time"], np.arange(1, 501) * 0.02, atol=1e-10
            )
            counts.append(len(trace["time"]))
    result = {
        "video_report": str(args.report),
        "verified_models": len(counts),
        "verified_trajectories": len(counts),
        "recorded_control_states": sum(counts),
        "all_hashes_match": True,
        "all_saved_states_finite": True,
        "complete_ten_second_traces": True,
        "simulation_or_training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
