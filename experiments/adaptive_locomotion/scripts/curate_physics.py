"""Archive every completed physics timing attempt with concise validation metadata."""

import json
import shutil

from adaptive_locomotion.bodies import ROOT


def main():
    source = ROOT / "outputs/locomotion/warp_backend"
    target = ROOT / "docs/locomotion/warp_backend"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for path in sorted(source.glob("*.json")):
        result = json.loads(path.read_text())
        if "median_control_intervals_per_second" not in result:
            continue
        assert result["physics_timestep_s"] == 0.002
        assert result["full_collision_and_support_sensors"]
        shutil.copyfile(path, target / path.name)
        copied.append(path.name)
    result = {
        "reports": copied,
        "failed_initial_attempt": "Warp 1.17 occupancy query/CCD symbol mismatch; failed before stepping",
        "block256_attempt": "All nine physical tests passed; healthy-512 speed poor; remaining matrix intentionally stopped",
        "other_gpu_workload": "Left running; GPU telemetry includes other work",
        "official_v1_unchanged": True,
    }
    (target / "index.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
