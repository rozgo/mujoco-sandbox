import json
from types import SimpleNamespace

import numpy as np
import torch
from scipy import sparse

from embodied_fly.train import train


def test_one_actor_trains_both_demonstration_clocks_and_reports_each(tmp_path):
    graph = tmp_path / "graph"
    graph.mkdir()
    matrix = sparse.csr_matrix(
        (np.ones(4, np.float32), ([2, 3, 4, 5], [0, 1, 2, 3])), shape=(6, 6)
    )
    sparse.save_npz(graph / "weights.npz", matrix)
    np.savez(
        graph / "brain.npz",
        superclass=np.array(
            ["vnc_sensory"] * 2 + ["descending_neuron"] * 2 + ["vnc_motor"] * 2
        ),
    )
    rng = np.random.default_rng(12)
    datasets = []
    for rate in (500, 5000):
        path = tmp_path / str(rate)
        path.mkdir()
        reports = []
        for episode in range(4):
            np.savez(
                path / f"episode_{episode:03d}.npz",
                observation=rng.normal(size=(12, 4)).astype(np.float32),
                action=rng.uniform(-1, 1, size=(12, 78)).astype(np.float32),
                activity=np.ones(12, np.int64),
            )
            reports.append({"episode": episode, "failure": None, "final_upright": 1})
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "control_hz": rate,
                    "episodes": reports,
                    "environment": {
                        "actuation": [
                            {"name": "wing_" + str(i) if i < 6 else "leg_" + str(i)}
                            for i in range(78)
                        ]
                    },
                }
            )
        )
        datasets.append(path)
    output = tmp_path / "training"
    train(
        SimpleNamespace(
            graph=graph,
            data=datasets[0],
            additional_data=[datasets[1]],
            resume=None,
            reset_fraction=0.25,
            output=output,
            device="cpu",
            seconds=0.3,
            worlds=2,
            sequence=2,
            burnin=2,
            internal_steps=4,
            lr=1e-4,
            seed=31,
            freeze_core=False,
            wing_loss_weight=1.0,
        )
    )
    report = json.loads((output / "report.json").read_text())
    assert set(report["clock_updates"]) == {"1.0", "0.1"}
    assert min(report["clock_updates"].values()) > 0
    assert sum(report["clock_updates"].values()) == report["updates"]
    assert set(report["final_validation_by_neural_time_scale"]) == {"1.0", "0.1"}
    checkpoint = torch.load(output / "actor.pt", weights_only=True)
    assert checkpoint["observation_size"] == 4 and checkpoint["action_size"] == 78
    assert len(checkpoint["all_data_manifest_sha256"]) == 2
