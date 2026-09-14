"""Evaluate integrated acceleration residuals on the unchanged pilot tests."""

import argparse
import json
from pathlib import Path

import mujoco
import torch

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import sha256
from embodied_fly.world_analytic import AnalyticalFly
from embodied_fly.world_data import HISTORY
from embodied_fly.world_evaluate import Evaluator, run
from embodied_fly.world_residual_model import ResidualWorldModel
from embodied_fly.world_residual_physics import DifferentiableFly


class ResidualEvaluator(Evaluator):
    def __init__(self, args):
        self.args = args
        self.manifest = json.loads((args.dataset / "manifest.json").read_text())
        self.device = torch.device(args.device)
        saved = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        if saved["format"] != "integrated-acceleration-residual-v1":
            raise ValueError("Expected physics-integrated residual checkpoint")
        if saved["physical_contract"] != self.manifest["physical_contract"] or saved[
            "dataset_sha256"
        ] != sha256(args.dataset / "manifest.json"):
            raise ValueError("Training data or physics identity differs")
        self.net = ResidualWorldModel(**saved["config"]).to(self.device)
        self.net.load_state_dict(saved["state_dict"])
        self.net.eval()
        self.model = mujoco.MjModel.from_binary_path(
            str(args.root / "velocity_teacher_dataset_01/model.mjb")
        )
        if physical_contract(self.model) != saved["physical_contract"]:
            raise ValueError("Integrator model identity differs")
        self.analytic = AnalyticalFly(self.model)
        self.plant = DifferentiableFly(self.model).to(device=self.device, dtype=torch.float32)
        self.cache = {}

    @torch.no_grad()
    def predict_with_support(self, history, actions, initial):
        h, u, x = [
            torch.as_tensor(a, dtype=torch.float32, device=self.device)
            for a in (history, actions, initial)
        ]
        qpos = torch.cat((x[:, :3], h[:, HISTORY - 1, : self.model.nq - 3]), -1)
        inertia, inverse = self.plant.inertia(qpos.cpu().numpy(), x)
        latent = self.net.latent_rollout(h, u)
        residual = self.net.accelerations(latent)
        prediction, lift = self.plant(
            x, self.plant.wing_rollout(x, u), inertia, inverse, residual
        )
        return prediction.cpu().numpy(), lift.cpu().numpy()

    def predict(self, history, actions, initial):
        return self.predict_with_support(history, actions, initial)[0]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("outputs/embodied_fly"))
    p.add_argument("--device", default="cuda")
    run(p.parse_args(), evaluator_class=ResidualEvaluator)
