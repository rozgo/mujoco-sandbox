"""Two-stage prediction-only JEPA pilot, with physical validation selection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.train import synchronize
from embodied_fly.world_data import HISTORY, HORIZON
from embodied_fly.world_model import FlyWorldModel, physical_errors


class Corpus:
    def __init__(self, folder, device):
        self.folder, self.device = folder, device
        self.manifest = json.loads((folder / "manifest.json").read_text())
        self.records = self.manifest["records"]
        arrays = {key: [] for key in ("features", "metrics", "actions")}
        offset = 0
        for record in self.records:
            record["offset"] = offset
            for key, values in arrays.items():
                values.append(np.load(folder / record["name"] / f"{key}.npy"))
            offset += record["states"]
        self.arrays = {
            k: torch.as_tensor(np.concatenate(v), device=device) for k, v in arrays.items()
        }
        self.ranges = {
            split: [i for i, r in enumerate(self.records) if r["split"] == split]
            for split in ("train", "validation", "test")
        }
        if not all(self.ranges.values()):
            raise ValueError("All three episode-disjoint splits required")

    def select(self, split, count, rng):
        pool = self.ranges[split]
        families = sorted({self.records[i]["family"] for i in pool})
        ids, starts = [], []
        for _ in range(count):
            family = rng.choice(families)
            i = int(rng.choice([i for i in pool if self.records[i]["family"] == family]))
            maximum = self.records[i]["states"] - HORIZON
            # Deliberately retain early wing startup as well as established motion.
            start = int(rng.integers(min(maximum, 500) if rng.random() < 0.25 else maximum))
            ids.append(i)
            starts.append(start)
        return self.batch(ids, starts)

    def batch(self, ids, starts):
        offset = torch.tensor([self.records[i]["offset"] for i in ids], device=self.device)
        starts = torch.tensor(starts, device=self.device)
        local = starts[:, None] + torch.arange(-HISTORY + 1, HORIZON + 1, device=self.device)
        indices = offset[:, None] + local.clamp_min(0)
        seq = torch.cat((self.arrays["features"][indices], (local >= 0).unsqueeze(-1)), dim=-1)
        future = offset[:, None] + starts[:, None] + torch.arange(HORIZON, device=self.device)
        return {
            "sequence": seq,
            "actions": self.arrays["actions"][future],
            "initial": self.arrays["metrics"][offset + starts],
            "target": self.arrays["metrics"][future + 1],
        }

    def normalization(self):
        train = torch.cat(
            [
                self.arrays["features"][r["offset"] : r["offset"] + r["states"]]
                for r in self.records
                if r["split"] == "train"
            ]
        )
        mean = torch.cat((train.mean(0), train.new_zeros(1)))
        scale = torch.cat((train.std(0).clamp_min(1e-3), train.new_ones(1)))
        return mean, scale


@torch.no_grad()
def validate(model, batches, stage, directions):
    total, mse, spread = [], [], []
    pred, actual = [], []
    for batch in batches:
        if stage == "latent":
            loss, parts = model.latent_loss(batch["sequence"], batch["actions"], directions)
            total.append(loss.item())
            mse.append(parts["prediction"].item())
            spread.append(parts["latent_std"].item())
        else:
            prediction = model.metric_rollout(
                batch["sequence"][:, :HISTORY], batch["actions"], batch["initial"]
            )
            pred.append(prediction[:, [24, 49, 74, 99]].cpu().numpy())
            actual.append(batch["target"][:, [24, 49, 74, 99]].cpu().numpy())
    if stage == "latent":
        return {
            "selection": float(np.mean(total)),
            "prediction": float(np.mean(mse)),
            "latent_std": float(np.mean(spread)),
        }
    errors = physical_errors(np.concatenate(pred), np.concatenate(actual))
    return errors | {"selection": errors["velocity_mm_s"]}


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    corpus = Corpus(args.dataset, device)
    model = FlyWorldModel(corpus.arrays["features"].shape[-1] + 1).to(device)
    model.mean[:], model.scale[:] = corpus.normalization()
    rng = np.random.default_rng(args.seed)
    vrng = np.random.default_rng(391102)
    validation = [corpus.select("validation", 64, vrng) for _ in range(4)]
    generator = torch.Generator(device=device).manual_seed(391103)
    directions = torch.randn(64, 32, generator=generator, device=device)
    synchronize(device)
    setup_seconds = time.perf_counter() - begin
    report = {
        "provenance": evidence(),
        "started_utc": utc_now(),
        "config": vars(args) | {"dataset": args.dataset.name, "output": args.output.name},
        "model_config": model.config,
        "dataset_sha256": sha256(args.dataset / "manifest.json"),
        "physical_contract": corpus.manifest["physical_contract"],
        "setup_seconds": setup_seconds,
        "stages": {},
        "scope": "Only the physical world model learns; no MaleCNS or policy update",
    }
    for stage, steps in (("latent", args.latent_steps), ("prober", args.prober_steps)):
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(name.startswith("prober.") == (stage == "prober"))
        frozen = {
            k: p.detach().clone() for k, p in model.named_parameters() if not p.requires_grad
        }
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, steps, eta_min=0.0001
        )
        best, best_state, training_seconds, eval_seconds = float("inf"), None, 0.0, 0.0
        records = []
        for step in range(1, steps + 1):
            started = time.perf_counter()
            batch = corpus.select("train", args.batch, rng)
            optimizer.zero_grad(set_to_none=True)
            if stage == "latent":
                loss, _ = model.latent_loss(batch["sequence"], batch["actions"])
            else:
                with torch.no_grad():
                    latent = model.latent_rollout(
                        batch["sequence"][:, :HISTORY], batch["actions"]
                    )
                predicted = model.probe(latent, batch["initial"])
                loss = ((predicted - batch["target"]) / model.metric_scale).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite world-model objective")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            synchronize(device)
            training_seconds += time.perf_counter() - started
            if step % 100 == 0 or step == steps:
                started = time.perf_counter()
                result = validate(model, validation, stage, directions)
                if result["selection"] < best:
                    best = result["selection"]
                    best_state = {
                        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                    }
                    best_step = step
                synchronize(device)
                eval_seconds += time.perf_counter() - started
                record = {
                    "stage": stage,
                    "step": step,
                    "loss": loss.item(),
                    "validation": result,
                    "training_seconds": training_seconds,
                }
                records.append(record)
                print(json.dumps(record), flush=True)
        if not all(
            torch.equal(p, frozen[k]) for k, p in model.named_parameters() if k in frozen
        ):
            raise RuntimeError("Frozen stage parameters changed")
        model.load_state_dict(best_state)
        report["stages"][stage] = {
            "steps": steps,
            "best_step": best_step,
            "training_seconds": training_seconds,
            "evaluation_seconds": eval_seconds,
            "records": records,
            "presentations": steps * args.batch,
            "frozen_parameters_unchanged": True,
        }
        torch.save(
            {
                "config": model.config,
                "state_dict": model.state_dict(),
                "physical_contract": report["physical_contract"],
                "dataset_sha256": report["dataset_sha256"],
                "stage": stage,
                "seed": args.seed,
            },
            args.output / f"{stage}.pt",
        )
    report["completed_utc"] = utc_now()
    report["total_wall_seconds"] = time.perf_counter() - begin
    report["checkpoint_sha256"] = sha256(args.output / "prober.pt")
    report["parameters"] = sum(p.numel() for p in model.parameters())
    report["peak_cuda_bytes"] = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=391101)
    p.add_argument("--latent-steps", type=int, default=1000)
    p.add_argument("--prober-steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=128)
    run(p.parse_args())
