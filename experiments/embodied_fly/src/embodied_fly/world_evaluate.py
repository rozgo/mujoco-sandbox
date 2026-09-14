"""Held-out forecasts and matched physical interventions for fly world models."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from embodied_fly.physical_contract import physical_contract
from embodied_fly.provenance import evidence, sha256, utc_now
from embodied_fly.velocity_demonstrations import environment
from embodied_fly.velocity_hover import failures
from embodied_fly.world_analytic import AnalyticalFly, sampled_lift
from embodied_fly.world_data import DT, HISTORY, HORIZON, physical_metrics, window_arrays
from embodied_fly.world_model import FlyWorldModel, constant_velocity, physical_errors

HORIZONS = (25, 50, 75, 100)


def action_variants(actions, wings, bias=0.003, gain=0.03):
    variants = np.repeat(actions[None], 16, axis=0)
    names = ["unchanged"]
    for axis, actuator in enumerate(wings):
        variants[1 + 2 * axis, :, actuator] += bias
        variants[2 + 2 * axis, :, actuator] -= bias
        names.extend([f"bias_{axis}_plus", f"bias_{axis}_minus"])
    for row, multiplier in ((13, 1 + gain), (14, 1 - gain)):
        variants[row, :, np.asarray(wings)[[0, 3]]] *= multiplier
    names.extend(["sweep_gain_plus", "sweep_gain_minus", "unchanged_duplicate"])
    clipped = np.clip(variants, -1, 1)
    return clipped, names, int(np.count_nonzero(clipped != variants))


def effect_errors(predicted, actual, threshold):
    p, a = np.asarray(predicted).reshape(-1), np.asarray(actual).reshape(-1)
    mask = np.abs(a) >= threshold
    if not mask.any():
        return {
            "samples": len(a),
            "informative": 0,
            "sign_agreement": None,
            "normalized_rmse": None,
            "passed": False,
        }
    rmse = float(np.sqrt(np.mean((p[mask] - a[mask]) ** 2)))
    normalized = rmse / float(np.sqrt(np.mean(a[mask] ** 2)))
    sign = float(np.mean(np.sign(p[mask]) == np.sign(a[mask])))
    return {
        "samples": len(a),
        "informative": int(mask.sum()),
        "sign_agreement": sign,
        "normalized_rmse": normalized,
        "rmse": rmse,
        "mean_absolute_actual": float(np.abs(a[mask]).mean()),
        "passed": sign >= 0.8 and normalized < 0.5,
    }


class Evaluator:
    def __init__(self, args):
        self.args = args
        self.manifest = json.loads((args.dataset / "manifest.json").read_text())
        self.device = torch.device(args.device)
        saved = torch.load(args.checkpoint, map_location=self.device, weights_only=False)
        if saved["physical_contract"] != self.manifest["physical_contract"]:
            raise ValueError("World-model physical identity differs")
        if saved["dataset_sha256"] != sha256(args.dataset / "manifest.json"):
            raise ValueError("World-model training dataset differs")
        self.net = FlyWorldModel(**saved["config"]).to(self.device)
        self.net.load_state_dict(saved["state_dict"])
        self.net.eval()
        self.model = mujoco.MjModel.from_binary_path(
            str(args.root / "velocity_teacher_dataset_01/model.mjb")
        )
        if physical_contract(self.model) != saved["physical_contract"]:
            raise ValueError("Analytical model identity differs")
        self.analytic = AnalyticalFly(self.model)
        self.cache = {}

    def load(self, record):
        name = record["name"]
        if name not in self.cache:
            arrays = {
                k: np.load(self.args.dataset / name / f"{k}.npy")
                for k in ("features", "metrics", "actions")
            }
            source = self.args.root / record["source"]
            if sha256(source) != record["source_sha256"]:
                raise ValueError("Raw source differs from audited training data")
            with np.load(source) as z:
                arrays.update(
                    {
                        k: z[k][: record["states"]].copy()
                        for k in ("qpos", "qvel", "act", "ctrl")
                    }
                )
            self.cache[name] = arrays
        return self.cache[name]

    @torch.no_grad()
    def predict(self, history, actions, initial):
        tensors = [
            torch.as_tensor(a, dtype=torch.float32, device=self.device)
            for a in (history, actions, initial)
        ]
        return self.net.metric_rollout(*tensors).cpu().numpy()

    def forecasts(self):
        records, all_predictions, all_truth = (
            [],
            {k: [] for k in ("jepa", "analytical", "constant_velocity")},
            [],
        )
        for record in self.manifest["records"]:
            if record["split"] != "test":
                continue
            data = self.load(record)
            # Fixed windows selected before looking at any model predictions.
            starts = np.unique(np.linspace(0, record["states"] - HORIZON - 1, 24, dtype=int))
            batch = window_arrays(data["features"], data["metrics"], data["actions"], starts)
            predictions = {
                "jepa": self.predict(
                    batch["sequence"][:, :HISTORY], batch["actions"], batch["initial"]
                ),
                "constant_velocity": constant_velocity(batch["initial"], HORIZON),
            }
            predictions["analytical"], _ = self.analytic.predict(
                batch["initial"], batch["actions"], data["qpos"][starts]
            )
            result = {
                "source": record["name"],
                "family": record["family"],
                "windows": len(starts),
                "starts": starts.tolist(),
                "horizons": {},
            }
            for h in HORIZONS:
                result["horizons"][str(h * DT)] = {
                    k: physical_errors(v[:, h - 1], batch["target"][:, h - 1])
                    for k, v in predictions.items()
                }
            records.append(result)
            all_truth.append(batch["target"])
            for k, values in all_predictions.items():
                values.append(predictions[k])
        truth = np.concatenate(all_truth)
        aggregate = {
            str(h * DT): {
                k: physical_errors(np.concatenate(v)[:, h - 1], truth[:, h - 1])
                for k, v in all_predictions.items()
            }
            for h in HORIZONS
        }
        return {"cases": records, "windows": len(truth), "aggregate": aggregate}

    def interventions(self):
        env = environment(16, 16)
        if physical_contract(env.model) != self.manifest["physical_contract"]:
            raise ValueError("Intervention physics differs")
        force_samples = []
        original = env.wing_forces.advance

        def observe(*args):
            result = original(*args)
            force_samples.append(env.wing_forces.lift.copy() / env.body_weight)
            return result

        env.wing_forces.advance = observe
        results = []
        arrays_for_scores = []
        for record in self.manifest["records"]:
            if record["split"] != "test" or record["family"] not in (
                "pid",
                "student11",
                "student13",
            ):
                continue
            data = self.load(record)
            for seconds in (0.0, 0.2, 0.6, 2.0, 2.016, 4.0):
                start = round(seconds / DT)
                if start + HORIZON >= record["states"]:
                    continue
                actions, names, clipped = action_variants(
                    data["actions"][start : start + HORIZON],
                    self.manifest["layout"]["wing_action"],
                )
                batch = window_arrays(
                    data["features"], data["metrics"], data["actions"], [start]
                )
                history = np.repeat(batch["sequence"][:, :HISTORY], 16, axis=0)
                initial = np.repeat(batch["initial"], 16, axis=0)
                predictions = self.predict(history, actions, initial)
                states = {
                    k: np.repeat(data[k][start : start + 1], 16, axis=0)
                    for k in ("qpos", "qvel", "act", "ctrl")
                }
                analytical, analytic_lift = self.analytic.predict(
                    initial, actions, states["qpos"]
                )
                env.reset(np.arange(16), state=states)
                force_samples.clear()
                observed, valid = [], []
                alive = np.ones(16, bool)
                for step in range(HORIZON):
                    env.step(actions[:, step])
                    observed.append(
                        physical_metrics(
                            env.fields["qpos"], env.fields["qvel"], self.manifest["layout"]
                        )
                    )
                    alive &= ~failures(env)
                    valid.append(alive.copy())
                truth = np.stack(observed, axis=1)
                lift = np.stack(force_samples, axis=1)
                valid = np.stack(valid, axis=1)
                if lift.shape != (16, 2 * HORIZON):
                    raise RuntimeError("Missing 1 kHz force samples")
                replay = float(np.max(np.abs(truth[0, :, :3] - batch["target"][0, :, :3])))
                duplicate = float(np.max(np.abs(truth[0] - truth[15])))
                label = f"{record['name']}_{start:05d}"
                np.savez_compressed(
                    self.args.output / f"{label}.npz",
                    history=history[:1],
                    initial=initial,
                    actions=actions,
                    predicted=predictions,
                    analytical=analytical,
                    truth=truth,
                    lift=lift,
                    analytical_lift=analytic_lift,
                    valid=valid,
                )
                results.append(
                    {
                        "name": label,
                        "family": record["family"],
                        "start_seconds": seconds,
                        "phase": "cold" if seconds < 1 else "warm",
                        "variants": names,
                        "clipped_action_elements": clipped,
                        "replay_position_error_cm": replay,
                        "duplicate_metric_error": duplicate,
                        "failed_variants": int((~valid[:, -1]).sum()),
                        "passed_replay": replay <= 1e-4 and duplicate <= 1e-7,
                        "trace_sha256": sha256(self.args.output / f"{label}.npz"),
                    }
                )
                arrays_for_scores.append(
                    (
                        results[-1],
                        predictions,
                        analytical,
                        truth,
                        lift,
                        analytic_lift,
                        valid,
                        initial,
                    )
                )
                print(json.dumps(results[-1]), flush=True)
        scores = {}
        for phase in ("cold", "warm"):
            for h in HORIZONS:
                values = {
                    k: []
                    for k in (
                        "jepa",
                        "analytical",
                        "actual",
                        "jepa_lift",
                        "analytical_lift",
                        "actual_lift",
                    )
                }
                excluded = 0
                for (
                    row,
                    prediction,
                    analytical,
                    truth,
                    lift,
                    analytic_lift,
                    valid,
                    initial,
                ) in arrays_for_scores:
                    if row["phase"] != phase:
                        continue
                    mask = valid[1:15, h - 1] & valid[0, h - 1]
                    excluded += int((~mask).sum())
                    for key, x in (
                        ("jepa", prediction),
                        ("analytical", analytical),
                        ("actual", truth),
                    ):
                        values[key].append((x[1:15, h - 1, 3:6] - x[0, h - 1, 3:6])[mask] * 10)
                    # Prediction has 500 Hz endpoint wing samples. Include the
                    # known initial lift and trapezoid-average those samples.
                    pred_lift = sampled_lift(prediction[:, :h])
                    first = sampled_lift(initial[:, None])
                    pred_mean = (
                        np.concatenate((first, pred_lift), axis=1)[:, :-1]
                        + np.concatenate((first, pred_lift), axis=1)[:, 1:]
                    ).mean(1) / 2
                    for key, x in (
                        ("jepa_lift", pred_mean),
                        ("analytical_lift", analytic_lift[:, : 2 * h].mean(1)),
                        ("actual_lift", lift[:, : 2 * h].mean(1)),
                    ):
                        values[key].append((x[1:15] - x[0])[mask])
                cat = {k: np.concatenate(v) for k, v in values.items()}
                key = f"{phase}_{h * DT:g}s"
                scores[key] = {
                    "excluded_failed_pairs": excluded,
                    "velocity": {
                        model: {
                            axis: effect_errors(cat[model][:, i], cat["actual"][:, i], 0.2)
                            for i, axis in enumerate(("x", "y", "z"))
                        }
                        for model in ("jepa", "analytical")
                    },
                    "support": {
                        model: effect_errors(cat[model + "_lift"], cat["actual_lift"], 0.001)
                        for model in ("jepa", "analytical")
                    },
                }
        return {
            "cases": results,
            "scores": scores,
            "passed_replay": all(r["passed_replay"] for r in results),
            "scope": "Fixed candidate sequences branched from matched states; full MuJoCo steps, no acting brain or teacher feedback. Prediction-only causal intervention test.",
        }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    torch.set_num_threads(4)
    evaluation = Evaluator(args)
    setup = time.perf_counter() - begin
    started = time.perf_counter()
    forecasts = evaluation.forecasts()
    forecast_seconds = time.perf_counter() - started
    (args.output / "forecasts.json").write_text(json.dumps(forecasts, indent=2) + "\n")
    started = time.perf_counter()
    interventions = evaluation.interventions()
    intervention_seconds = time.perf_counter() - started
    report = {
        "provenance": evidence(),
        "checkpoint_sha256": sha256(args.checkpoint),
        "dataset_sha256": sha256(args.dataset / "manifest.json"),
        "physical_contract": evaluation.manifest["physical_contract"],
        "completed_utc": utc_now(),
        "setup_seconds": setup,
        "forecast_seconds": forecast_seconds,
        "intervention_seconds": intervention_seconds,
        "total_wall_seconds": time.perf_counter() - begin,
        "forecasts": forecasts,
        "interventions": interventions,
    }
    aggregate = list(forecasts["aggregate"].values())
    forecast_gate = all(
        v["jepa"]["velocity_mm_s"] < v["constant_velocity"]["velocity_mm_s"] for v in aggregate
    ) and np.mean([v["jepa"]["velocity_mm_s"] for v in aggregate]) <= np.mean(
        [v["analytical"]["velocity_mm_s"] for v in aggregate]
    )
    effect_gate = all(
        axis["passed"]
        for row in interventions["scores"].values()
        for axis in row["velocity"]["jepa"].values()
    )
    support_gate = all(
        row["support"]["jepa"]["passed"] for row in interventions["scores"].values()
    )
    report["gates"] = {
        "replay": interventions["passed_replay"],
        "forecast": bool(forecast_gate),
        "velocity_effects": effect_gate,
        "support_effects": support_gate,
    }
    report["go_for_policy_guidance"] = all(report["gates"].values())
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "gates": report["gates"],
                "go_for_policy_guidance": report["go_for_policy_guidance"],
                "wall_seconds": report["total_wall_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("outputs/embodied_fly"))
    p.add_argument("--device", default="cuda")
    run(p.parse_args())
