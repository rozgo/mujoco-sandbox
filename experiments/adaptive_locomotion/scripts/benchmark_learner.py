"""Bounded, matched CPU/MPS/CUDA continuations of the v1 training recipe."""

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import threading
import time

import numpy as np
import torch

from adaptive_locomotion.bodies import ROOT
from adaptive_locomotion.evaluate import lane_command, make_case
from adaptive_locomotion.limb_loss import LOSS_CASES
from adaptive_locomotion.train import load_checkpoint, train

ASSETS = ROOT / "assets/locomotion/benchmarks"
PARENT = ROOT / "assets/locomotion/checkpoints/limb_rear_overlap_iter100_seed2.pt"
PARENT_SHA = "aefb52f219090ccdfe48cbe9b217106520669f7999cc25a6a251dffc4a346825"
REFERENCE = ROOT / "assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt"
MOTION = ASSETS / "healthy_motion_v1.npz"
MOTION_SHA = "c8cc59cc4b61e6268aa8da94aa9c5f05b684cd6c53d19a0d1d27691b64b06733"
BANK = ASSETS / "v1_observation_bank.npz"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bank():
    torch.set_num_threads(1)
    policy, _ = load_checkpoint(PARENT)
    bank = {"obs": [], "context": [], "extra": []}
    with torch.no_grad():
        for case in LOSS_CASES:
            env = make_case(case, trials=4, seed=9211, timestep=0.0005)
            try:
                for step in range(200):
                    lane_command(env)
                    obs = env.obs()
                    if step >= 50 and step % 5 == 0:
                        bank["obs"].append(obs.copy())
                        bank["context"].append(env.context.copy())
                        bank["extra"].append(
                            np.c_[env.vel, env.pos[:, 2]].astype(np.float32)
                        )
                    action, _ = policy(torch.tensor(obs))
                    env.step(action.numpy())
            finally:
                env.close()
    np.savez_compressed(BANK, **{k: np.concatenate(v) for k, v in bank.items()})
    print("BANK", digest(BANK), flush=True)


def backend_probe(device):
    torch.set_num_threads(1)
    policy, _ = load_checkpoint(PARENT)
    with np.load(BANK, allow_pickle=False) as bank:
        args = [torch.tensor(bank[k]) for k in ("obs", "context", "extra")]
    other = copy.deepcopy(policy).to(device)
    a, va = policy(args[0], args[1], critic_extra=args[2])
    b, vb = other(
        args[0].to(device), args[1].to(device), critic_extra=args[2].to(device)
    )
    action_error = float((a.detach() - b.detach().cpu()).abs().max())
    value_error = float((va.detach() - vb.detach().cpu()).abs().max())
    (a.square().mean() + va.square().mean()).backward()
    (b.square().mean() + vb.square().mean()).backward()
    pairs = list(zip(policy.parameters(), other.parameters(), strict=True))
    scale = max(float(p.grad.abs().max()) for p, _ in pairs if p.grad is not None)
    error = max(
        float((p.grad - q.grad.cpu()).abs().max())
        for p, q in pairs
        if p.grad is not None
    )
    report = {
        "device": device,
        "states": len(args[0]),
        "states_sha256": digest(BANK),
        "maximum_action_error": action_error,
        "maximum_value_error": value_error,
        "maximum_gradient_error": error,
        "gradient_relative_to_global_max": error / max(scale, 1e-8),
    }
    report["passed"] = (
        action_error < 1e-4 and value_error < 1e-3 and error / max(scale, 1e-8) < 1e-4
    )
    return report


def gpu_sample():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        ).strip()
        return dict(
            zip(
                ("utilization_percent", "memory_mib", "power_w"),
                map(float, output.split(",")),
            )
        )
    except FileNotFoundError, subprocess.SubprocessError, ValueError:
        return None


def run(label, device):
    assert digest(PARENT) == PARENT_SHA and digest(MOTION) == MOTION_SHA
    out = ROOT / "outputs/locomotion/learner_comparison" / label
    out.mkdir(parents=True, exist_ok=True)
    if (out / "training.json").exists():
        raise ValueError("Preserve the existing attempt; use another label")
    probe = backend_probe(device)
    (out / "backend_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
    if not probe["passed"]:
        raise ValueError(f"Backend preflight failed: {probe}")
    samples = []
    stop = threading.Event()
    start = time.perf_counter()

    def monitor():
        while not stop.is_set():
            sample = gpu_sample()
            samples.append(
                {
                    "elapsed_s": time.perf_counter() - start,
                    "gpu": sample,
                    "load_average": list(os.getloadavg()),
                }
            )
            stop.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        train(
            output=out,
            seconds=90,
            allowance=2000,
            extension_reason="Matched 90-second learner backend benchmark from the identical archived parent; official v1 is frozen.",
            seed=2,
            num_envs=512,
            mode="blind",
            resume=PARENT,
            bodies="all",
            terrain="flat",
            device=device,
            threads=16,
            epochs=4,
            horizon=24,
            reward_profile="walk",
            support_weight=2,
            stride_weight=1,
            balance_weight=5,
            body_motion_weight=1,
            damage_flight_weight=3,
            visible_step_weight=2,
            rear_overlap_weight=3,
            reference=REFERENCE,
            reference_reward_weight=0.5,
            reference_loss_weight=0.5,
            damage_healthy_reward_weight=2,
            healthy_style_source="motion_sequence",
            healthy_motion_path=MOTION,
            learning_rate=0.0001,
            limb_stage="consolidate",
        )
    finally:
        stop.set()
        thread.join(timeout=4)
        meta = {
            "label": label,
            "learner_device": device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "architecture": platform.machine(),
            "system": platform.system(),
            "gpu_name": torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else None,
            "cuda_runtime": torch.version.cuda,
            "parent_sha256": digest(PARENT),
            "healthy_motion_sha256": digest(MOTION),
            "process_wall_s_including_setup": time.perf_counter() - start,
            "telemetry": samples,
            "external_gpu_workload_observed_before_benchmark": platform.system() == "Linux",
        }
        (out / "benchmark.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--make-bank", action="store_true")
    parser.add_argument("--label")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"))
    args = parser.parse_args()
    if args.make_bank:
        make_bank()
    elif args.label and args.device:
        run(args.label, args.device)
    else:
        parser.error("Provide --make-bank or both --label and --device")
