"""Maintained adaptive-walking recipe shared by CPU and GPU physics."""

from pathlib import Path

from .bodies import ROOT
from .train import train


def adaptive_walking(
    output,
    seconds=90,
    seed=2,
    physics_backend="mjbatch",
    device="auto",
    num_envs=None,
    minibatch_size=3072,
    max_iterations=None,
    resume=None,
    learning_rate=0.0001,
):
    """Continue the approved dog with unchanged architecture and gait rewards.

    This is the final adaptive-walking stage, not the historical curriculum from
    scratch. Explicit world/minibatch counts support matched backend comparisons.
    """
    output = Path(output)
    if (output / "initial.pt").exists() or (output / "training.json").exists():
        raise ValueError(
            "Existing learning run must be preserved; choose another output"
        )
    if num_envs is None:
        num_envs = 4096 if physics_backend == "warp" else 512
    # Each invocation is bounded independently. Ancestry remains recorded in
    # checkpoints; increasing that accounting allowance is not extra wall time.
    if not 0 < seconds <= 300:
        raise ValueError("Use a learning round between zero and five minutes")
    resume = (
        Path(resume)
        if resume
        else ROOT / "assets/locomotion/checkpoints/adaptive_dog_v1.pt"
    )
    import torch

    parent = torch.load(resume, weights_only=False, map_location="cpu")
    ancestry = parent.get(
        "cumulative_training_seconds", parent.get("training_seconds", 0)
    )
    return train(
        output=output,
        seconds=seconds,
        seed=seed,
        num_envs=num_envs,
        mode="blind",
        resume=resume,
        device=device,
        physics_backend=physics_backend,
        minibatch_size=minibatch_size,
        max_iterations=max_iterations,
        allowance=ancestry + seconds + 1,
        extension_reason="Bounded continuation of the approved adaptive-walking recipe with an explicit physics backend.",
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
        reference=ROOT / "assets/locomotion/checkpoints/healthy_balanced_405s_seed2.pt",
        reference_reward_weight=0.5,
        reference_loss_weight=0.5,
        damage_healthy_reward_weight=2,
        healthy_style_source="motion_sequence",
        healthy_motion_path=ROOT / "assets/locomotion/benchmarks/healthy_motion_v1.npz",
        learning_rate=learning_rate,
        limb_stage="consolidate",
    )
