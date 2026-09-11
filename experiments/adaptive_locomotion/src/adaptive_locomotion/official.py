"""Frozen, user-accepted adaptive dog v1 demonstration."""

import hashlib

from .bodies import ROOT

CHECKPOINT = ROOT / "assets/locomotion/checkpoints/adaptive_dog_v1.pt"
CHECKPOINT_SHA256 = "174b3094eaf20233f151ff0ad19fc21c4612e8b1e7468e683c81ab8f980d846b"
TIMESTEP = 0.0005


def view_demo(case="whole_fr", seconds=0):
    from .record import view

    if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
        raise ValueError("Official checkpoint differs from v1; restore it with Git LFS")
    return view(
        checkpoint=CHECKPOINT,
        case=case,
        seconds=seconds,
        presentation="damage",
        timestep=TIMESTEP,
    )
