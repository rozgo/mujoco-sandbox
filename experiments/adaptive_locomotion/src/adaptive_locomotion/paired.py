"""Declared paired-damage splits; geometry is compiled only at stage setup."""

from .bodies import LEGS, BodySpec

TRAIN_PAIRS = ((0, 3), (1, 2), (0, 1), (2, 3))
HELDOUT_PAIRS = ((0, 2), (1, 3))


def paired_body(name, legs, lengths):
    calf = [1.0] * 4
    for leg, length in zip(legs, lengths, strict=True):
        calf[leg] = length
    return BodySpec(name, tuple(calf))


def training_pairs(level):
    lengths = {"mild": (0.90, 0.85), "hard": (0.75, 0.65)}[level]
    return [
        paired_body(
            f"pair_{LEGS[a].lower()}_{LEGS[b].lower()}_{level}", (a, b), lengths
        )
        for a, b in TRAIN_PAIRS
    ]


PAIR_CASES = {body.name: (body, "flat", None) for body in training_pairs("hard")}
PAIR_CASES.update(
    {
        "pair_holdout_left": (
            paired_body("pair_holdout_left", (0, 2), (0.78, 0.68)),
            "flat",
            None,
        ),
        "pair_holdout_right": (
            paired_body("pair_holdout_right", (1, 3), (0.78, 0.68)),
            "flat",
            None,
        ),
        "pair_holdout_lengths": (
            paired_body("pair_holdout_lengths", (0, 3), (0.68, 0.78)),
            "flat",
            None,
        ),
    }
)
