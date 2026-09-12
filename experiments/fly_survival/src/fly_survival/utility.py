"""Tiny trainable utility system. Scores choose skills; outcomes supply reward."""

from dataclasses import dataclass, field

import numpy as np

ACTIONS = ("explore", "forage", "drink", "rest", "refuge", "evade")
FEATURES = (
    "bias",
    "hunger",
    "thirst",
    "fatigue",
    "heat",
    "threat",
    "food_odor",
    "water_odor",
    "shade",
    "crowding",
    "neural_escape",
    "neural_steer",
)


def handcrafted_weights():
    w = np.zeros((6, len(FEATURES)), np.float32)
    w[:, 0] = [0.0, -1.3, -1.3, -2.4, -2.5, -3.0]
    w[0, 5] = -3
    w[1, [1, 4, 5, 6, 9]] = [4, -1.5, -4, 1.2, -0.4]
    w[2, [2, 4, 5, 7, 9]] = [4, -1, -4, 1.2, -0.4]
    w[3, [3, 4, 5, 8]] = [5, -3, -4, 0.3]
    w[4, [4, 5, 8, 10]] = [3, 1.5, 1.0, 1.0]
    w[5, [4, 5, 10]] = [1, 6, 1.5]
    return w


@dataclass
class Thinker:
    weights: np.ndarray = field(default_factory=handcrafted_weights)
    minimum_commitment: float = 0.18
    hysteresis: float = 0.045
    current: int = 0
    elapsed: float = 0.0
    scores: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def tick(self, features, dt):
        logits = np.clip(self.weights @ np.asarray(features), -30, 30)
        self.scores = 1 / (1 + np.exp(-logits))
        winner = int(np.argmax(self.scores))
        self.elapsed += dt
        urgent = winner == 5 and features[5] > 0.7
        can_change = urgent or (
            self.elapsed >= self.minimum_commitment
            and self.scores[winner] > self.scores[self.current] + self.hysteresis
        )
        if can_change and winner != self.current:
            self.current = winner
            self.elapsed = 0.0
        return self.current
