"""Explicit illustrative needs, exposure and finite shared resources."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Needs:
    energy: float = 0.7
    hydration: float = 0.7
    fatigue: float = 0.1
    heat: float = 0.0
    health: float = 1.0
    alive: bool = True
    food_intake: float = 0.0
    water_intake: float = 0.0
    reward: float = 0.0

    def advance(self, dt, speed, exposure, food=0.0, water=0.0, impact=0.0):
        if not self.alive:
            return 0.0
        moving = min(abs(speed) / 12, 1.5)
        # Compressed physiology: needs can become consequential within a short
        # physical episode. These are illustrative units, not measured fly metabolism.
        self.energy -= dt * (0.035 + 0.02 * moving)
        self.hydration -= dt * (0.03 + 0.015 * self.heat)
        self.fatigue += dt * (0.11 * moving - 0.24 * max(0, 1 - moving * 3))
        self.heat += dt * (1.7 * exposure - 0.75 * self.heat)
        self.energy += food
        self.hydration += water
        self.food_intake += food
        self.water_intake += water
        injury = dt * (
            max(0, self.heat - 0.75) * 0.9
            + 0.18 * (self.energy <= 0)
            + 0.2 * (self.hydration <= 0)
        )
        injury += max(0, impact) / 0.22
        injury = min(self.health, injury)
        self.health -= injury
        self.energy = float(np.clip(self.energy, 0, 1))
        self.hydration = float(np.clip(self.hydration, 0, 1))
        self.fatigue = float(np.clip(self.fatigue, 0, 1))
        self.heat = float(np.clip(self.heat, 0, 2))
        self.health = float(np.clip(self.health, 0, 1))
        self.alive = self.health > 0
        r = (
            dt
            * (
                float(self.alive)
                + 0.8 * min(self.energy, self.hydration)
                - 0.12 * self.fatigue
                - 0.3 * self.heat
            )
            - 4 * injury
        )
        self.reward += r
        return r

    def values(self):
        return [
            self.energy,
            self.hydration,
            self.fatigue,
            min(self.heat, 1.0),
            self.health,
        ]


class Resources:
    def __init__(self, quantities=(2.5, 3.5, 5.0)):
        self.remaining = np.asarray(quantities, dtype=float).copy()
        self.initial = self.remaining.copy()

    def consume(self, requests):
        """Allocate concurrent requests proportionally, avoiding fly-index advantage."""
        requests = np.maximum(np.asarray(requests, dtype=float), 0)
        total = requests.sum(axis=0)
        ratio = np.minimum(1, self.remaining / np.maximum(total, 1e-12))
        actual = requests * ratio
        self.remaining = np.maximum(0, self.remaining - actual.sum(axis=0))
        return actual
