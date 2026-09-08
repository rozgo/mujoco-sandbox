"""Discrete packet-level LoRa channel; propagation assumptions are explicit.

Airtime follows Semtech SX1276 section 4.1.1.7. This is not an RF waveform solver.
"""

import hashlib
import math
import struct
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

WIRE = struct.Struct("!BBBBHIhh")
KINDS = {"seen": 1, "claim": 2, "done": 3}


@dataclass(frozen=True)
class Message:
    kind: str
    marker: int
    origin: int
    sequence: int
    observed: float
    x: float
    y: float
    ttl: int = 1

    @property
    def key(self):
        return (self.origin, self.sequence)

    def encode(self):
        return WIRE.pack(
            KINDS[self.kind],
            self.marker,
            self.origin,
            self.ttl,
            self.sequence,
            round(self.observed * 1000),
            round(self.x * 100),
            round(self.y * 100),
        )

    @classmethod
    def decode(cls, payload):
        kind, marker, origin, ttl, sequence, t, x, y = WIRE.unpack(payload)
        return cls(
            {v: k for k, v in KINDS.items()}[kind],
            marker,
            origin,
            sequence,
            t / 1000,
            x / 100,
            y / 100,
            ttl,
        )


@dataclass(frozen=True)
class LoRaConfig:
    spreading_factor: int = 7
    bandwidth_hz: int = 125000
    coding_rate: int = 1  # 1 means 4/5
    preamble_symbols: int = 8
    tx_dbm: float = 14.0
    sensitivity_dbm: float = -123.0
    noise_dbm: float = -112.0
    required_snr_db: float = -7.5
    path_loss_at_1m_db: float = 31.7
    path_loss_exponent: float = 2.7
    obstacle_loss_db: float = 12.0
    airtime_fraction: float = 0.08  # engineering budget, not a regulatory claim

    def airtime(self, payload_bytes):
        sf = self.spreading_factor
        symbol = 2**sf / self.bandwidth_hz
        low_rate = int(symbol > 0.016)
        payload_symbols = 8 + max(
            math.ceil(
                (8 * payload_bytes - 4 * sf + 28 + 16) / (4 * (sf - 2 * low_rate))
            )
            * (self.coding_rate + 4),
            0,
        )
        return (self.preamble_symbols + 4.25 + payload_symbols) * symbol


def intersects(a, b, rect):
    """Segment vs a building's horizontal footprint, for assumed attenuation."""
    x, y, hx, hy = rect[:4]
    low, high = 0.0, 1.0
    for start, end, center, half in zip(a, b, (x, y), (hx, hy)):
        delta = end - start
        if abs(delta) < 1e-10:
            if not center - half <= start <= center + half:
                return False
        else:
            p, q = (center - half - start) / delta, (center + half - start) / delta
            low = max(low, min(p, q))
            high = min(high, max(p, q))
            if low > high:
                return False
    return True


@dataclass
class Transmission:
    id: int
    sender: int
    payload: bytes
    start: float
    end: float
    failures: dict = field(default_factory=dict)
    minimum_rssi: dict = field(default_factory=dict)


class LoRaNetwork:
    def __init__(self, case="healthy", seed=0, buildings=(), config=None):
        if case not in ("healthy", "degraded", "disabled"):
            raise ValueError(case)
        self.case = case
        self.seed = seed
        self.config = config or LoRaConfig()
        self.buildings = buildings
        self.rng = np.random.default_rng(seed)
        self.queues = [deque() for _ in range(6)]
        self.inboxes = [[] for _ in range(6)]
        self.next_tx = self.rng.uniform(0, 0.25, 6)
        self.tokens = np.full(6, 0.2)
        self.active = []
        self.events = []
        self.counts = Counter()
        self.sent_bytes = np.zeros(6, dtype=int)
        self.airtime_used = np.zeros(6)
        self.last_time = 0.0
        self.serial = 0
        self.latencies = []

    def enqueue(self, sender, payload, time):
        if len(payload) > 255:
            raise ValueError("LoRa payload exceeds 255 bytes")
        if len(self.queues[sender]) >= 16:
            self.counts["queue_overflow"] += 1
            return False
        if not self.queues[sender]:
            self.next_tx[sender] = max(
                self.next_tx[sender], time + self.rng.uniform(0.01, 0.2)
            )
        self.queues[sender].append((bytes(payload), time))
        self.counts["queued"] += 1
        return True

    def radio_on(self, node, time):
        if self.case == "disabled":
            return False
        return not (
            self.case == "degraded"
            and ((node == 0 and 8 <= time < 28) or (node == 3 and 36 <= time < 64))
        )

    def rssi(self, a, b):
        distance = max(float(np.linalg.norm(np.asarray(a[:2]) - b[:2])), 1.0)
        cfg = self.config
        walls = sum(intersects(a[:2], b[:2], rect) for rect in self.buildings)
        return (
            cfg.tx_dbm
            - cfg.path_loss_at_1m_db
            - 10 * cfg.path_loss_exponent * math.log10(distance)
            - walls * cfg.obstacle_loss_db
        )

    def _random(self, packet, receiver):
        key = f"{self.seed}:{packet.id}:{receiver}".encode()
        return (
            int.from_bytes(hashlib.blake2s(key, digest_size=8).digest(), "big") / 2**64
        )

    def advance(self, time, positions):
        dt = max(0.0, time - self.last_time)
        self.last_time = time
        self.tokens = np.minimum(0.2, self.tokens + dt * self.config.airtime_fraction)
        # Accumulate failures over the complete on-air interval, not just at arrival.
        for p in self.active:
            for receiver in range(6):
                if receiver == p.sender:
                    continue
                rssi = self.rssi(positions[p.sender], positions[receiver])
                p.minimum_rssi[receiver] = min(
                    p.minimum_rssi.get(receiver, math.inf), rssi
                )
                reason = None
                if not self.radio_on(p.sender, time) or not self.radio_on(
                    receiver, time
                ):
                    reason = "radio_off"
                elif any(q.sender == receiver for q in self.active):
                    reason = "half_duplex"
                elif rssi < self.config.sensitivity_dbm:
                    reason = "below_sensitivity"
                else:
                    noise = self.config.noise_dbm
                    if self.case == "degraded" and 18 <= time < 76:
                        noise = (
                            -54.0
                            if np.linalg.norm(np.asarray(positions[receiver][:2])) < 7
                            else -74.0
                        )
                    if rssi - noise < self.config.required_snr_db:
                        reason = "noise_floor"
                    for other in self.active:
                        if (
                            other.id != p.id
                            and self.rssi(positions[other.sender], positions[receiver])
                            > rssi - 6
                        ):
                            reason = "collision"
                if reason:
                    p.failures[receiver] = reason
        for p in list(self.active):
            if time + 1e-9 < p.end:
                continue
            self.active.remove(p)
            for receiver in range(6):
                if receiver == p.sender:
                    continue
                reason = p.failures.get(receiver)
                # Residual random loss is an explicit impairment, seeded for replay.
                loss = (
                    0.02
                    if self.case == "healthy"
                    else (0.3 if 18 <= time < 76 else 0.02)
                )
                if not reason and self._random(p, receiver) < loss:
                    reason = "injected_loss"
                event = {
                    "time": round(time, 4),
                    "packet": p.id,
                    "sender": p.sender,
                    "receiver": receiver,
                    "start": p.start,
                    "end": p.end,
                    "status": "lost" if reason else "delivered",
                    "reason": reason,
                    "bytes": len(p.payload),
                    "rssi_dbm": p.minimum_rssi.get(receiver),
                }
                self.events.append(event)
                self.counts[event["status"]] += 1
                if reason:
                    self.counts[reason] += 1
                else:
                    self.inboxes[receiver].append((p.payload, p.sender, time, p.id))
                    self.latencies.append(time - p.start)
        # A simple local carrier-sense approximation; simultaneous contenders still collide.
        contenders = []
        for node in range(6):
            if (
                not self.radio_on(node, time)
                or not self.queues[node]
                or time < self.next_tx[node]
            ):
                continue
            payload, _queued = self.queues[node][0]
            duration = self.config.airtime(len(payload))
            if self.tokens[node] < duration:
                continue
            busy = any(
                self.rssi(positions[p.sender], positions[node])
                > self.config.sensitivity_dbm
                for p in self.active
            )
            if busy:
                self.next_tx[node] = time + self.rng.uniform(0.03, 0.15)
                continue
            contenders.append((node, payload, duration))
        for node, payload, duration in contenders:
            self.queues[node].popleft()
            self.serial += 1
            self.active.append(
                Transmission(self.serial, node, payload, time, time + duration)
            )
            self.tokens[node] -= duration
            self.airtime_used[node] += duration
            self.sent_bytes[node] += len(payload)
            self.next_tx[node] = time + duration + self.rng.uniform(0.05, 0.2)
            self.counts["transmitted"] += 1

    def receive(self, node):
        messages = self.inboxes[node]
        self.inboxes[node] = []
        return messages

    def report(self):
        attempts = self.counts["delivered"] + self.counts["lost"]
        return {
            "case": self.case,
            "counts": dict(self.counts),
            "delivery_ratio": self.counts["delivered"] / max(1, attempts),
            "airtime_seconds_per_node": self.airtime_used.tolist(),
            "payload_bytes_per_node": self.sent_bytes.tolist(),
            "mean_delivery_latency_s": float(np.mean(self.latencies))
            if self.latencies
            else None,
        }
