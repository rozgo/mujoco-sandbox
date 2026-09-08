"""Agents receive local observations and delivered bytes, never MuJoCo state."""

import heapq
import math
from dataclasses import dataclass, replace

import numpy as np

from .radio import Message, intersects


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


@dataclass(frozen=True)
class Detection:
    marker: int
    x: float
    y: float
    observed: float


@dataclass(frozen=True)
class Observation:
    time: float
    x: float
    y: float
    yaw: float
    speed: float
    detections: tuple
    nearby: tuple  # local proximity detections (x,y), not global peer positions


class FacilityMap:
    """Public static obstacle map; inspection marker positions are not included."""

    def __init__(self, buildings):
        self.rects = [(x, y, hx + 0.65, hy + 0.65) for x, y, hx, hy, *_ in buildings]
        self.rects.append(
            (7, 6.5, 1.5, 1.1)
        )  # avoid the optional ramp during this mission
        self.free = {
            (x, y)
            for x in range(-18, 19)
            for y in range(-14, 15)
            if not any(
                abs(x * 0.5 - a) <= hx and abs(y * 0.5 - b) <= hy
                for a, b, hx, hy in self.rects
            )
        }

    def clear(self, a, b):
        return not any(intersects(a, b, r) for r in self.rects)

    def path(self, start, goal):
        source = min(
            self.free,
            key=lambda p: (p[0] * 0.5 - start[0]) ** 2 + (p[1] * 0.5 - start[1]) ** 2,
        )
        target = min(
            self.free,
            key=lambda p: (p[0] * 0.5 - goal[0]) ** 2 + (p[1] * 0.5 - goal[1]) ** 2,
        )
        queue = [(0.0, source)]
        cost = {source: 0.0}
        parent = {}
        while queue:
            _, p = heapq.heappop(queue)
            if p == target:
                break
            for dx, dy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                q = (p[0] + dx, p[1] + dy)
                if q not in self.free or not self.clear(
                    np.array(p) * 0.5, np.array(q) * 0.5
                ):
                    continue
                value = cost[p] + math.hypot(dx, dy)
                if value < cost.get(q, math.inf):
                    cost[q] = value
                    parent[q] = p
                    heapq.heappush(queue, (value + math.dist(q, target), q))
        if target not in cost:
            return [tuple(start)]
        nodes = [target]
        while nodes[-1] != source:
            nodes.append(parent[nodes[-1]])
        points = [tuple(start)] + [tuple(np.array(n) * 0.5) for n in reversed(nodes)]
        if self.clear(points[-1], goal):
            points.append(tuple(goal))
        # String-pull the grid path for feasible steering arcs.
        result = []
        i = 0
        while i < len(points) - 1:
            j = len(points) - 1
            while j > i + 1 and not self.clear(points[i], points[j]):
                j -= 1
            result.append(points[j])
            i = j
        return result


PATROLS = [
    [(7, -5.5), (7, -3), (-5.5, -3), (-5.5, -5.5)],
    [(0, -3.8), (7, -5.5), (7, 4.8), (-6, 5.5), (-6, -5.5)],
    [(3, -0.5), (7, 1), (7, 5.3), (-5.5, 5.3), (-5.5, -1)],
    [(2, 0.0), (7, 4.8), (1, 5.5), (-6, 1), (-6, -5.5)],
    [(-3.8, 4.8), (1, 5.5), (7, 4.8), (7, -5.5), (-6, -5.5)],
    [(-2, 5.5), (7, 4.8), (7, -5.5), (-6, -5.5), (-6, 5.5)],
]


class RoverAgent:
    def __init__(self, index, facility_map, seed=0):
        self.id = index
        self.inspector = index % 2 == 1
        self.map = facility_map
        self.belief = {}
        self.completed = set()
        self.claims = {}
        self.received = set()
        self.sequence = 0
        self.advertised = {}
        self.patrol = 0
        self.target = None
        self.target_source = "patrol"
        self.path = []
        self.destination = None
        self.last_plan = -10.0
        self.events = []
        self.inspections = []
        self.outgoing = []
        self.inspection_started = None
        self.inspecting = None
        self.next_send = index * 0.18
        self.rng = np.random.default_rng(seed * 31 + index)
        self.last_claim = -100.0
        self.relayed = set()
        self.avoid_until = -1.0
        self.avoid_target = None
        self.reverse_until = -1.0
        self.last_gossip = -100.0
        self.gossip_index = 0

    def emit(self, kind, marker, x, y, time):
        self.sequence += 1
        self.outgoing.append(Message(kind, marker, self.id, self.sequence, time, x, y))

    def update(self, obs, inbox):
        self.outgoing = []
        now = obs.time
        for payload, sender, arrival, packet_id in inbox:
            message = Message.decode(payload)
            if message.key in self.received:
                continue
            self.received.add(message.key)
            if message.kind == "seen":
                old = self.belief.get(message.marker)
                if old is None or message.observed > old["observed"]:
                    self.belief[message.marker] = {
                        "x": message.x,
                        "y": message.y,
                        "observed": message.observed,
                        "source": f"radio:r{sender}",
                        "packet": packet_id,
                        "received": arrival,
                    }
            elif message.kind == "done":
                self.completed.add(message.marker)
            elif message.kind == "claim":
                entries = self.claims.setdefault(message.marker, {})
                old = entries.get(message.origin)
                if old is None or message.observed > old[0]:
                    entries[message.origin] = (message.observed, message.x, message.y)
            if message.ttl > 0 and not self.inspector:
                self.outgoing.append(replace(message, ttl=message.ttl - 1))
        for sighting in obs.detections:
            self.belief[sighting.marker] = {
                "x": sighting.x,
                "y": sighting.y,
                "observed": now,
                "source": "local",
                "packet": None,
                "received": now,
            }
            if now - self.advertised.get(sighting.marker, -100) > 8:
                self.emit(
                    "done" if sighting.marker in self.completed else "seen",
                    sighting.marker,
                    sighting.x,
                    sighting.y,
                    now,
                )
                self.advertised[sighting.marker] = now
        # Rotate compact completion summaries so a lost report can heal later.
        if self.completed and now - self.last_gossip > 5.5:
            marker = sorted(self.completed)[self.gossip_index % len(self.completed)]
            if marker in self.belief:
                b = self.belief[marker]
                self.emit("done", marker, b["x"], b["y"], now)
            self.gossip_index += 1
            self.last_gossip = now
        old_target = self.target

        def better_claim(marker):
            b = self.belief[marker]
            own = math.hypot(b["x"] - obs.x, b["y"] - obs.y)
            for claimant, (at, x, y) in self.claims.get(marker, {}).items():
                if claimant == self.id or now - at >= 15:
                    continue
                other = math.hypot(b["x"] - x, b["y"] - y)
                if other + 0.6 < own or (abs(other - own) < 0.6 and claimant < self.id):
                    return True
            return False

        if self.target in self.completed:
            self.target = None
        if self.target is not None and better_claim(self.target):
            self.target = None
        if self.inspector and self.target is None and now > 0.8:
            choices = []
            for marker, b in self.belief.items():
                if marker in self.completed or better_claim(marker):
                    continue
                choices.append((math.hypot(b["x"] - obs.x, b["y"] - obs.y), marker))
            if choices:
                self.target = min(choices)[1]
        if self.target != old_target and self.target is not None:
            b = self.belief[self.target]
            self.target_source = b["source"]
            self.events.append(
                {
                    "time": now,
                    "rover": self.id,
                    "event": "route_changed",
                    "marker": self.target,
                    "source": b["source"],
                    "packet": b["packet"],
                    "received": b["received"],
                }
            )
            self.path = []
            self.destination = None
        if self.target is not None:
            b = self.belief[self.target]
            goal = (b["x"], b["y"])
            if now - self.last_claim > 6:
                self.emit("claim", self.target, obs.x, obs.y, now)
                self.last_claim = now
            visible = {v.marker for v in obs.detections}
            # A nose-mounted camera can pass over a marker during avoidance.
            # Back up physically until it is in front again; proximity alone
            # must never count as a successful visual inspection.
            if math.dist((obs.x, obs.y), goal) < 0.8 and self.target not in visible:
                self.inspecting = None
                return -0.18, 0.0, self.outgoing
            if math.dist((obs.x, obs.y), goal) < 0.8 and self.target in visible:
                if self.inspecting != self.target:
                    self.inspecting = self.target
                    self.inspection_started = now
                if now - self.inspection_started >= 1.0:
                    marker = self.target
                    self.completed.add(marker)
                    self.inspections.append(
                        {"time": now, "rover": self.id, "marker": marker}
                    )
                    self.emit("done", marker, *goal, now)
                    self.events.append(
                        {
                            "time": now,
                            "rover": self.id,
                            "event": "inspected",
                            "marker": marker,
                            "source": self.target_source,
                        }
                    )
                    self.target = None
                    self.inspecting = None
                    self.destination = None
                return 0.0, 0.0, self.outgoing
            self.inspecting = None
        else:
            goal = PATROLS[self.id][self.patrol % len(PATROLS[self.id])]
            if math.dist((obs.x, obs.y), goal) < 0.8:
                self.patrol += 1
                goal = PATROLS[self.id][self.patrol % len(PATROLS[self.id])]
        if self.destination != goal or now - self.last_plan > 3:
            self.path = self.map.path((obs.x, obs.y), goal)
            self.destination = goal
            self.last_plan = now
        while len(self.path) > 1 and math.dist((obs.x, obs.y), self.path[0]) < 0.65:
            self.path.pop(0)
        waypoint = self.path[0] if self.path else goal
        # Pass on the right using only nearby range detections. Persist the detour
        # long enough to avoid oscillating between the route and the obstacle.
        if now >= self.avoid_until:
            for px, py in obs.nearby:
                delta = wrap(math.atan2(py - obs.y, px - obs.x) - obs.yaw)
                if math.hypot(px - obs.x, py - obs.y) < 1.8 and abs(delta) < 0.75:
                    f = np.array([math.cos(obs.yaw), math.sin(obs.yaw)])
                    right = np.array([f[1], -f[0]])
                    candidate = np.array([obs.x, obs.y]) + right * 1.0 + f * 0.25
                    if self.map.clear((obs.x, obs.y), candidate):
                        self.avoid_target = candidate
                        self.avoid_until = now + 3.5
                        if math.hypot(px - obs.x, py - obs.y) < 0.95:
                            self.reverse_until = now + 1.5
                        break
        if now < self.avoid_until:
            waypoint = self.avoid_target
        distance = math.dist((obs.x, obs.y), waypoint)
        error = wrap(math.atan2(waypoint[1] - obs.y, waypoint[0] - obs.x) - obs.yaw)
        speed = min(0.65, 0.25 + distance * 0.5)
        steer = math.atan2(2 * 0.38 * math.sin(error), max(0.6, min(distance, 1.0)))
        steer = float(np.clip(steer, -0.55, 0.55))
        if abs(error) > 1.35:
            speed = -0.23
            steer = -math.copysign(0.55, error)
        else:
            speed *= max(0.3, math.cos(error))
        if now < self.reverse_until:
            speed = -0.22
            steer = 0.45
        for px, py in obs.nearby:
            delta = wrap(math.atan2(py - obs.y, px - obs.x) - obs.yaw)
            if math.hypot(px - obs.x, py - obs.y) < 0.72 and (
                (speed > 0 and abs(delta) < 0.8) or (speed < 0 and abs(delta) > 2.35)
            ):
                speed = 0.0
        return speed, steer, self.outgoing
