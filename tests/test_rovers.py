"""Physical motion, sensor isolation, and delivered-packet causal contracts."""

import math

import mujoco
import numpy as np
import pytest

from sixlegs.rovers.agent import FacilityMap, Observation, RoverAgent
from sixlegs.rovers.radio import LoRaConfig, LoRaNetwork, Message
from sixlegs.rovers.scene import BUILDINGS, load_scene
from sixlegs.rovers.simulation import RoverSimulation, run


def test_packet_roundtrip_and_semtech_airtime():
    packet = Message("seen", 3, 2, 42, 12.345, -4.25, 2.75)
    assert len(packet.encode()) == 14
    assert Message.decode(packet.encode()) == packet
    assert LoRaConfig().airtime(14) == pytest.approx(0.046336)
    assert LoRaConfig().airtime(20) == pytest.approx(0.056576)
    assert LoRaConfig(spreading_factor=12).airtime(14) == pytest.approx(1.155072)


def test_overlap_half_duplex_airtime_budget_and_disabled_radio():
    positions = np.array([[i, 0.0, 0.6] for i in range(6)])
    network = LoRaNetwork()
    payload = Message("seen", 0, 0, 1, 0.0, 0.0, 0.0).encode()
    for sender in (0, 1):
        network.enqueue(sender, payload, 0.0)
        network.next_tx[sender] = 0.0
    network.advance(0.0, positions)
    for t in np.arange(0.004, 0.06, 0.004):
        network.advance(float(t), positions)
    assert network.counts["half_duplex"] == 2
    assert network.counts["collision"] > 0
    assert not network.receive(0) and not network.receive(1)
    # R2 captures the nearer sender, which exceeds the 6 dB threshold.
    assert network.receive(2)
    for _ in range(16):
        network.enqueue(0, payload, 0.06)
    for t in np.arange(0.06, 3.0, 0.004):
        network.advance(float(t), positions)
    assert np.all(network.airtime_used <= 0.2 + 0.08 * network.last_time + 1e-9)
    off = LoRaNetwork("disabled")
    off.enqueue(0, payload, 0.0)
    off.advance(10.0, positions)
    assert off.counts["transmitted"] == 0
    assert not any(off.inboxes)
    degraded = LoRaNetwork("degraded")
    assert not degraded.radio_on(0, 8.0)
    assert degraded.radio_on(0, 28.0)
    assert not degraded.radio_on(3, 36.0)
    assert degraded.radio_on(3, 64.0)


def test_delivered_evidence_changes_only_recipient_and_stale_data_cannot_replace_it():
    a, b = [RoverAgent(1, FacilityMap(BUILDINGS)) for _ in range(2)]
    obs = Observation(1.0, -7.0, -4.0, 0.0, 0.0, (), ())
    report = Message("seen", 2, 0, 1, 0.5, 3.2, -5.0)
    a.update(obs, [(report.encode(), 0, 0.75, 123)])
    b.update(obs, [])
    assert a.target == 2 and a.events[-1]["packet"] == 123
    assert b.target is None and not b.belief
    stale = Message("seen", 2, 2, 2, 0.2, 8.0, 8.0)
    a.update(obs, [(stale.encode(), 2, 0.9, 124)])
    assert a.belief[2]["x"] == 3.2
    assert a.belief[2]["packet"] == 123


def test_local_sensor_uses_range_fov_and_physical_occlusion():
    sim = RoverSimulation()
    assert 0 in {d.marker for d in sim.observation(0).detections}
    assert 3 not in {d.marker for d in sim.observation(0).detections}
    # Place a physical wall between the front sensor and initially visible M0.
    m, d = sim.model, sim.data
    m.body_pos[m.body("building_0").id] = [-6.3, -5.35, 0.0]
    m.geom_size[m.geom("building_0").id] = [0.1, 0.3, 0.5]
    m.geom_pos[m.geom("building_0").id] = [0.0, 0.0, 0.5]
    mujoco.mj_forward(m, d)
    assert 0 not in {v.marker for v in sim.observation(0).detections}


def test_steering_clearance_and_contact_driven_turn():
    m, d = load_scene()
    assert m.body_subtreemass[m.body("r0").id] == pytest.approx(4.18)
    assert m.nu == 36 and m.ncam == 9
    assert np.all(m.actuator_forcelimited)
    for angle in np.linspace(-0.65, 0.65, 13):
        for side in ("fl", "fr"):
            d.joint(f"r0_{side}_steer").qpos[0] = angle
        mujoco.mj_forward(m, d)
        for contact in d.contact:
            if m.geom("floor").id not in contact.geom:
                assert contact.dist >= -1e-5
    sim = RoverSimulation()
    before = sim.data.qpos.copy()
    sim.actuate(0, 0.5, 0.4)
    assert np.array_equal(before, sim.data.qpos)
    for _ in range(750):
        mujoco.mj_step(sim.model, sim.data)
    yaw = math.atan2(sim.data.body("r0").xmat[3], sim.data.body("r0").xmat[0])
    assert yaw > 0.4
    assert np.linalg.norm(sim.data.body("r0").xpos[:2] - before[:2]) > 0.5
    assert not sim.data.warning.number.any()
    assert not sim.data.xfrc_applied.any() and not sim.data.qfrc_applied.any()
    assert np.all(
        abs(sim.data.actuator_force) <= sim.model.actuator_forcerange[:, 1] + 1e-8
    )


@pytest.mark.parametrize("case", ["healthy", "degraded", "disabled"])
def test_complete_inspection_and_causal_log(case):
    report = run(case, duration=150, record=False)
    assert report["complete"] and report["unique_inspections"] == 8
    assert not any(report["warnings"])
    assert report["max_actuator_fraction"] <= 1.000001
    if case == "disabled":
        assert report["radio"]["counts"].get("delivered", 0) == 0
        assert not report["remote_route_changes"]
    else:
        assert report["remote_route_changes"]
        for event in report["remote_route_changes"]:
            assert event["received"] <= event["time"]
            assert (
                event["first_local_sighting"] is None
                or event["time"] < event["first_local_sighting"]
            )


def test_optional_real_reticulum_frame_delivery():
    pytest.importorskip("RNS")
    from sixlegs.rovers.reticulum import ReticulumBridge

    bridge = ReticulumBridge(count=2)
    try:
        payload = Message("seen", 1, 0, 1, 0.0, 1.0, 2.0).encode()
        frames = bridge.encode(0, payload)
        assert len(frames) == 1 and len(frames[0]) > len(payload)
        assert bridge.decode(1, frames[0]) == [payload]
        # PLAIN destinations can deliver duplicates; the application deduplicates.
        assert bridge.decode(1, frames[0]) == [payload]
    finally:
        bridge.close()
    assert not bridge.processes
